# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import h5py
import shutil
import logging
import os
import re
from copy import deepcopy
import numpy as np
from src.utils import GoppaCodeParams

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class ICodeDataSource:

    def read_item(self, index):
        raise NotImplementedError()

    def __len__(self):
        return self.size


class DataH5FileSource(ICodeDataSource):
    def __init__(self, params, size, datapath, keys):
        if not (
            os.path.exists(datapath)
            and os.path.isfile(datapath)
            and datapath.endswith(".h5")
        ):
            raise ValueError(f"Data path {datapath} is not valid")

        self.datapath = datapath
        self.params = params
        if params.copy_data:
            self._copy_file(params)

        self._set_datakeys(keys)
        self._set_data_size(size)

    def read_item(self, index):
        with h5py.File(self.datapath, "r") as file:
            dpoint = {k: file[k][index] for k in self.datakeys}

        return dpoint

    def _set_datakeys(self, keys):
        self.datakeys = keys

    def _set_data_size(self, size):
        assert len(self.datakeys) > 0

        with h5py.File(self.datapath, "r") as file:
            sz = len(file[self.datakeys[0]])

        """
        if sz < size:
            logger.info(f"Have less samples than specified: {sz} vs {size}")
        elif sz > size:
            logger.info(
                f"Have more samples than specified: {sz} vs {size}, limiting to {size}"
            )
        """
        self.size = sz

    def _copy_file(self, params):
        local_path = os.path.join(
            f"./local/data_{params.exp_id}",
            os.path.basename(os.path.dirname(self.datapath)),
        )
        if not os.path.exists(local_path):
            os.makedirs(local_path, exist_ok=True)

        local_path = os.path.join(local_path, os.path.basename(self.datapath))
        logger.info(
            f"Copying {self.datapath} to {os.getcwd()} / {local_path} for faster access."
        )
        shutil.copy(self.datapath, local_path)
        self.datapath = local_path


class GoppaDataDirSource(DataH5FileSource):
    def __init__(self, params, size, keys, is_random=False):
        code = params.task.split("-")[-1]
        datapath = params.data_path
        if params.representation == "G" :
            subfolder = f"A_{code}_nmt_{params.code_len}_{params.m_alt}_{params.t_alt}"
        elif params.representation == "GT" :
            subfolder = f"AT_{code}_nmt_{params.code_len}_{params.m_alt}_{params.t_alt}"
        else :
            subfolder = f"{params.representation}_{code}_nmt_{params.code_len}_{params.m_alt}_{params.t_alt}"
        datapath = params.data_path.replace("<subfolder>", subfolder)

        datapath = datapath.replace("<codelen>", str(params.code_len))
        datapath = datapath.replace("<talt>", str(params.t_alt))
        datapath = datapath.replace("<malt>", str(params.m_alt))
        if is_random:
            datapath = datapath.replace(code, "random")
        # if is_random:
        #     datapath="./data/dataset_random_<codelen>_H5/AT_random_nmt_<codelen>_<malt>_<talt>/dataset_10K_xorfree.h5".replace("<codelen>", str(params.code_len)).replace("<talt>", str(params.t_alt)).replace("<malt>", str(params.m_alt))
        print(datapath)
        datapath = self.get_dataset_path(datapath)
        super().__init__(params, size, datapath, keys)

    def get_dataset_path(self, path):
        if os.path.exists(path):
            return path
        dir_name, _ = os.path.split(path)
        pattern = re.compile(r"^dataset_(\d+)M\.h5$", re.IGNORECASE)
        files = [f for f in os.listdir(dir_name) if pattern.match(f)]
        if not files:
            raise FileNotFoundError(f"No dataset files found in {dir_name}")
        files.sort(key=lambda f: int(pattern.match(f).group(1)), reverse=True)

        datapath = os.path.join(dir_name, files[0])
        logger.info(f"Dataset {path} not found, replacing path by {datapath}")
        return datapath


class MDPCDataDirSource(DataH5FileSource):
    def __init__(self, params, size, keys, is_random=False):
        assert params.data_path and params.random_data_path
        code = params.task.split("-")[-1]
        if is_random:
            datapath = params.random_data_path
        else:
            subfolder = f"{code}_rw_{params.r}_{params.w}"
            datapath = params.data_path.replace("<subfolder>", subfolder)

        assert os.path.exists(datapath), datapath
        super().__init__(params, size, datapath, keys)


class GoppaAllDataSource(ICodeDataSource):
    def __init__(self, params, size, keys, is_random=False):
        self.max_code_len = 0
        self.max_k = 0
        self.max_r = 0

        _sets = GoppaCodeParams(params.param_sets)
        self.sample_t_uniform = True

        assert len(_sets) > 0, "No valid parameter set"

        size_per_set = size // len(_sets)
        self.datasources = {}

        for _set in _sets:
            n, m, t = _set

            k = n - m * t
            r = m * t

            self.max_code_len = max(self.max_code_len, n)
            self.max_k = max(self.max_k, k)
            self.max_r = max(self.max_r, r)

            this_params = deepcopy(params)
            this_params.code_len = n
            this_params.m_alt = m
            this_params.t_alt = t

            datasource = GoppaDataDirSource(this_params, size_per_set, keys, is_random)
            self.datasources[_set] = datasource

        self._sets = _sets
        logger.info(
            f"Using {len(self._sets)} data parameters: {self._sets.param_sets_nested}"
        )
        self._set_data_size(size)

    def distribute_dataset_to_rank(self, params, sets):
        # shuffle but identical on all ranks
        rng = np.random.default_rng(seed=42)
        rng.shuffle(sets)

        if hasattr(params, "world_size") and params.world_size > 0:
            rank = params.local_rank
            assert rank >= 0
            world_size = params.world_size
            count = len(sets)
            # General case: Divide files among ranks
            base_workload = int(np.ceil(count / world_size))
            lacking = base_workload * world_size - count
            for i in range(lacking):
                sets.append(sets[i % len(sets)])
            assert len(sets) / world_size == base_workload

            # Divide the main chunk evenly
            start_idx = rank * base_workload
            end_idx = start_idx + base_workload
            assigned = sets[start_idx:end_idx]

            # shuffle differently for each rank
            np.random.shuffle(assigned)
            return assigned
        else:
            return sets

    def _set_data_size(self, size):
        self.size = 0
        for dsource in self.datasources.values():
            self.size += len(dsource)
        self.size = min(self.size, size)

        logger.info(f"Total dataset size is {self.size}")

    def read_item(self, index):
        if self.sample_t_uniform:
            _set = self._sets.sample(index)
        else:
            _set = self._sets[index % len(self._sets)]

        dsource = self.datasources[_set]
        dpoint = dsource.read_item((index // len(self._sets)) % len(dsource))
        dpoint["set"] = _set
        return dpoint


class QCDataDirSource(DataH5FileSource):
    def __init__(self, params, size, keys):
        code = params.task.split("-")[-1]
        datapath = params.data_path
        if "<subfolder>" in datapath:
            subfolder = f"{code}_rw_{params.r}_{params.w}"
            datapath = params.data_path.replace("<subfolder>", subfolder)

        super().__init__(params, size, datapath, keys)


# class GHDataSource(ICodeDataSource):
#     def __init__(self, params, size, keys, is_random=False):
#         assert params.data_path is not None
#         assert params.random_data_path is not None

#         self.params = params
#         self.keys = keys

#         datapath = params.random_data_path if is_random else params.data_path
#         datapath = datapath.replace("<codelen>", str(params.code_len))
#         datapath = datapath.replace("<malt>", str(params.m_alt))
#         datapath = datapath.replace("<talt>", str(params.t_alt))

#         datapath = self.get_dataset_path(datapath)

#         if not (
#             os.path.exists(datapath)
#             and os.path.isfile(datapath)
#             and datapath.endswith(".h5")
#         ):
#             raise ValueError(f"Data path {datapath} is not valid")

#         self.datapath = datapath

#         with h5py.File(self.datapath, "r") as f:
#             for k in self.keys:
#                 if k not in f:
#                     raise ValueError(f"Key {k} missing in {self.datapath}")

#             self.size = len(f[self.keys[0]])

#             g_shape = f["G"].shape
#             h_key = "H" if "H" in f else "GT"
#             h_shape = f[h_key].shape

#             if len(g_shape) != 3 or len(h_shape) != 3:
#                 raise ValueError(f"Invalid shapes in {self.datapath}: G={g_shape}, H={h_shape}")

#             _, self.k, self.n = g_shape
#             _, self.r, n2 = h_shape
#             if self.n != n2:
#                 raise ValueError(f"Inconsistent widths in {self.datapath}: G={g_shape}, H={h_shape}")

#         self.max_code_len = self.n
#         self.max_k = self.k
#         self.max_r = self.r

#     def get_dataset_path(self, path):
#         if os.path.exists(path):
#             return path

#         dir_name, _ = os.path.split(path)
#         if not os.path.isdir(dir_name):
#             raise FileNotFoundError(f"Directory not found: {dir_name}")

#         pattern = re.compile(r"^dataset_(\d+)M\.h5$", re.IGNORECASE)
#         files = [f for f in os.listdir(dir_name) if pattern.match(f)]

#         if not files:
#             raise FileNotFoundError(f"No dataset files found in {dir_name}")

#         files.sort(key=lambda f: int(pattern.match(f).group(1)), reverse=True)
#         datapath = os.path.join(dir_name, files[0])
#         logger.info(f"Dataset {path} not found, replacing path by {datapath}")
#         return datapath

#     def read_item(self, index):
#         with h5py.File(self.datapath, "r") as f:
#             return {k: f[k][index] for k in self.keys}

#     def __len__(self):
#         return self.size


# class GHAllDataSource(ICodeDataSource):
#     def __init__(self, params, size, keys, is_random=False):
#         self.max_code_len = 0
#         self.max_k = 0
#         self.max_r = 0
#         self.sample_t_uniform = True

#         _sets = GoppaCodeParams(params.param_sets)
#         assert len(_sets) > 0, "No valid parameter set"

#         size_per_set = size // len(_sets)
#         self.datasources = {}

#         for _set in _sets:
#             n, m, t = _set
#             this_params = deepcopy(params)
#             this_params.code_len = n
#             this_params.m_alt = m
#             this_params.t_alt = t

#             ds = GHDataSource(this_params, size_per_set, keys, is_random)
#             self.datasources[_set] = ds

#             self.max_code_len = max(self.max_code_len, ds.max_code_len)
#             self.max_k = max(self.max_k, ds.max_k)
#             self.max_r = max(self.max_r, ds.max_r)

#         self._sets = _sets
#         self._set_data_size(size)

#         logger.info(
#             f"Using {len(self._sets)} GH parameter sets: {self._sets.param_sets_nested}"
#         )

#     def _set_data_size(self, size):
#         self.size = sum(len(ds) for ds in self.datasources.values())
#         self.size = min(self.size, size)
#         logger.info(f"Total GH dataset size is {self.size}")

#     def __len__(self):
#         return self.size

#     def read_item(self, index):
#         if self.sample_t_uniform:
#             _set = self._sets.sample(index)
#         else:
#             _set = self._sets[index % len(self._sets)]

#         dsource = self.datasources[_set]
#         dpoint = dsource.read_item((index // len(self._sets)) % len(dsource))
#         dpoint["set"] = _set
#         return dpoint