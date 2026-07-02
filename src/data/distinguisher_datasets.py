# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import Any
import torch
from torch.utils.data import Dataset, random_split
import os
import io
import sys
from time import time
import numpy as np
from tqdm import tqdm
import logging
from torch.multiprocessing import Manager
import h5py
import shutil
from scipy.linalg import circulant
from src.utils import GoppaCodeParams
import copy

def _set_datasources(self, params, size):
    if params.data_path:
        _cls = GoppaAllDataSource
    else:
        logger.info("Data path not provided! generating data while training")
        _cls = GoppaAllGenerator

    self.datasources = [
        _cls(params, size // 2, [self.key], True),
        _cls(params, size // 2, [self.key], False),
    ]

    self.size = sum(map(len, self.datasources))

    self.max_n = max(d.max_code_len for d in self.datasources)
    self.max_k = max(d.max_k for d in self.datasources)
    self.max_r = max(d.max_r for d in self.datasources)

    self.code_len = params.code_len = self.max_n

from src.data.datasource import (
    GoppaAllDataSource,
    GoppaDataDirSource,
    MDPCDataDirSource,
    QCDataDirSource,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from src.data.generators import (
    AlternantCodeGenerator,
    CodeGenerator,
    GoppaAllGenerator,
    GoppaCodeGenerator,
    MDPCCodeGenerator,
    QCCodeGenerator,
    RandomQCGenerator,
)


class CodeDistDataset(Dataset):
    GENERATORS_CLS = dict(
        goppa=GoppaCodeGenerator,
        alternant=AlternantCodeGenerator,
        random=CodeGenerator,
        mdpc=MDPCCodeGenerator,
        qc=QCCodeGenerator,
    )

    @classmethod
    def create(cls, params, tokenizer):
        assert params.eval_samples > 0

        dataset_size = params.eval_samples + params.train_samples
        dataset = cls(tokenizer, params, dataset_size)

        train, test = len(dataset) - params.eval_samples, params.eval_samples

        train_dataset, test_dataset = random_split(
            dataset, [train, test], generator=torch.Generator().manual_seed(42)
        )
        return train_dataset, test_dataset

    def __init__(self, tokenizer, params, size):

        self.tokenizer = tokenizer
        self.size = size
        self.nworkers = params.workers
        self.params = params
        self.representation = params.representation
        self.code = params.task.split("-")[-1]
        params.code = self.code
        assert self.code in ["goppa", "mdpc", "qc", "alternant"]

        self.code_len = params.code_len

        if self.code in ["goppa", "alternant"]:
            params.k = params.code_len - params.m_alt * params.t_alt
        elif self.code in ["qc", "mdpc"]:
            params.k = params.r

        self.key = "G"
        if self.code == "qc":
            self.key = "h"

        self._set_datasources(params, size)
        self._set_processed_data_dims(params, tokenizer)

    def _set_datasources(self, params, size):
        if self.code in self.GENERATORS_CLS:
            self.code_generator = self.GENERATORS_CLS[self.code](params)
        else:
            raise NotImplementedError(f"task {params.task} not supported")

        self.datasources = [CodeGenerator(params), self.code_generator]

    def _set_processed_data_dims(self, params, tokenizer):
        dim_mul = tokenizer.data_dim_multiplier
        len_mul = tokenizer.data_len_multiplier
    
        n = params.code_len
        k = params.k
        r = n - k
        
        if params.representation in ["A","AT"]:
            params.model_input_dim = r * dim_mul
            params.model_input_len = k * len_mul
    
        elif params.representation in ["G", "GT","GD"]:
            params.model_input_dim = n * dim_mul
            params.model_input_len = k * len_mul
    
        elif params.representation in ["H", "HT"]:
            params.model_input_dim = n * dim_mul
            params.model_input_len = r * len_mul
    
        elif params.representation == "T":
            # à adapter selon ta convention exacte pour T
            params.model_input_dim = k * dim_mul
            params.model_input_len = r * len_mul

        elif params.representation == "GH":
            params.model_input_dim = n * dim_mul
            params.model_input_len = k * len_mul
            params.model_input_dim_H = n * dim_mul
            params.model_input_len_H = r * len_mul
        
        else:
            raise ValueError(f"Unsupported representation: {params.representation}")
    
        params.model_output_dim = 1
        params.model_output_len = 1
        params.output_vocab_size = 1

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """
        Open the HDF5 file and return a sample from the dataset at the specified index.
        Args:
        index (int): Index of the sample to return.
        """
        label = index % 2
        ix = index // 2
        data = self.datasources[label].read_item(ix)
        G = data[self.key]
        k = len(G)
        #if G.shape[-1] < self.params.code_len:
        if self.representation in ["G","GT","H","HT"]:
            G = np.concatenate([np.eye(k, dtype=G.dtype), G], axis=-1)
        if self.representation == "T":
            G=np.transpose(G)
        return dict(inputs=G, labels=np.array([label]))

    def collate_fn(self, elements):
        data = {
            key: [d[key] if key in d else None for d in elements] for key in elements[0]
        }

        data["inputs"] = self.tokenizer(data["inputs"])
        data["labels"] = self.tokenizer(data["labels"], labels=True)
        return data


class GoppaDistAllDataset(CodeDistDataset):
    def __init__(self, tokenizer, params, size):
        self.key = "G"
        assert (
            hasattr(params, "param_sets") and params.param_sets is not None
        ), "param_sets is required!"
        super().__init__(tokenizer, params, size)

    def _set_datasources(self, params, size):

        if params.data_path:
            _cls = GoppaAllDataSource
        else:
            logger.info(f"Data path not provided! generating data while training")
            _cls = GoppaAllGenerator
        self.datasources = [
            _cls(params, size // 2, [self.key], True),
            _cls(params, size // 2, [self.key], False),
        ]

        self.size = sum(map(len, self.datasources))
        self.code_len = params.code_len = max(
            dsource.max_code_len for dsource in self.datasources
        )
        self.max_n = max(d.max_code_len for d in self.datasources)
        self.max_k = max(d.max_k for d in self.datasources)
        self.max_r = max(d.max_r for d in self.datasources)

    def __getitem__(self, index):
        label = index % 2
        ix = index // 2
        data = self.datasources[label].read_item(ix)
    
        A = data[self.key]
        data.pop(self.key)
    
        k, r = A.shape
    
        if self.representation in ["A","AT"]:
            X = np.zeros((self.max_k, self.max_r), dtype=A.dtype)
            X[:k, :r] = A
    
        elif self.representation in {"G", "GT", "GD", "H", "HT"}:
            X = np.eye(self.max_n, dtype=A.dtype)
            X[:k, -r:] = A
            X[k:] = 0
    
        else:
            raise ValueError(f"Unsupported representation {self.representation}")
    
        return dict(inputs=X, labels=np.array([label]), **data)


class CodeDistH5Dataset(CodeDistDataset):
    SOURCE_CLS = dict(
        goppa=GoppaDataDirSource,
        alternant=GoppaDataDirSource,
        mdpc=MDPCDataDirSource,
    )

    def _set_datasources(self, params, size):
        _source_cls = self.SOURCE_CLS[params.code]
        self.datasources = [
            _source_cls(params, size // 2, [self.key], True),
            _source_cls(params, size // 2, [self.key], False),
        ]

        self.size = sum(map(len, self.datasources))


class QCDistDataset(CodeDistDataset):
    def __init__(self, tokenizer, params, size):
        self.key = "h"
        super().__init__(tokenizer, params, size)

    def _set_datasources(self, params, size):
        if params.data_path:
            logger.info(f"Loading data from path {params.data_path}")
            dsource = QCDataDirSource(params, size // 2, [self.key])
        else:
            logger.info(f"Generating data on the fly")
            dsource = QCCodeGenerator(params)

        self.datasources = [RandomQCGenerator(params, size // 2), dsource]

    def __getitem__(self, index):
        """
        Return a sample from the dataset at the specified index.
        Args:
        index (int): Index of the sample to return.
        """
        label = index % 2
        ix = index // 2
        data = self.datasources[label].read_item(ix)
        A = circulant(data[self.key])
        k = len(A)

        G = np.concatenate([np.eye(k, dtype=A.dtype), A], axis=-1)
        return dict(inputs=G, labels=np.array([label]))


class GoppaDistGenDataset(CodeDistDataset):

    def __init__(self, tokenizer, params, size):
        super().__init__(tokenizer, params, size)

        self.init_cache = False

    def __init_cache__(self):
        self.cache = {}
        self.worker_ix = 0
        self.worker_size = int(np.ceil(self.size // self.nworkers))
        self.init_cache = True

    def __getitem__(self, index):
        if not self.init_cache:
            self.__init_cache__()

        label = index % 2
        if self.worker_ix < self.worker_size:
            data = self.datasources[label].read_item(index // 2)
            G = data["G"]
            self.cache[self.worker_ix] = G, label
        else:
            G, label = self.cache[self.worker_ix % self.worker_size]

        self.worker_ix += 1

        return dict(inputs=G, labels=np.array([label]))


class GoppaDistDataset(CodeDistDataset):

    def __init__(self, tokenizer, params, size):
        super().__init__(tokenizer, params, size)
        self.load_data = params.data_path is not None
        self.bundle_size = params.data_bundle_size

        self.manager = Manager()

        if self.load_data:
            self.code = params.task.split("-")[-1]
            self.paths, n_files = self.preload_file_paths(params)
            self.size = self.bundle_size * n_files
            self.size = min(size, self.size)
            self.cache_size = size
            self.cache = self.manager.dict()
            self.load_all()
        else:
            self.cache = self.manager.dict()

    def __getitem__(self, index):
        logger.info(f"Worker {os.getpid()} idx {index}")
        label = index % 2
        if not self.load_data and index not in self.cache:
            G = self.datasources[label].generate()
            self.cache[index] = G
        elif not self.load_data:
            G = self.cache[index]
        else:
            G = self.load(index // 2, label)

        return torch.from_numpy(G), torch.Tensor([label])

    def load_all(self):
        start = time()
        logger.info(f"Started Loading data")
        for idx in range(self.size):
            label = idx % 2
            self.load(idx // 2, label)
        logger.info(f"Finished Loading Data in {time()-start:.2f}")

    def load(self, index, label):
        file_index, sample_index = index // self.bundle_size, index % self.bundle_size
        filepath = self.paths[label][file_index]

        if filepath not in self.cache:
            try:
                # Load the whole file if not in cache
                data = np.load(filepath, allow_pickle=True)["G"].astype(np.uint8)
                Id = np.eye(self.params.k, dtype=np.uint8)
                data = np.concatenate((np.tile(Id, (len(data), 1, 1)), data), axis=2)
            except (OSError, IOError) as e:
                logger.error(f"Error reading file {filepath}: {e}")
                return self.load(np.random.randint(self.size) // 2, label)
            except (KeyError, ValueError) as e:
                logger.error(f"Corrupt or invalid data in file {filepath}: {e}")
                return self.load(np.random.randint(self.size) // 2, label)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

            if len(self.cache) >= self.cache_size:
                self.cache.pop(
                    next(iter(self.cache))
                )  # Remove the first item from the cache
            self.cache[filepath] = data
        else:
            data = self.cache[filepath]
        return data[sample_index]

    def preload_file_paths(self, params):
        paths = dict()
        n_files = np.inf
        for label, code, data_path in [
            (0, "random", params.random_data_path),
            (1, self.code, params.data_path),
        ]:
            file_list = []
            # Walk through the directory structure and build a list of files
            for subdir, dirs, files in tqdm(os.walk(data_path)):
                for file in files:
                    if file.startswith(code) and file.endswith(".npz"):
                        file_path = os.path.join(subdir, file)
                        with np.load(file_path, mmap_mode="r") as f:
                            if "G" in f.files:
                                file_list.append(file_path)
                if len(file_list) * self.bundle_size * 2 > self.size:
                    break
            paths[label] = file_list
            n_files = min(len(file_list), n_files)
        return paths, n_files * 2


# #Ici, params.data_path et params.random_data_path doivent être des fichiers HDF5, pas des dossiers.
# #N’essaie pas de réutiliser GoppaDataDirSource : il est pensé pour un seul objet sous une seule clé.
# class GHDistH5Dataset(Dataset):
#     """
#     Attend deux fichiers HDF5 :
#       - params.data_path         -> fichier Goppa
#       - params.random_data_path  -> fichier random

#     Chaque fichier doit contenir au minimum :
#       - "G": shape (N, k, n)
#       - "H": shape (N, n-k, n)
#     ou bien remplacer "H" par "GT" si tu as gardé ce nom.

#     Le dataset alterne comme le pipeline d'origine :
#       index pair   -> label 0 -> random
#       index impair -> label 1 -> goppa
#     """

#     @classmethod
#     def create(cls, params, tokenizer):
#         assert params.eval_samples > 0
#         dataset_size = params.eval_samples + params.train_samples
#         dataset = cls(tokenizer, params, dataset_size)

#         if len(dataset) < params.eval_samples:
#             raise ValueError(
#                 f"Dataset trop petit: len(dataset)={len(dataset)} < eval_samples={params.eval_samples}"
#             )

#         train = len(dataset) - params.eval_samples
#         test = params.eval_samples

#         train_dataset, test_dataset = random_split(
#             dataset, [train, test], generator=torch.Generator().manual_seed(42)
#         )
#         return train_dataset, test_dataset

#     def __init__(self, tokenizer, params, size):
#         self.tokenizer = tokenizer
#         self.params = params
#         self.size_requested = size

#         self.goppa_path = params.data_path
#         self.random_path = params.random_data_path

#         if self.goppa_path is None or self.random_path is None:
#             raise ValueError("Pour GH il faut --data_path et --random_data_path")

#         self.key_G = "G"
#         self.key_H = getattr(params, "h_key", "H")  # mettre GT si besoin

#         self._goppa_h5 = None
#         self._random_h5 = None

#         with h5py.File(self.goppa_path, "r") as fg, h5py.File(self.random_path, "r") as fr:
#             if self.key_G not in fg or self.key_H not in fg:
#                 raise ValueError(f"Clés absentes dans {self.goppa_path}: attendu {self.key_G} et {self.key_H}")
#             if self.key_G not in fr or self.key_H not in fr:
#                 raise ValueError(f"Clés absentes dans {self.random_path}: attendu {self.key_G} et {self.key_H}")

#             g_shape = fg[self.key_G].shape
#             h_shape = fg[self.key_H].shape
#             rg_shape = fr[self.key_G].shape
#             rh_shape = fr[self.key_H].shape

#             # G: (N, k, n), H: (N, r, n)
#             if len(g_shape) != 3 or len(h_shape) != 3:
#                 raise ValueError(f"Shapes invalides côté goppa: G={g_shape}, H={h_shape}")
#             if len(rg_shape) != 3 or len(rh_shape) != 3:
#                 raise ValueError(f"Shapes invalides côté random: G={rg_shape}, H={rh_shape}")

#             Ng, k, n = g_shape
#             Ngh, r, nh = h_shape
#             Nr, kr, nr = rg_shape
#             Nrh, rr, nrh = rh_shape

#             if Ng != Ngh:
#                 raise ValueError(f"G/H incohérents côté goppa: {g_shape} vs {h_shape}")
#             if Nr != Nrh:
#                 raise ValueError(f"G/H incohérents côté random: {rg_shape} vs {rh_shape}")
#             if n != nh or nr != nrh:
#                 raise ValueError(f"Nombre de colonnes incohérent: goppa n={n}/{nh}, random n={nr}/{nrh}")
#             if k != kr or r != rr or n != nr:
#                 raise ValueError(
#                     f"Goppa/random incompatibles: Goppa G={g_shape}, H={h_shape}, Random G={rg_shape}, H={rh_shape}"
#                 )

#             self.n = n
#             self.k = k
#             self.r = r

#             # taille équilibrée
#             self.n_per_class = min(Ng, Nr)
#             self.size = min(size, 2 * self.n_per_class)

#         # paramètres pour le modèle GH
#         params.k = self.k
#         params.model_input_dim = self.n      # n
#         params.model_input_len = self.k      # k
#         params.model_output_dim = 1
#         params.model_output_len = 1
#         params.output_vocab_size = 1

#     def _ensure_open(self):
#         if self._goppa_h5 is None:
#             self._goppa_h5 = h5py.File(self.goppa_path, "r")
#         if self._random_h5 is None:
#             self._random_h5 = h5py.File(self.random_path, "r")

#     def __len__(self):
#         return self.size

#     def __getitem__(self, index):
#         self._ensure_open()

#         label = index % 2
#         ix = index // 2

#         if ix >= self.n_per_class:
#             raise IndexError(index)

#         if label == 0:
#             f = self._random_h5
#         else:
#             f = self._goppa_h5

#         G = f[self.key_G][ix].astype(np.float32)   # (k, n)
#         H = f[self.key_H][ix].astype(np.float32)   # (r, n)
#         y = np.array([label], dtype=np.float32)

#         return {
#             "inputs" : G,
#             "inputs_G": G,
#             "inputs_H": H,
#             "labels": y,
#         }

#     def collate_fn(self, elements):
#         inputs_G = torch.from_numpy(np.stack([e["inputs_G"] for e in elements], axis=0)).float()
#         inputs_H = torch.from_numpy(np.stack([e["inputs_H"] for e in elements], axis=0)).float()
#         labels = torch.from_numpy(np.stack([e["labels"] for e in elements], axis=0)).float()

#         return {
#             "inputs" : inputs_G,
#             "inputs_G": inputs_G,
#             "inputs_H": inputs_H,
#             "labels": labels,
#         }

# class GHDistAllDataset(CodeDistDataset):
#     def __init__(self, tokenizer, params, size):
#         self.keys = ["G", getattr(params, "h_key", "H")]
#         assert hasattr(params, "param_sets") and params.param_sets is not None, "param_sets is required!"
#         super().__init__(tokenizer, params, size)

#     def _set_datasources(self, params, size):
#         _cls = GHAllDataSource
#         self.datasources = [
#             _cls(params, size // 2, self.keys, True),   # random
#             _cls(params, size // 2, self.keys, False),  # goppa
#         ]

#         self.size = sum(map(len, self.datasources))
#         self.code_len = params.code_len = max(d.max_code_len for d in self.datasources)
#         self.max_n = max(d.max_code_len for d in self.datasources)
#         self.max_k = max(d.max_k for d in self.datasources)
#         self.max_r = max(d.max_r for d in self.datasources)

#         params.k = self.max_k
#         params.model_input_dim = self.max_n
#         params.model_input_len = self.max_k
#         params.model_output_dim = 1
#         params.model_output_len = 1
#         params.output_vocab_size = 1

#     def _pad_G(self, G):
#         k, n = G.shape
#         X = np.zeros((self.max_k, self.max_n), dtype=G.dtype)
#         X[:k, :n] = G
#         return X

#     def _pad_H(self, H):
#         r, n = H.shape
#         X = np.zeros((self.max_r, self.max_n), dtype=H.dtype)
#         X[:r, :n] = H
#         return X

#     def __getitem__(self, index):
#         label = index % 2
#         ix = index // 2
#         data = self.datasources[label].read_item(ix)

#         G = data["G"]
#         H_key = "H" if "H" in data else "GT"
#         H = data[H_key]

#         data.pop("G")
#         data.pop(H_key)

#         return dict(
#             inputs=self._pad_G(G),      # compat éventuelle avec le pipeline
#             inputs_G=self._pad_G(G),
#             inputs_H=self._pad_H(H),
#             labels=np.array([label], dtype=np.float32),
#             **data,
#         )

#     def collate_fn(self, elements):
#         inputs_G = torch.from_numpy(np.stack([e["inputs_G"] for e in elements], axis=0)).float()
#         inputs_H = torch.from_numpy(np.stack([e["inputs_H"] for e in elements], axis=0)).float()
#         labels = torch.from_numpy(np.stack([e["labels"] for e in elements], axis=0)).float()
#         sets = [e["set"] for e in elements] if "set" in elements[0] else None

#         batch = {
#             "inputs": inputs_G,
#             "inputs_G": inputs_G,
#             "inputs_H": inputs_H,
#             "labels": labels,
#         }
#         if sets is not None:
#             batch["set"] = sets
#         return batch

# class GoppaViewDataset(Dataset):
#     """
#     Dataset autonome pour l'expérience "parent -> vue locale".

#     Hypothèses :
#     - on travaille sur la représentation A, stockée sous la clé "G"
#       dans les fichiers/datasources existants ;
#     - classes binaires :
#         label 0 -> random
#         label 1 -> goppa
#     - chaque item renvoie une vue fixe de taille (patch_rows, patch_cols)
#       extraite d'une matrice parente.

#     Attribut important :
#     - view_deterministic = False  -> vue aléatoire (train)
#     - view_deterministic = True   -> vue figée par parent (val/test)
#     """

#     @classmethod
#     def create(cls, params, tokenizer):
#         assert params.eval_samples > 0
#         total_requested = params.train_samples + params.eval_samples
    
#         probe_params = copy.deepcopy(params)
#         probe_params.view_deterministic = False
#         full_dataset = cls(
#             tokenizer=tokenizer,
#             params=probe_params,
#             size=total_requested,
#             indices=None,
#         )
    
#         total_available = len(full_dataset)
#         if total_available < total_requested:
#             raise ValueError(
#                 f"Dataset trop petit : demandé {total_requested}, disponible {total_available}"
#             )
    
#         g = torch.Generator().manual_seed(42)
#         perm = torch.randperm(total_available, generator=g).tolist()
    
#         train_indices = perm[:params.train_samples]
#         val_indices = perm[params.train_samples: params.train_samples + params.eval_samples]
    
#         train_params = copy.deepcopy(params)
#         train_params.view_deterministic = False
    
#         val_params = copy.deepcopy(params)
#         val_params.view_deterministic = True
    
#         train_dataset = cls(
#             tokenizer=tokenizer,
#             params=train_params,
#             size=total_requested,
#             indices=train_indices,
#         )
#         val_dataset = cls(
#             tokenizer=tokenizer,
#             params=val_params,
#             size=total_requested,
#             indices=val_indices,
#         )
    
#         # recopier sur le params original
#         params.model_input_dim = train_dataset.params.model_input_dim
#         params.model_input_len = train_dataset.params.model_input_len
#         params.model_output_dim = 1
#         params.model_output_len = 1
#         params.output_vocab_size = 1
    
#         return train_dataset, val_dataset

#     def __init__(self, tokenizer, params, size, indices=None):
#         super().__init__()

#         self.tokenizer = tokenizer
#         self.params = params
#         self.size_requested = size

#         self.repr = params.repr
#         self.code = params.task.split("-")[-1]
#         self.key = "G"  # dans ton pipeline A est stocké sous "G"

#         if self.code not in {"goppa", "alternant"}:
#             raise ValueError(
#                 f"GoppaViewDataset ne supporte que goppa/alternant, reçu {self.code}"
#             )

#         if self.repr not in {"A", "AT"}:
#             raise ValueError(
#                 f"GoppaViewDataset attend repr='A' ou 'AT', reçu {self.repr}"
#             )

#         self.patch_rows = int(params.patch_rows)
#         self.patch_cols = int(params.patch_cols)

#         self.view_deterministic = bool(getattr(params, "view_deterministic", False))
#         self.view_seed = int(getattr(params, "view_seed", 0))

#         self._rng = None

#         required = ["code_len", "m_alt", "t_alt", "data_path", "random_data_path"]
#         missing = [name for name in required if not hasattr(params, name)]
#         if missing:
#             raise AttributeError(
#                 f"Paramètres manquants pour GoppaViewDataset : {missing}"
#             )

#         if params.data_path is None:
#             raise ValueError("params.data_path est requis")
#         if params.random_data_path is None:
#             raise ValueError("params.random_data_path est requis")

#         # requis par GoppaDataDirSource
#         params.code = self.code

#         # Sources parentes
#         # label 0 -> random
#         # label 1 -> goppa
#         self.datasources = [
#             GoppaDataDirSource(params, size // 2, [self.key], True),
#             GoppaDataDirSource(params, size // 2, [self.key], False),
#         ]

#         self.n_per_class = min(len(ds) for ds in self.datasources)
#         self.base_size = 2 * self.n_per_class

#         if indices is None:
#             self.indices = list(range(self.base_size))
#         else:
#             self.indices = list(indices)

#         self.size = len(self.indices)

#         dim_mul = tokenizer.data_dim_multiplier
#         len_mul = tokenizer.data_len_multiplier

#         params.model_input_dim = self.patch_cols * dim_mul
#         params.model_input_len = self.patch_rows * len_mul
#         params.model_output_dim = 1
#         params.model_output_len = 1
#         params.output_vocab_size = 1

#     def __len__(self):
#         return self.size

#     def _get_rng(self):
#         """
#         RNG local par worker pour le train.
#         """
#         if self._rng is None:
#             seed = torch.initial_seed() % (2**32)
#             self._rng = np.random.default_rng(seed)
#         return self._rng

#     def _make_rng(self, parent_id, view_id=None):
#         if self.view_deterministic:
#             if view_id is None:
#                 view_id = 0
#             seed = self.view_seed + 1000003 * int(parent_id) + int(view_id)
#             return np.random.default_rng(seed)
#         return self._get_rng()
    
#     def _sample_view(self, A, rng):
#         A = np.asarray(A)
#         k, r = A.shape

#         if self.patch_rows > k or self.patch_cols > r:
#             raise ValueError(
#                 f"Patch demandé ({self.patch_rows}, {self.patch_cols}) "
#                 f"incompatible avec A.shape={A.shape}"
#             )

#         rows = np.sort(rng.choice(k, size=self.patch_rows, replace=False))
#         cols = np.sort(rng.choice(r, size=self.patch_cols, replace=False))
#         return A[rows][:, cols]

#     def _get_parent_and_label_from_global_index(self, global_index):
#         label = global_index % 2
#         ix = global_index // 2
#         return label, ix

#     def _get_item_from_global_index(self, global_index, view_id=None):
#         label = global_index % 2
#         ix = global_index // 2
    
#         data = self.datasources[label].read_item(ix)
#         A_parent = np.asarray(data[self.key])
    
#         parent_id = label * self.n_per_class + ix
#         rng = self._make_rng(parent_id, view_id=view_id)
#         view = self._sample_view(A_parent, rng)
    
#         return {
#             "inputs": view.astype(np.float32),
#             "labels": np.array([label], dtype=np.float32),
#             "parent_id": np.int64(parent_id),
#         }

#     def __getitem__(self, index):
#         global_index = self.indices[index]
#         return self._get_item_from_global_index(global_index, view_id=None)

#     def collate_fn(self, elements):
#         data = {
#             key: [elem[key] for elem in elements]
#             for key in elements[0]
#         }

#         return {
#             "inputs": self.tokenizer(data["inputs"]),
#             "labels": self.tokenizer(data["labels"], labels=True),
#             "parent_id": torch.tensor(data["parent_id"], dtype=torch.long),
#         }

# class MultiViewEvalDataset(Dataset):
#     def __init__(self, base_dataset, num_views):
#         self.base_dataset = base_dataset
#         self.num_views = int(num_views)

#         if not getattr(base_dataset, "view_deterministic", False):
#             raise ValueError("MultiViewEvalDataset attend un base_dataset deterministic")

#     def __len__(self):
#         return len(self.base_dataset) * self.num_views

#     def __getitem__(self, index):
#         parent_local_idx = index // self.num_views
#         view_id = index % self.num_views

#         global_index = self.base_dataset.indices[parent_local_idx]
#         item = self.base_dataset._get_item_from_global_index(global_index, view_id=view_id)
#         item["view_id"] = np.int64(view_id)
#         return item

#     def collate_fn(self, elements):
#         data = {
#             key: [elem[key] for elem in elements]
#             for key in elements[0]
#         }

#         return {
#             "inputs": self.base_dataset.tokenizer(data["inputs"]),
#             "labels": self.base_dataset.tokenizer(data["labels"], labels=True),
#             "parent_id": torch.tensor(data["parent_id"], dtype=torch.long),
#             "view_id": torch.tensor(data["view_id"], dtype=torch.long),
#         }