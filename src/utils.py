# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict
import datetime
import itertools
import os
import re
import sys
import pickle
import random
import getpass
import argparse
import subprocess
from typing import Union
import torch
from pathlib import Path
from src.logger import create_logger


FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}

DUMP_PATH = "/checkpoint/%s/dumped" % getpass.getuser()
CUDA = True


def create_this_logger(params):
    return create_logger(
        os.path.join(params.dump_path, "train.log"),
        rank=getattr(params, "global_rank", 0),
    )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    try:
        pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))
    except Exception as e:
        print(e)

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith("--"):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match("^[a-zA-Z0-9_]+$", x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = " ".join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_this_logger(params)
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_job_id(sweep_path):
    chronos_job_id = os.environ.get("CHRONOS_JOB_ID")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    assert chronos_job_id is None or slurm_job_id is None
    exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
    if exp_id is None:
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(10))
            if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                break
    else:
        assert exp_id.isdigit()
    return exp_id


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    params.dump_path = DUMP_PATH if params.dump_path == "" else params.dump_path
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    if hasattr(params, "group_folder") and len(params.group_folder) > 0:
        sweep_path = os.path.join(sweep_path, params.group_folder)

    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == "":
        params.exp_id = get_job_id(sweep_path)

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


def to_device_rec(data, device, keys=None):
    """
    Move data to the specified device recursively.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device_rec(x, device, keys) for x in data)
    elif isinstance(data, dict):
        return {
            k: to_device_rec(v, device, keys)
            for k, v in data.items()
            if keys is None or k in keys
        }
    else:
        return data


# unit of time
hour = datetime.timedelta(hours=1)


def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return "{:.3f}MB".format(size_all_mb)


def init_wandb(params):
    report_to = "none"
    if params.is_master and params.wandb:
        from dotenv import load_dotenv

        load_dotenv()
        os.environ["WANDB_PROJECT"] = params.exp_name
        if hasattr(params, "checkpoint_model") and params.checkpoint_model:
            os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        wandb.login()
        if hasattr(params, "wandb_primary_key"):
            primkeys = [
                f"{k}:{params.__dict__[k]}"
                for k in params.wandb_primary_key.split(";")
                if k and k != "exp_id"
            ]
            expid = "|".join(primkeys)
            expid = str(params.exp_id)[-3:] + "|" + expid
        else:
            expid = params.exp_id
        run = wandb.init(
            project=params.exp_name, name=expid, config=params.__dict__, resume="allow"
        )
        report_to = "wandb"
    return report_to


def end_wandb(params, **results):

    if params.is_master and params.wandb:
        for k, v in results.items():
            # if isinstance(v, pd.DataFrame):
            #     wandb.log({k:wandb.Table(dataframe=v)})
            # else:
            wandb.log({k: v})

        wandb.finish()


def try_load_params(params):
    if params.data_path and os.path.exists(
        os.path.join(params.data_path, "params.pkl")
    ):
        with open(os.path.join(params.data_path, "params.pkl"), "rb") as fd:
            data = pickle.load(fd)

        to_update = ["N", "Q", "sigma", "gamma", "secret_type"]

        subset = {k: data[k] for k in to_update if k in data}
        vars(params).update(subset)
        if params.k == -1:
            params.k = params.N


class GoppaCodeParams:
    def __init__(self, param_sets: Union[str, list[tuple[int]]], min_rate: float = 0.0):
        self.deterministic = True
        if isinstance(param_sets, str):
            param_sets = self.parse_sets(param_sets)

        self.param_sets_list = self.filter_sets(param_sets, min_rate)
        self.items = set(self.param_sets_list)
        self.param_sets_nested = self.list_to_nested_dict(self.param_sets_list)

    def list_to_nested_dict(self, elements):
        param_sets_nested = defaultdict(lambda: defaultdict(list))

        for n, m, t in elements:
            param_sets_nested[m][t].append(n)

        return param_sets_nested

    def __getitem__(self, index):
        ix = index
        return self.param_sets_list[ix]

    def __len__(self):
        return len(self.param_sets_list)

    def __contains__(self, item):
        return item in self.items

    def get_max_n(self):
        return max(self.param_sets_list, key=lambda x: x[0])[0]

    def sample(self, index=-1):
        if self.deterministic and index >= 0:
            root = list(self.param_sets_nested.keys())
            m_ix, ix = index % len(root), index // len(root)
            m = root[m_ix]
            node = list(self.param_sets_nested[m].keys())
            t_ix, ix = ix % len(node), ix // len(node)
            t = node[t_ix]
            node = self.param_sets_nested[m][t]
            n_ix = ix % len(node)
            n = node[n_ix]
        else:
            m = random.choice(list(self.param_sets_nested.keys()))

            t = random.choice(list(self.param_sets_nested[m].keys()))

            n = random.choice(self.param_sets_nested[m][t])

        return (n, m, t)

    def filter_sets(self, sets, min_rate=0):
        valid = []
        for n, m, t in sets:
            k = n - m * t
            if m * t > 0 and k > 0 and k / n >= min_rate:
                valid.append((n, m, t))
        return valid

    def apply(self, filter_fn):
        self.param_sets_list = filter_fn(self.param_sets_list)
        self.param_sets_nested = self.list_to_nested_dict(self.param_sets_list)
        self.items = set(self.param_sets_list)

    def parse_sets(self, sets_str):
        subsets = sets_str.split(";")
        tuples = []
        for subset in subsets:
            expanded_elements = []
            for el in subset.split(","):
                if ":" in el:
                    rnge = list(map(int, el.split(":")))
                    if len(rnge) == 3:
                        emin, emax, step = rnge
                    else:
                        emin, emax = rnge
                        step = 1
                    values = list(range(emin, emax + 1, step))
                    expanded_elements.append(values)
                else:
                    e = int(el)
                    expanded_elements.append([e])

            for combination in itertools.product(*expanded_elements):
                tuples.append(tuple(combination))
        return tuples


def concat_nested(elements, cpu=True):
    assert len(elements)
    element = elements[0]
    if isinstance(element, torch.Tensor):
        elements = torch.cat(elements, dim=0)
        if cpu:
            elements = elements.cpu()
        return elements
    elif isinstance(element, (tuple, list)):
        return type(element)(
            [concat_nested([el[i] for el in elements]) for i in range(len(element))]
        )
    else:
        raise NotImplementedError()


def add_prefix_to_keys(prefix, data):
    """
    Add a prefix to all keys in a dictionary or nested dictionaries.
    """
    if isinstance(data, dict):
        return {f"{prefix}{k}": add_prefix_to_keys(prefix, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(add_prefix_to_keys(prefix, x) for x in data)
    else:
        return data


def get_base_folder(exp_name) -> Path:
    username = getpass.getuser()
    super_base_folder = f"/checkpoint/{username}/dumped/"
    base_folder = f"{super_base_folder}{exp_name}"
    return Path(base_folder)
