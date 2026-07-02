# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from logging import Logger
import numpy as np
from src import utils
from src.utils import bool_flag, initialize_exp
from joblib import Parallel, delayed, cpu_count
import os
from argparse import Namespace
from src.data.generators import (
    AlternantCodeGenerator,
    CodeGenerator,
    GoppaCodeGenerator,
    MDPCCodeGenerator,
    QCCodeGenerator,
    parse_fn,
)
from src.logger import create_logger
import getpass
import logging


np.seterr(all="raise")

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def get_params():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Goppa Dataset Generation")

    user = getpass.getuser()
    parser.add_argument("--dump_path", default=f"/checkpoint/{user}/dumped/debug")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of CPU workers for DataLoader",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="number of matrices to reduced per worker",
    )

    parser.add_argument("--code", type=str, default="goppa")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Multi-GPU - Local rank"
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=-1,
        help="Master port (for multi-node SLURM jobs)",
    )
    parser.add_argument("--save_every", type=int, default=10)

    parser.add_argument("--code_dim", type=int, default=-1, help="code dimension")
    parser.add_argument("--code_len", type=int, default=-1, help="code length")
    parser.add_argument("--Q", type=int, default=2)
    parser.add_argument(
        "--standard_only",
        type=bool_flag,
        default=False,
        help="filter out non standard form codes",
    )
    parser.add_argument("--representation", type=str, default = "G")
    params, unknown = parser.parse_known_args()
    params = parse_fn[params.code](unknown, params)
    return params


class MKDataset:
    def __init__(self, params: Namespace, worker: int) -> None:
        self.params: Namespace = params
        self.worker: int = worker
        self.n: int = params.code_len
        self.code: str = params.code
        self.dump_path: str = params.dump_path

        self.logger: Logger = create_logger(
            os.path.join(params.dump_path, "train.log"),
            rank=0,
        )
        self.representation = params.representation
        
        if params.code == "goppa":
            _cls = GoppaCodeGenerator
        elif params.code == "alternant":
            _cls = AlternantCodeGenerator
        elif params.code == "random":
            _cls = CodeGenerator
        elif params.code == "qc":
            _cls = QCCodeGenerator
        elif params.code == "mdpc":
            _cls = MDPCCodeGenerator
        else:
            raise NotImplementedError()

        self.generator = _cls(params, loggr=self.logger, worker=worker)
        self.data = []
        self.counter = 0
        self.save_every = params.save_every

    def save_data(self, sample_id, item):
        """Save the basis and shortest vector to disk."""
        self.counter += 1
        self.data.append(item)
        if self.counter % self.save_every == 0:
            try:
                filepath = os.path.join(
                    self.dump_path, f"{self.code}_{self.worker}_{sample_id}.npz"
                )
                data = {
                    key: np.stack([d[key] for d in self.data])
                    for key in self.data[0].keys()
                }
                np.savez(filepath, **data)
                self.logger.info(
                    f"[{self.worker}] Saved batch {sample_id//self.save_every} to {filepath}"
                )
            except Exception as e:
                self.logger.error(str(e))

            self.data = []

    def process_sample(self, sample_id):
        """Generate and save a single lattice sample."""
        if self.representation in ["H","T"]:
            item = self.generator.generate_code_H()
        elif self.representation in ["G","A"]:
            item = self.generator.generate_code_G()
        elif self.representation =="HT" :
            item = self.generator.generate_code_HT()
        elif self.representation == "GH":
            item = self.generator.generate_code_GH()
        elif self.representation =="GD":
            if params.code =="random" :
                item = self.generator.generate_code_GD()
            else :
                item = self.generator.generate_code_G()
        else :
            item = self.generator.generate_code_GT()
        
        self.save_data(sample_id, item)


def process(worker, params):
    cls = MKDataset(params, worker)
    for sample_id in range(int(np.ceil(params.n_samples / params.num_workers))):
        cls.process_sample(sample_id)


def get_group_folder(params):
    if params.code in ["goppa", "alternant"] or (
        params.code == "random" and params.t_alt is not None
    ):
        if params.representation == "GN" :
            return f"{params.code}_nmt_{params.code_len}_{params.m_alt}_{params.t_alt}"
        else :
            return f"{params.representation}_{params.code}_nmt_{params.code_len}_{params.m_alt}_{params.t_alt}"
    elif params.code == "random":
        return f"{params.representation}_{params.code}_nk_{params.code_len}_{params.code_dim}"
    elif params.code in ["qc", "mdpc"]:
        return f"{params.code}_rw_{params.r}_{params.w}"
    else:
        raise ValueError(f"Code {params.code} not supported")


def main(params):
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    params.group_folder = get_group_folder(params)
    logger = initialize_exp(params)

    utils.CUDA = False

    n_cpu = cpu_count()
    n_jobs = min(n_cpu, params.num_workers)
    logger.info(f" Nb CPU: {n_cpu} and Nb worker: {params.num_workers}")
    Parallel(n_jobs=n_jobs)(delayed(process)(n, params) for n in range(n_jobs))


if __name__ == "__main__":
    # generate parser / parse parameters
    params = get_params()

    params.cpu, params.debug_slurm = True, False

    # run experiment
    main(params)
