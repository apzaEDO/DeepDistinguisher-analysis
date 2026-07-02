# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import os
import h5py
from tqdm import tqdm
import argparse
import getpass


def get_params():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Goppa Dataset Generation")

    parser.add_argument("--data_path")
    parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="number of matrices to reduced per worker",
    )

    parser.add_argument("--code", type=str, default="goppa")

    parser.add_argument("--code_len", type=int, default=-1, help="code length")
    parser.add_argument("--code_dim", type=int, default=-1, help="code dimension")

    parser.add_argument("--m_alt", type=int, help="defines modulus q = 2^m")
    parser.add_argument("--t_alt", type=int, help="degree of irreducible polynomial g")
    parser.add_argument("--r", type=int, help="degree of x^r-1")
    parser.add_argument("--w", type=int, help="weight of h0 and h1")
    parser.add_argument("--Q", type=int)
    parser.add_argument("--representation",type=str,default="G")
    params = parser.parse_args()
    return params


def size_to_readable(n):
    log10 = int(np.log10(n))
    L = ["", "K", "M", "B"]
    ix = log10 // 3
    l = L[ix]
    return f"{int(n//10**(3*ix))}{l}"


def get_key(code):
    if code == "qc":
        return "h"
    return "G"


def read_files(code, root_path):

    for subdir, dirs, files in tqdm(os.walk(root_path)):
        for file in files:
            if file.startswith(code) and file.endswith(".npz"):
                filepath = os.path.join(subdir, file)
                data = np.load(filepath, allow_pickle=True)
                if get_key(code) not in data.files:
                    continue
                yield data


def npz_to_h5(code, input_dir, output_path, dataset_size=100):
    for dpoint in read_files(code, input_dir):
        break

    # Create HDF5 file
    outfilename = f"dataset_{size_to_readable(dataset_size)}.h5"
    output_file = os.path.join(output_path, outfilename)
    print(f"Creating file in {output_file}")
    with h5py.File(output_file, "w") as hf:
        dataset = dict()
        for k, v in dpoint.items():
            # Create dataset with total number of matrices
            dataset[k] = hf.create_dataset(
                k, shape=(dataset_size,) + tuple(v.shape[1:]), dtype=np.uint8
            )

        # Initialize counter for matrix index
        matrix_idx = 0
        end = False
        # Iterate over all npz files in the directory
        for dpoint in read_files(code, input_dir):
            for k, v in dpoint.items():
                if matrix_idx + v.shape[0] <= dataset_size:
                    # Write to HDF5 file
                    dataset[k][matrix_idx : matrix_idx + len(v)] = v

                else:
                    di = dataset_size - matrix_idx
                    dataset[k][matrix_idx:] = v[:di]
                    end = True
            if end:
                print(f"Finished loading {dataset_size} matrices")
                break
            # Increment matrix index
            matrix_idx += len(v)

        if not end and matrix_idx < dataset_size:
            print(
                f"Problem: created dataset of size {dataset_size} but found only {matrix_idx} samples"
            )

    print(f"NPZ files have been successfully written to {output_file}")


def get_group_folder(params):
    if params.code in ["goppa", "alternant"] or (
        params.code == "random" and params.m_alt and params.t_alt
    ):
        if params.representation =="GN":
            return f"{params.code}_nmt_{params.code_len}_{params.m_alt}_{params.t_alt}"
        else :
            return f"{params.representation}_{params.code}_nmt_{params.code_len}_{params.m_alt}_{params.t_alt}"
    elif params.code == "random":
        return f"{params.representation}_{params.code}_nk_{params.code_len}_{params.code_dim}"

    elif params.code == "qc" or params.code == "mdpc":
        return f"{params.representation}_{params.code}_rw_{params.r}_{params.w}"
    else:
        raise ValueError(f"Code {params.code} not supported")


def get_folder(params):
    main_param = None
    if params.code in ["goppa", "alternant", "random"]:
        main_param = params.code_len
    elif params.code == "qc":
        main_param = params.r

    name = os.path.basename(params.data_path) + "_H5"
    # if params.Q == 2:
    #     name = f'dataset_H5_{params.code}_{main_param}'
    # else:
    #     name = f'dataset_H5_{params.code}_{main_param}_q_{params.Q}'

    return name


if __name__ == "__main__":
    # generate parser / parse parameters
    params = get_params()

    dirname = get_group_folder(params)

    folder = get_folder(params)

    destination_path = os.path.join(os.path.dirname(params.data_path), folder, dirname)
    destination_path = destination_path
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    if os.path.exists(os.path.join(params.data_path, dirname)):
        path = os.path.join(params.data_path, dirname)
    else:
        path = params.data_path

    # run experiment
    npz_to_h5(params.code, path, destination_path, params.n_samples)
