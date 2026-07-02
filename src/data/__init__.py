# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse

from src.data.distinguisher_datasets import (
    CodeDistDataset,
    GoppaDistAllDataset,
    GoppaDistGenDataset,
    CodeDistH5Dataset,
    QCDistDataset,
)
from src.data.decoding_datasets import (
    FlatGoppaCompleteDataset,
    GoppaCompleteDataset,
    GoppaCorrectDataset,
)
from src.data.tokenizers import (
    BinaryCodePatchTokenizer,
    BinaryCodeTokenizer,
    FlatCodeCompleteTokenizer,
    BaseCodeTokenizer,
    FqTokenizer,
    Tokenizer,
)
from src.data.symbolic_datasets import GoppaSymDistDataset


def get_goppa_parser():
    parser = argparse.ArgumentParser(description="Goppa task parser")
    parser.add_argument("--m_alt", type=int, help="defines modulus q = 2^m")
    parser.add_argument("--t_alt", type=int, help="degree of irreducible polynomial g")
    parser.add_argument("--beta_dist", type=str, default="uniform")
    parser.add_argument("--alpha_dist", type=str, default="uniform")
    return parser


def parse_goppa_args(unknown_args, namespace):
    parser = get_goppa_parser()
    parser.add_argument(
        "--data_bundle_size",
        type=int,
        default=100,
        help="if data is loaded, how many samples each file contains",
    )
    parser.add_argument("--param_sets", type=str, default=None, required=False)

    # Parse the additional arguments directly into the provided Namespace object
    return parser.parse_args(unknown_args, namespace=namespace)


def parse_qc_args(unknown_args, namespace):
    parser = argparse.ArgumentParser(description="Goppa task parser")
    parser.add_argument("--r", type=int, help="degree of x^r-1")
    parser.add_argument("--w", type=int, help="weight of h0 and h1")

    parser.add_argument(
        "--data_bundle_size",
        type=int,
        default=100,
        help="if data is loaded, how many samples each file contains",
    )

    # Parse the additional arguments directly into the provided Namespace object
    return parser.parse_args(unknown_args, namespace=namespace)


def parse_codecomplete_args(unknown_args, namespace):
    parser = get_goppa_parser()
    parser.add_argument("--n_masked", type=float, default=1)
    parser.add_argument("--repset_size", type=int, default=0)
    return parser.parse_args(unknown_args, namespace=namespace)


def get_datasets(params):
    tokenizer = get_tokenizer(params)
    if params.task.startswith("view-goppa"):
        _cls = GoppaViewDataset
    elif params.task.startswith("code-dist-gh") :
        _cls = GHDistH5Dataset
        
    elif params.task == "code-dist-all-gh-goppa":
        _cls = GHDistAllDataset
    
    elif params.task.startswith("code-dist"):
        if params.task.endswith("goppa") or params.task.endswith("alternant"):
            if "symbolic" in params.task:
                _cls = GoppaSymDistDataset
            elif "all" in params.task:
                _cls = GoppaDistAllDataset
            elif params.data_path:
                _cls = CodeDistH5Dataset
            else:
                _cls = GoppaDistGenDataset
        elif params.task.endswith("mdpc") and params.data_path:
            _cls = CodeDistH5Dataset
        elif params.task.endswith("mdpc"):
            _cls = CodeDistDataset
        elif params.task.endswith("qc"):
            _cls = QCDistDataset
        else:
            raise ValueError("Not supported")

    elif params.task.startswith("code-complete"):
        if "flat" in params.task:
            _cls = FlatGoppaCompleteDataset
        elif "correct" in params.task:
            _cls = GoppaCorrectDataset
        else:
            _cls = GoppaCompleteDataset
    else:
        raise ValueError()

    train_dataset, test_dataset = _cls.create(params, tokenizer)
    return train_dataset, test_dataset


def get_tokenizer(params):
    if params.task.startswith("code-dist-gh"):
        params.Q = 2
        tokenizer = FqTokenizer(params)
    elif params.task.startswith("code-dist") or params.task.startswith("view") :
        if params.task.endswith("qc") or params.task.endswith("mdpc"):
            tokenizer = FqTokenizer(params)
        elif "symbolic" in params.task:
            tokenizer = BinaryCodeTokenizer(params)
        elif "patchsymbolic" in params.task:
            tokenizer = BinaryCodePatchTokenizer(params, patch_h=1, patch_w=1)
        else:
            tokenizer = FqTokenizer(params)

    elif params.task.startswith("code-symbolic-qc"):
        tokenizer = Tokenizer(params)

    elif params.task.startswith("code-complete"):
        if "flat" in params.task:
            tokenizer = FlatCodeCompleteTokenizer(params)
        else:
            tokenizer = BaseCodeTokenizer()

    else:
        raise ValueError("What tokenizer to use ?")
    return tokenizer
