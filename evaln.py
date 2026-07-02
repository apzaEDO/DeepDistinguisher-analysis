from pathlib import Path
import argparse
import copy
import pickle

import torch
from torch.utils.data import Subset

from src.data import get_datasets
from src.model import get_model
from src.metrics import compute_classification_metrics_per_cat_acc_only
from src.trainer import Trainer, TrainingArguments
from src.utils import init_wandb


# ============================================================
# Parameter loading
# ============================================================

def namespace_from_pickle_object(loaded_object):
    """
    Convert the content of params.pkl into an argparse.Namespace-like object.

    Accepted cases:
      - argparse.Namespace
      - dict
      - dict containing a "params" key
      - object with __dict__
    """
    if isinstance(loaded_object, dict) and "params" in loaded_object:
        loaded_object = loaded_object["params"]

    if isinstance(loaded_object, argparse.Namespace):
        return copy.deepcopy(loaded_object)

    if isinstance(loaded_object, dict):
        return argparse.Namespace(**copy.deepcopy(loaded_object))

    if hasattr(loaded_object, "__dict__"):
        return copy.deepcopy(loaded_object)

    raise TypeError(
        f"Unsupported params.pkl object type: {type(loaded_object)!r}. "
        "Expected argparse.Namespace, dict, or namespace-like object."
    )


def load_params_from_checkpoint(checkpoint_path):
    """
    Load params.pkl located in the same directory as checkpoint.pth.
    """
    checkpoint_path = Path(checkpoint_path)
    params_path = checkpoint_path.with_name("params.pkl")

    if not params_path.exists():
        raise FileNotFoundError(f"params.pkl not found: {params_path}")

    with params_path.open("rb") as param_file:
        loaded_object = pickle.load(param_file)

    params = namespace_from_pickle_object(loaded_object)
    print(f"Loaded params from: {params_path}")

    return params


# ============================================================
# Checkpoint path resolution
# ============================================================

def get_checkpoint_path(
    n,
    m,
    t,
    representation="AT",
    checkpoint_kind="all",
    checkpoint_root="./checkpoint/debug_pretrain",
):
    """
    Resolve the checkpoint path from n, m, t, and checkpoint_kind.
    """
    checkpoint_root = Path(checkpoint_root)

    if checkpoint_kind == "standard":
        candidate_paths = [
            checkpoint_root / f"{representation}_model_Goppa_N{n}_T{t}_M{m}" / "checkpoint.pth",
        ]

    elif checkpoint_kind == "all":
        candidate_paths = [
            checkpoint_root / f"{representation}_model_NGoppa_N{n}_T{t}_M{m}" / "checkpoint.pth",
        ]

    elif checkpoint_kind == "goppaall":
        candidate_paths = [
            checkpoint_root / f"{representation}_model_GoppaAll_Nmax{n}_T{t}_M{m}" / "checkpoint.pth",
        ]

    else:
        raise ValueError(
            "checkpoint_kind must be one of: 'standard', 'all', 'goppaall'."
        )

    for path in candidate_paths:
        if path.exists():
            return path

    tried_paths = "\n".join(f"  - {path}" for path in candidate_paths)
    raise FileNotFoundError(f"Checkpoint not found. Tried:\n{tried_paths}")


# ============================================================
# Evaluation parameters
# ============================================================

def build_param_sets(n, m, t):
    """
    Automatically build params.param_sets for all checkpoints.

    Example:
      t=3 -> 24, 32, 40, 48, 56, 64
      t=4 -> 32, 40, 48, 56, 64
      t=5 -> 40, 48, 56, 64
    """
    code_lengths = [
        code_length
        for code_length in range(8 * t, n + 1, 8)
        if code_length > m * t
    ]

    return ";".join(
        f"{code_length}, {m}, {t}"
        for code_length in code_lengths
    )


def patch_eval_params(
    params,
    n,
    m,
    t,
    representation="AT",
    checkpoint_kind="all",
    device="cuda",
):
    """
    Apply the required overrides to reproduce the evaluation code.

    Important:
    even with representation='AT', the model is built with raw matrices of size:

        model_input_len = n - m*t
        model_input_dim = m*t

    Therefore, matrices must not be transformed into [A.T | I].
    """
    k = n - m * t

    if k <= 0:
        raise ValueError(
            f"Invalid parameters: k = n - m*t = {n} - {m}*{t} = {k}. "
            "Expected k > 0."
        )

    # Code parameters.
    params.code = "goppa"
    params.code_len = n
    params.k = k
    params.m_alt = m
    params.t_alt = t
    params.representation = representation

    # Model input geometry.
    params.model_input_dim = m * t
    params.model_input_len = k
    params.model_output_dim = 1
    params.model_output_len = 1
    params.output_vocab_size = 1

    # Task.
    if checkpoint_kind == "standard":
        params.task = "code-dist-goppa"
        params.param_sets = None

    elif checkpoint_kind in {"all", "goppaall"}:
        params.task = "code-dist-all-goppa"
        params.param_sets = build_param_sets(n=n, m=m, t=t)

    else:
        raise ValueError(
            "checkpoint_kind must be one of: 'standard', 'all', 'goppaall'."
        )

    params.model = "encoder"

    # Runtime parameters.
    params.device = device
    params.local_rank = -1
    params.multi_gpu = False
    params.is_master = True
    params.compile = False
    params.wandb = False
    params.checkpoint_model = False
    params.resume = ""
    params.resume_from_checkpoint = None

    # Default values if missing from params.pkl.
    if not hasattr(params, "dump_path"):
        params.dump_path = "./checkpoint"

    if not hasattr(params, "num_train_epochs"):
        params.num_train_epochs = 1

    if not hasattr(params, "val_every"):
        params.val_every = 1000

    if not hasattr(params, "log_every"):
        params.log_every = 10

    if not hasattr(params, "save_every"):
        params.save_every = 10000

    if not hasattr(params, "train_batch_size"):
        params.train_batch_size = 16

    if not hasattr(params, "val_batch_size"):
        params.val_batch_size = 32

    if not hasattr(params, "workers"):
        params.workers = 8

    if not hasattr(params, "dtype"):
        params.dtype = "float16"

    if not hasattr(params, "clip_grad_norm"):
        params.clip_grad_norm = 5.0

    if not hasattr(params, "optimizer"):
        params.optimizer = "adam_warmup,lr=0.00001,warmup_updates=1000,weight_decay=0.001"

    # Default data paths if missing from params.pkl.
    if not hasattr(params, "data_path") or params.data_path in {None, ""}:
        params.data_path = (
            f"./data/dataset_goppa_<codelen>_H5/"
            f"{representation}_goppa_nmt_<codelen>_<malt>_<talt>/"
            f"dataset_10K.h5"
        )

    if not hasattr(params, "random_data_path"):
        params.random_data_path = None

    return params


# ============================================================
# Model and trainer loading
# ============================================================

def load_state_dict_into_model(model, checkpoint_path):
    """
    Load checkpoint.pth into the model.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = {
        key.replace("module.", ""): value
        for key, value in checkpoint["model"].items()
    }

    if hasattr(model, "pos_emb") and "pos_emb.weight" not in state_dict:
        print(
            "Warning: pos_emb.weight missing from checkpoint. "
            "Initializing it to zero."
        )
        state_dict["pos_emb.weight"] = torch.zeros_like(model.pos_emb.weight)

    model.load_state_dict(state_dict)

    return model


def build_trainer(
    n,
    m,
    t,
    checkpoint_kind="all",
    representation="AT",
    checkpoint_root="./checkpoint/debug_pretrain",
    device=None,
):
    """
    Load params.pkl, build the dataset, load the model, and return the trainer.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = get_checkpoint_path(
        n=n,
        m=m,
        t=t,
        representation=representation,
        checkpoint_kind=checkpoint_kind,
        checkpoint_root=checkpoint_root,
    )

    print(f"Loading checkpoint: {checkpoint_path}")

    params = load_params_from_checkpoint(checkpoint_path)

    params = patch_eval_params(
        params=params,
        n=n,
        m=m,
        t=t,
        representation=representation,
        checkpoint_kind=checkpoint_kind,
        device=device,
    )

    print("Evaluation parameters:")
    print(params)

    _, test_dataset = get_datasets(params)

    model = get_model(params)

    model = load_state_dict_into_model(
        model=model,
        checkpoint_path=checkpoint_path,
    )

    model.eval()
    model.to(params.device)

    report_to = init_wandb(params)

    training_args = TrainingArguments(
        dump_path=params.dump_path,
        evaluation_strategy="steps",
        num_train_epochs=params.num_train_epochs,
        eval_steps=params.val_every,
        logging_steps=params.log_every,
        save_steps=params.save_every,
        per_device_train_batch_size=params.train_batch_size,
        per_device_eval_batch_size=params.val_batch_size,
        report_to=report_to,
        local_rank=params.local_rank,
        dataloader_num_workers=params.workers,
        device=params.device,
        multi_gpu=params.multi_gpu,
        dtype=params.dtype,
        max_grad_norm=params.clip_grad_norm,
        compile=params.compile,
        optimizer=params.optimizer,
        resume_from_checkpoint=params.resume_from_checkpoint,
    )

    trainer = Trainer(
        model=model,
        training_args=training_args,
        args=params,
        train_dataset=None,
        eval_dataset=test_dataset,
        data_collator=(
            test_dataset.dataset.collate_fn
            if isinstance(test_dataset, Subset)
            else test_dataset.collate_fn
        ),
        compute_metrics=compute_classification_metrics_per_cat_acc_only,
        callbacks=[],
    )

    trainer.uncompiled_model = model

    return trainer, params


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    n,
    m,
    t,
    checkpoint_kind="all",
    representation="AT",
    checkpoint_root="./checkpoint/debug_pretrain",
    device=None,
):
    """
    Main evaluation function: load the model, then run trainer.evaluate().
    """
    trainer, params = build_trainer(
        n=n,
        m=m,
        t=t,
        checkpoint_kind=checkpoint_kind,
        representation=representation,
        checkpoint_root=checkpoint_root,
        device=device,
    )

    print(
        "\nRunning trainer.evaluate() with "
        f"n={n}, m={m}, t={t}, "
        f"representation={representation}, "
        f"checkpoint_kind={checkpoint_kind}"
    )

    metrics = trainer.evaluate()

    return metrics


# ============================================================
# Main
# ============================================================

def main():
    # ============================================================
    # User configuration
    # ============================================================

    n = 64
    m = 6
    t = 4

    representation = "AT"

    # Possible values:
    #   "standard" : AT_model_Goppa_N64_T3_M6/checkpoint.pth
    #   "all"      : AT_model_NGoppa_N64_T3_M6/checkpoint.pth
    #   "goppaall" : AT_model_GoppaAll_Nmax64_T3_M6/checkpoint.pth
    checkpoint_kind = "all"

    checkpoint_root = "./checkpoint/debug_pretrain"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = evaluate_model(
        n=n,
        m=m,
        t=t,
        checkpoint_kind=checkpoint_kind,
        representation=representation,
        checkpoint_root=checkpoint_root,
        device=device,
    )

    return metrics


if __name__ == "__main__":
    main()