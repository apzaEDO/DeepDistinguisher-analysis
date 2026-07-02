# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import argparse
import getpass
import os
import socket
from time import time
import torch
from torch.utils.data import Subset
import torch.distributed as dist
from src.metrics import (
    compute_accuracy,
    compute_accuracy_with_mask,
    compute_classification_metrics_per_cat,
    compute_classification_metrics_per_cat_acc_only
)
    
from src.data import (
    parse_codecomplete_args,
    parse_goppa_args,
    parse_qc_args,
    get_datasets,
)
from src.model import get_model
from src.trainer import TrainingArguments, Trainer
from src.utils import end_wandb, init_wandb, initialize_exp, try_load_params, bool_flag
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import Counter


log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


def setup_distributed(params):
    params.rank = int(os.environ.get("RANK", 0))
    params.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    params.world_size = int(os.environ.get("WORLD_SIZE", 1))
    params.distributed = params.world_size > 1

    
    print(
        f"[pre-set_device] host={socket.gethostname()} pid={os.getpid()} "
        f"rank={params.rank} local_rank={params.local_rank} "
        f"world_size={params.world_size} "
        f"cuda_count={torch.cuda.device_count()} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    
    if params.distributed:
        assert torch.cuda.is_available()
        torch.cuda.set_device(params.local_rank)
        dist.init_process_group(backend="nccl")
        params.device = f"cuda:{params.local_rank}"
    else:
        params.device = "cuda" if torch.cuda.is_available() else "cpu"
    params.multi_gpu = params.world_size > 1
    params.is_master = params.rank == 0
    return params

def is_main_process(params):
    return (not getattr(params, "distributed", False)) or params.rank == 0


def get_params():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--seed", type=int, default=-1, help="-1 uses time() as seed")
    parser.add_argument("--resume", default="", help="Path to checkpoint .pt file")

    # Logging
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=10_000)

    parser.add_argument("--data_path", type=str, required=False, default=None)
    parser.add_argument("--random_data_path", type=str, required=False)

    user = getpass.getuser()
    parser.add_argument("--dump_path", default=f"/checkpoint/{user}/dumped")
    parser.add_argument("--exp_name", default="debug_pretrain")
    parser.add_argument("--resume_from_checkpoint", default=None, type=str)

    # Model args
    parser.add_argument("--enc_emb_dim", type=int, default=1024)
    parser.add_argument("--n_enc_layers", type=int, default=4)
    parser.add_argument("--n_enc_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--attention_dropout", type=float, default=0)
    parser.add_argument(
        "--angular_emb",
        type=bool_flag,
        default=False,
        help="Whether to use xy coordinate embeddings",
    )

    parser.add_argument(
        "--compile", type=bool_flag, default=True, help="Use torch.compile?"
    )

    # Optimizer args
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam_warmup,lr=0.00001,warmup_updates=1000,weight_decay=0.001",
        help="Optimizer (SGD / RMSprop / Adam, etc.)",
    )
    parser.add_argument(
        "--timescale", type=int, default=40, help="How fast to decay the inv sqrt lr."
    )
    parser.add_argument(
        "--dtype", default="float16", choices=["float32", "float16", "bfloat16"]
    )

    # Training args
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument(
        "--eval_samples", type=int, default=1000, help="Number of evaluation samples"
    )
    parser.add_argument("--train_samples", type=int, default=2000000)
    parser.add_argument("--num_train_epochs", type=int, default=3)

    parser.add_argument("--shuffle", type=bool_flag, default=True)
    parser.add_argument("--workers", type=int, default=8, help="CPU workers for data")

    parser.add_argument(
        "--master_port", type=int, default=int(os.environ.get("MASTER_PORT", 10035))
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--is_master", type=bool_flag, default=True)
    parser.add_argument(
        "--multi_gpu", type=bool_flag, default=False, help="Run on multiple GPUs"
    )

    parser.add_argument("--task", type=str, default="lattice")
    parser.add_argument("--model", type=str, default="encoder")
    parser.add_argument("--Q", type=int, default=-1)

    parser.add_argument("--B", help="Angular Embedding Scale", type=int, default=1)
    parser.add_argument(
        "--K",
        help="Number of precision dimension in Angular embedding",
        type=int,
        default=1,
    )

    parser.add_argument("--max_hours", type=float, default=72, help="Max time allowed")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--checkpoint_model", type=bool_flag, default=False)
    parser.add_argument("--wandb", type=bool_flag, default=False)
    parser.add_argument("--wandb_primary_key", type=str, default="exp_id")
    parser.add_argument("--tqdm", type=bool_flag, default=True)
    parser.add_argument("--copy_data", type=bool_flag, default=False)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--code_len", type=int, help="code length")
    parser.add_argument("--standard_only", type=bool_flag, default=True)
    parser.add_argument("--col_periods", type=str, default="")
    parser.add_argument("--row_periods", type=str, default="")
    parser.add_argument("--representation", type=str,default="G")
    parser.add_argument("--patch_rows",type=int,default=1)
    parser.add_argument("--patch_cols",type=int,default=1)
    parser.add_argument("--view_seed",type=int,default=0)
    parser.add_argument("--view_determinist",type=bool,default=False)
    parser.add_argument("--eval_num_views", type=int, default=1)
    params, unknown = parser.parse_known_args()

    if "qc" in params.task.split("-") or "mdpc" in params.task.split("-"):
        params = parse_qc_args(unknown, params)
    elif params.task.startswith("code-dist") or params.task.startswith("view-"):
        params = parse_goppa_args(unknown, params)
    elif params.task.startswith("code-complete"):
        params = parse_codecomplete_args(unknown, params)

    return params


def get_compute_metrics(params):
    if params.task.startswith("code-dist") or params.task.startswith("view"):
        return compute_classification_metrics_per_cat
    elif params.task.startswith("code-ident"):
        return compute_accuracy
    else:
        return compute_accuracy_with_mask


if __name__ == "__main__":
    
    params = get_params()

    if params.seed < 0:
        params.seed = int(time()) % 1000000
    
    try_load_params(params)

    params = setup_distributed(params)
    
    logger = initialize_exp(params)
    train_dataset, eval_dataset = get_datasets(params)

    params.model_output_dim = 1
    params.model_output_len = 1
    params.output_vocab_size = 1

    model = get_model(params).to(params.device)

    if is_main_process(params):
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

    report_to = init_wandb(params) if is_main_process(params) else None

    logger.info(
        f"rank={params.rank} local_rank={params.local_rank} "
        f"world_size={params.world_size} visible_gpus={torch.cuda.device_count()}"
    )

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
        multi_gpu=params.distributed,  
        dtype=params.dtype,
        max_grad_norm=params.clip_grad_norm,
        compile=params.compile,
        optimizer=params.optimizer,
        resume_from_checkpoint=params.resume_from_checkpoint,
    )

    callbacks = []

    trainer = Trainer(
        model=model,
        training_args=training_args,
        args=params,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=(
            train_dataset.dataset.collate_fn
            if isinstance(train_dataset, Subset)
            else train_dataset.collate_fn
        ),
        compute_metrics=get_compute_metrics(params),
        callbacks=callbacks,
    )


    loss_history = None
    if params.num_train_epochs > 0:
        loss_history = trainer.train()
    else:
        trainer.evaluate()

    if is_main_process(params) and loss_history is not None:
        end_wandb(params)

    if params.distributed:
        dist.barrier()
        dist.destroy_process_group()