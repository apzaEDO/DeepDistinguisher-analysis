# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from argparse import Namespace
from dataclasses import dataclass
from typing import Any
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import logging
import os
from torch.nn.utils import clip_grad_norm_
from time import time
from tqdm import tqdm
import getpass
from src.optim import get_optimizer
from src.utils import add_prefix_to_keys, concat_nested, to_device_rec
import numpy as np

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


def _to_cpu_obj(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _extract_n(batch_set):
    batch_set = _to_cpu_obj(batch_set)

    # cas déjà sous forme [ [n,m,t], [n,m,t], ... ]
    if isinstance(batch_set, list) and len(batch_set) > 0 and isinstance(batch_set[0], (list, tuple)):
        return [int(s[0]) for s in batch_set]

    # cas tensor/array [B, 3]
    if isinstance(batch_set, list) and len(batch_set) > 0 and not isinstance(batch_set[0], (list, tuple)):
        # déjà [n1, n2, ...]
        return [int(s) for s in batch_set]

    return [int(batch_set)]

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _binarize(pred):
    p = _to_numpy(pred)
    # Si déjà entier/bool -> nonzero = 1
    if p.dtype.kind in ("b", "i", "u"):
        return (p != 0).astype(np.uint8)
    # Float: si dans [0,1] on seuil à 0.5, sinon seuil à 0 (logits)
    pmin = np.nanmin(p)
    pmax = np.nanmax(p)
    thr = 0.5 if (pmin >= 0.0 and pmax <= 1.0) else 0.0
    return (p >= thr).astype(np.uint8)

@dataclass
class TrainingArguments:
    dump_path: str = (
        f"/checkpoint/{getpass.getuser()}/dumped"  # Directory for saving outputs
    )
    num_train_epochs: int = 3  # Number of training epochs
    per_device_train_batch_size: int = 8  # Batch size for training
    per_device_eval_batch_size: int = 8  # Batch size for evaluation
    optimizer: str = (
        "adam_warmup,lr=0.00001,warmup_updates=2000,weight_decay=0.001"  # Learning rate
    )
    logging_dir: str = "./logs"  # Directory for logging
    logging_steps: int = 500  # Log every X steps
    save_steps: int = 500  # Save checkpoint every X steps
    evaluation_strategy: str = "no"  # Evaluation strategy: "no", "steps", "epoch"
    eval_steps: int = 500  # Evaluate every X steps
    save_total_limit: int = 3  # Max number of checkpoints to keep
    load_best_model_at_end: bool = (
        False  # Whether to load the best model at the end of training
    )
    gradient_accumulation_steps: int = 1  # Gradient accumulation steps
    seed: int = 42  # Random seed for reproducibility
    logging_first_step: bool = False  # Whether to log the first training step
    disable_tqdm: bool = False  # Whether to disable the tqdm progress bar
    max_grad_norm: float = 5.0  # Maximum gradient norm for clipping
    dataloader_num_workers: int = 8  # Number of workers for data loading
    report_to: str = None
    local_rank: int = 0
    device: Any = "cuda"
    compile: bool = True
    multi_gpu: bool = False
    resume_from_checkpoint: str = None
    max_grad_norm: float = 5.0
    dtype: str = "float32"
    is_master: bool = None


class Trainer:
    def __init__(
        self,
        model,
        args,
        training_args,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=None,
        callbacks=None,
    ):
        self.model = model
        self.args = args
        self.training_args = training_args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks
        self.loss_history = {
            "train": [],  # liste de (step, loss)
            "eval": [],   # liste de (step, eval_loss)
        }
        # Setup device
        self.device = training_args.device
        self.model.to(self.device)

        self.is_distributed = (
            training_args.multi_gpu
            and dist.is_available()
            and dist.is_initialized()
        )
        
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1
        if training_args.is_master is None:
            self.training_args.is_master = (self.rank == 0)
        
        if self.is_distributed:
            logger.info("Using torch.nn.parallel.DistributedDataParallel ...")
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[training_args.local_rank],
                output_device=training_args.local_rank,
                broadcast_buffers=False,
            )
        
        self.train_sampler = None
        if self.is_distributed:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        
        # pas d'eval_sampler si seul le master évalue
        self.eval_sampler = None

        

        self.epoch = 0
        self.step = 0
        self.metric_name = "accuracy"
        self.metric_threshold = 0.99
        self.optimizer = get_optimizer(
            self.model.parameters(), self.training_args.optimizer
        )

        self.init_amp()

        #self.try_reload_checkpoint(checkpoint_path=training_args.resume_from_checkpoint)

        self.uncompiled_model = self.model
        if training_args.compile:
            self.model = torch.compile(self.model)
            logger.debug("Model compiled!")

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        enabled = (
            self.training_args.dtype == "float16"
            and self.training_args.device.startswith("cuda")
        )
        logger.info(f"FP16 enabled: {enabled}")
        self.scaler = torch.amp.GradScaler(self.training_args.device, enabled=enabled)
        self.amp_ctx = torch.amp.autocast(
            device_type="cuda", dtype=getattr(torch, self.training_args.dtype)
        )

    def train(self):
        self.model.train()
        # sampler = DistributedSampler(self.train_dataset, num_replicas=self.args.world_size, rank=self.args.local_rank)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.training_args.dataloader_num_workers,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            prefetch_factor=16,
        )

        for epoch in range(self.epoch, self.training_args.num_train_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            for batch in train_loader:
                # Move batch to device
                batch = to_device_rec(
                    batch,
                    self.device,
                    keys=["inputs", "inputs_G", "inputs_H", "labels", "masks", "secret_keys", "set"],
                )
                with self.amp_ctx:
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
                grad_norm = self.optimize(loss)

                stop = self.end_step(loss, grad_norm)
                if stop:
                    logger.info(f"Finishing Training")
                    return stop
            self.end_epoch()
        return self.loss_history

        
    def optimize(self, loss):
        """
        Optimize.
        """
        args = self.training_args
        optimizer = self.optimizer
        scaler = self.scaler
        model = self.model

        # regular optimization
        scaler.scale(loss).backward()

        if args.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            gradn = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        else:
            # calculate norm of gradients but don't clip them
            gradn = clip_grad_norm_(model.parameters(), float("inf"))

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        return gradn

    def end_step(self, loss, grad_norm):
        
        if self.step % (self.training_args.logging_steps*10) == 0:
            metrics = {
                "loss": loss.item(),
                #"grad_norm": grad_norm.item(),
                "epoch": self.epoch,
                "step": self.step
                #"learning_rate": self.optimizer.param_groups[0]["lr"],
                #"time": time(),
            }
            loss_val = float(loss.detach().cpu())
            self.loss_history["train"].append((self.step, loss_val))
            self.log_metrics("train", metrics)
        
        self.step += 1
        """
        if (
            self.training_args.save_steps > 0
            and self.step % self.training_args.save_steps == 0
        ):
            self.save_checkpoint()
        """
        stop = False
        """
        if (self.step) % self.training_args.eval_steps == 0:
            metrics = self.evaluate()
            self.model.train()
            stop = self.check_for_end_train(metrics)
        """
        return stop

    def end_epoch(self):
        if self.is_distributed:
            dist.barrier()
    
        if self.training_args.is_master:
            metrics = self.evaluate()
            self.save_checkpoint()
    
        if self.is_distributed:
            dist.barrier()
    
        self.model.train()
        self.epoch += 1

    def check_for_end_train(self, eval_metrics):
        metric_name = self.metric_name
        metric = eval_metrics[metric_name]
        return metric > self.metric_threshold


    @torch.no_grad()
    def evaluate(self, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = self.eval_dataset
        
        num_views = int(getattr(self.args, "eval_num_views", 1))
        if num_views > 1:
            eval_dataset = MultiViewEvalDataset(eval_dataset, num_views)
            collate_fn = eval_dataset.collate_fn
        else:
            collate_fn = self.data_collator
    
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=collate_fn,
            num_workers=self.training_args.dataloader_num_workers,
            sampler=None,
            shuffle=False,
            
        )
    
        self.model.eval()

        loss_fn_parent = torch.nn.BCEWithLogitsLoss()
        total_loss = 0.0          # loss view-level
        parent_loss_value = None  # loss parent-level
        all_scores = []
        all_labels = []
        all_parent_ids = []
        all_sets = []
    
        logger.info("Starting evaluation.")
    
        for batch in eval_loader:
            parent_id = batch.pop("parent_id", None)
            view_id = batch.pop("view_id", None)
    
            if "set" in batch:
                all_sets.extend(_extract_n(batch["set"]))
                batch.pop("set")
    
            batch = to_device_rec(
                batch,
                self.device,
                keys=["inputs", "inputs_G", "inputs_H", "labels", "masks", "secret_keys"],
            )
    
            with self.amp_ctx:
                outputs = self.uncompiled_model(**batch)
    
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
    
            if "logits" in outputs:
                scores = outputs["logits"]
                score_type = "logits"
            else:
                scores = outputs["output"].float()
                score_type = "binary"
    
            all_scores.append(scores.detach().cpu())
            all_labels.append(batch["labels"].detach().cpu())
    
            if parent_id is not None:
                if torch.is_tensor(parent_id):
                    all_parent_ids.append(parent_id.detach().cpu())
                else:
                    all_parent_ids.append(torch.as_tensor(parent_id))
    
        all_scores = torch.cat(all_scores, dim=0)   # [N, 1]
        all_labels = torch.cat(all_labels, dim=0)   # [N, 1]
    
        if len(all_parent_ids) > 0:
            all_parent_ids = torch.cat(all_parent_ids, dim=0).numpy()
        else:
            all_parent_ids = None
    
        scores_np = all_scores.numpy().reshape(-1)
        labels_np = all_labels.numpy().reshape(-1)
    
        if num_views > 1:
            if all_parent_ids is None:
                raise ValueError("parent_id manquant pour l'évaluation multi-vues")
    
            uniq_ids, inv = np.unique(all_parent_ids, return_inverse=True)
    
            agg_scores = np.zeros(len(uniq_ids), dtype=np.float32)
            agg_labels = np.zeros(len(uniq_ids), dtype=np.int64)
            counts = np.zeros(len(uniq_ids), dtype=np.int32)
    
            for i, g in enumerate(inv):
                agg_scores[g] += scores_np[i]
                agg_labels[g] = int(labels_np[i])
                counts[g] += 1
    
            agg_scores /= counts
    
            # parent-level loss seulement si on a des logits continus
            if score_type == "logits":
                agg_logits_t = torch.from_numpy(agg_scores[:, None]).float()
                agg_labels_t = torch.from_numpy(agg_labels[:, None]).float()
                parent_loss_value = loss_fn_parent(agg_logits_t, agg_labels_t).item()
    
                pred_np = (agg_scores > 0).astype(np.int64)
            else:
                # moins propre si on n'a pas de logits
                parent_loss_value = None
                pred_np = (agg_scores > 0.5).astype(np.int64)
    
            label_np = agg_labels.astype(np.int64)
            hue_for_metrics = None
    
        else:
            if score_type == "logits":
                # même au cas standard, on peut définir une parent/view loss cohérente
                logits_t = all_scores.float()
                labels_t = all_labels.float()
                parent_loss_value = loss_fn_parent(logits_t, labels_t).item()
    
                pred_np = (scores_np > 0).astype(np.int64)
            else:
                parent_loss_value = None
                pred_np = (scores_np > 0.5).astype(np.int64)
    
            label_np = labels_np.astype(np.int64)
            hue_for_metrics = None if len(all_sets) == 0 else all_sets
    
        all_preds = torch.from_numpy(pred_np[:, None])
        all_labels = torch.from_numpy(label_np[:, None])
    
        metrics = {}
    
        if self.compute_metrics:
            logger.info("Computing metrics.")
            i_metrics = self.compute_metrics((all_preds, all_labels), hue=hue_for_metrics)
            i_metrics = add_prefix_to_keys("0/", i_metrics)
            metrics.update(i_metrics)
    
        for callback in self.callbacks:
            if hasattr(callback, "on_evaluate"):
                callback.on_evaluate(self, metrics)
        
        view_loss_value = total_loss / max(len(eval_loader), 1)
        
        
        self.log_metrics(metric_key_prefix, metrics)
        
        self.model.train()
        
        return metrics
            
    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.training_args.dataloader_num_workers,
        )

        self.model.eval()
        all_preds = []

        logger.info("Starting prediction.")

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with self.amp_ctx:
                    outputs = self.uncompiled_model(**batch)
                all_preds.append(outputs["logits"])

        all_preds = torch.cat(all_preds, dim=0).cpu()
        logger.info("Prediction completed.")
        return all_preds

    def log_metrics(self, split: str, metrics: dict):
        if not self.training_args.is_master:
            return
        metrics = {f"{split}/{key}": v for key, v in metrics.items()}
        logger.info(metrics)
        if self.training_args.report_to == "wandb":
            wandb.log(metrics)

    def reduce_mean(tensor):
        if not dist.is_available() or not dist.is_initialized():
            return tensor
        t = tensor.detach().clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
        return t
    
    def save_checkpoint(self, name="checkpoint", include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.training_args.is_master:
            return

        path = os.path.join(self.training_args.dump_path, f"{name}.pth")
        logger.info("Saving %s to %s ...", name, path)
        params = self.args
        # if params is omegaconfig, convert it to namespace
        if isinstance(params, DictConfig):
            params = Namespace(**OmegaConf.to_container(self.args, resolve=True))
        data = {
            "epoch": self.epoch,
            "step": self.step,
            "params": params,
        }

        logger.info(f"Saving model parameters ...")

        data["model"] = self.uncompiled_model.state_dict()

        if include_optimizer:
            logger.info("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()

        torch.save(data, path)

    def try_reload_checkpoint(self, checkpoint_path=None, name="checkpoint"):
        """
        Reload a checkpoint.
        """

        path = os.path.join(self.training_args.dump_path, f"{name}.pth")

        if not os.path.isfile(path):
            logger.info(f"No checkpoint found at {path}.")
            path = checkpoint_path

            if path is None or not os.path.isfile(path):
                logger.info(f"No checkpoint found at {path}.")
                return

        logger.warning(f"Reloading checkpoint from {path} ...")

        data = torch.load(path, map_location="cpu")

        state_dict = data["model"]
        if not self.training_args.multi_gpu:
            # bcs of ddp, module. prefix was added
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # reload model parameters
        self.model.load_state_dict(state_dict)

        # reload optimizer and scaler
        logger.warning("Reloading checkpoint optimizer ...")
        self.optimizer.load_state_dict(data["optimizer"])

        logger.warning("Reloading gradient scaler ...")
        if "scaler" in data and data["scaler"] is not None:
            self.scaler.load_state_dict(data["scaler"])

        self.epoch = data["epoch"]

        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} ...")
