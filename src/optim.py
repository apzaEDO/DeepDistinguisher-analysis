# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
import math
import inspect

import torch
from torch import optim
import numpy as np

torch.optim.Adam


class AdamWScheduleFree(torch.optim.Optimizer):
    r"""
    Schedule-Free AdamW
    As the name suggests, no scheduler is needed with this optimizer.
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    This optimizer requires that .train() and .eval() be called before the
    beginning of training and evaluation respectively. The optimizer should
    also be placed in eval mode when saving checkpoints.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        warmup_steps (int): Enables a linear learning rate warmup (default 0).
        r (float): Use polynomial weighting in the average
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
        foreach (bool): Use a foreach-backed implementation of the optimizer.
            Should be significantly faster, but will have higher peak memory
            usage (default True if supported in your PyTorch version).
    """

    def __init__(
        self,
        params,
        lr=0.0025,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_steps=0,
        r=0.0,
        weight_lr_power=2.0,
        foreach=hasattr(torch, "_foreach_mul_"),
    ):

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            r=r,
            k=0,
            warmup_steps=warmup_steps,
            train_mode=True,
            weight_sum=0.0,
            lr_max=-1.0,
            weight_lr_power=weight_lr_power,
            weight_decay=weight_decay,
            foreach=foreach,
        )
        super().__init__(params, defaults)

    def eval(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p.data to x
                        p.data.lerp_(end=state["z"], weight=1 - 1 / beta1)
                group["train_mode"] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1, _ = group["betas"]
            if not train_mode:
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        # Set p.data to y
                        p.data.lerp_(end=state["z"], weight=1 - beta1)
                group["train_mode"] = True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            k = group["k"]
            r = group["r"]
            warmup_steps = group["warmup_steps"]
            weight_lr_power = group["weight_lr_power"]

            if k < warmup_steps:
                sched = (k + 1) / warmup_steps
            else:
                sched = 1.0

            bias_correction2 = 1 - beta2 ** (k + 1)
            lr = group["lr"] * sched * math.sqrt(bias_correction2)

            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            weight = ((k + 1) ** r) * (lr_max**weight_lr_power)
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            if not group["train_mode"]:
                raise Exception("Not in train mode!")

            active_p = [p for p in group["params"] if p.grad is not None]

            for p in active_p:
                if "z" not in self.state[p]:
                    self.state[p]["z"] = torch.clone(p.data)
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)

            if group["foreach"] and len(active_p) > 0:
                y, grad, exp_avg_sq, z = zip(
                    *[
                        (
                            p.data,
                            p.grad,
                            self.state[p]["exp_avg_sq"],
                            self.state[p]["z"],
                        )
                        for p in active_p
                    ]
                )

                # Decay the first and second moment running average coefficient
                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1 - beta2)
                denom = torch._foreach_sqrt(exp_avg_sq)
                torch._foreach_add_(denom, eps)

                # Normalize grad in-place for memory efficiency
                torch._foreach_div_(grad, denom)

                # Weight decay calculated at y
                if decay != 0:
                    torch._foreach_add_(grad, y, alpha=decay)

                # These operations update y in-place,
                # without computing x explicitly.
                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad, alpha=lr * (beta1 * (1 - ckp1) - 1))

                # z step
                torch._foreach_sub_(z, grad, alpha=lr)
            else:
                for p in active_p:
                    y = p.data  # Notation to match theory
                    grad = p.grad.data

                    state = self.state[p]

                    z = state["z"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(eps)

                    # Reuse grad buffer for memory efficiency
                    grad_normalized = grad.div_(denom)

                    # Weight decay calculated at y
                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    # These operations update y in-place,
                    # without computing x explicitly.
                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))

                    # z step
                    z.sub_(grad_normalized, alpha=lr)

            group["k"] = k + 1
        return loss


class Adam(optim.Optimizer):
    """
    Same as https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py,
    without amsgrad, with step in a tensor, and states initialization in __init__.
    It was important to add `.item()` in `state['step'].item()`.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0  # torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """
        Step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # if group['weight_decay'] != 0:
                #     grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                # denom = exp_avg_sq.sqrt().clamp_(min=group['eps'])

                bias_correction1 = 1 - beta1 ** state["step"]  # .item()
                bias_correction2 = 1 - beta2 ** state["step"]  # .item()
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    p.data.add_(-group["weight_decay"] * group["lr"], p.data)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class AdamWithWarmup(Adam):
    """
    Adam with a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`).
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = lr
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_updates=4000,
        warmup_init_lr=1e-7,
    ):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = lr
        self.lr_step = (lr - warmup_init_lr) / warmup_updates

        # total number of updates
        for param_group in self.param_groups:
            param_group["num_updates"] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.warmup_end_lr

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group["num_updates"] += 1
            param_group["lr"] = self.get_lr_for_step(param_group["num_updates"])


class AdamInverseSqrtWithWarmup(Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_updates=4000,
        warmup_init_lr=1e-7,
        exp_factor=0.5,
    ):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.exp_factor = exp_factor
        self.decay_factor = warmup_end_lr * warmup_updates**self.exp_factor

        # total number of updates
        for param_group in self.param_groups:
            param_group["num_updates"] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates**-self.exp_factor)

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group["num_updates"] += 1
            param_group["lr"] = self.get_lr_for_step(param_group["num_updates"])


class AdamCosineWithWarmup(Adam):
    """
    Assign LR based on a cyclical schedule that follows the cosine function.
    See https://arxiv.org/pdf/1608.03983.pdf for details.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``).
    During warmup::
      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]
    After warmup::
      lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(t_curr / t_i))
    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        warmup_updates=4000,
        warmup_init_lr=1e-7,
        min_lr=1e-9,
        init_period=1000000,
        period_mult=1,
        lr_shrink=0.75,
        lr_shrink_min=0.75,
        smooth=False,
    ):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.smooth = smooth
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, apply cosine scheduler
        self.min_lr = min_lr
        self.max_lr = lr
        self.period = init_period
        self.period_mult = period_mult
        self.lr_shrink = lr_shrink
        self.lr_shrink_min = lr_shrink_min

        assert not self.smooth or self.period_mult == 1

        # total number of updates
        for param_group in self.param_groups:
            param_group["num_updates"] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            t = num_updates - self.warmup_updates
            if self.period_mult == 1:
                if self.smooth:
                    pid = math.floor(t / self.period - 1 / 2)
                else:
                    pid = math.floor(t / self.period)
                t_i = self.period
                t_curr = t - (self.period * pid)
            else:
                pid = math.floor(
                    math.log(
                        1 - t / self.period * (1 - self.period_mult), self.period_mult
                    )
                )
                t_i = self.period * (self.period_mult**pid)
                t_curr = (
                    t
                    - (1 - self.period_mult**pid) / (1 - self.period_mult) * self.period
                )
            lr_shrink = self.lr_shrink**pid
            lr_shrink_min = self.lr_shrink_min**pid
            min_lr = self.min_lr * lr_shrink_min
            max_lr = self.max_lr * lr_shrink
            if max_lr < min_lr:
                max_lr = min_lr
            if self.smooth:
                return min_lr + 0.5 * (max_lr - min_lr) * (
                    1 + math.cos(2 * math.pi * t_curr / t_i)
                )
            else:
                return min_lr + 0.5 * (max_lr - min_lr) * (
                    1 + math.cos(math.pi * t_curr / t_i)
                )

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group["num_updates"] += 1
            param_group["lr"] = self.get_lr_for_step(param_group["num_updates"])


class ConstantWithWarmup(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps):
        self.optimizer = optimizer
        self.original_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.warmup_steps = warmup_steps
        self.steps = 0

        # Initialize LR to 0.0
        self.set_lr([0.0 for _ in self.original_lrs])

    def set_lr(self, lrs):
        self._last_lr = lrs
        for lr, group in zip(lrs, self.optimizer.param_groups):
            group["lr"] = lr

    def get_lr(self):
        if self.steps >= self.warmup_steps:
            return self.original_lrs

        factor = self.steps / self.warmup_steps
        return [lr * factor for lr in self.original_lrs]

    def step(self):
        self.steps += 1
        lr = self.get_lr()
        self.set_lr(lr)


class InvSqrtWithWarmup(optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_steps, timescale):
        self.optimizer = optimizer
        self.original_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.timescale = timescale

        # Initialize LR to 0.0
        self.set_lr([0.0 for _ in self.original_lrs])

    def set_lr(self, lrs):
        self._last_lr = lrs
        for lr, group in zip(lrs, self.optimizer.param_groups):
            group["lr"] = lr

    def get_lr(self):
        # warmup
        if self.steps <= self.warmup_steps:
            factor = self.steps / self.warmup_steps
            return [lr * factor for lr in self.original_lrs]

        # Inverse reverse sqrt: https://github.com/google-research/big_vision/blob/3b8e5ab6ad4f96e32b32826f9e1b8fd277914f9c/big_vision/utils.py#L1062C32-L1062C32
        return [
            lr
            / np.sqrt(
                (self.steps + self.timescale - self.warmup_steps) / self.timescale
            )
            for lr in self.original_lrs
        ]

    def step(self):
        self.steps += 1
        lr = self.get_lr()
        self.set_lr(lr)


def get_optimizer(parameters, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[: s.find(",")]
        optim_params = {}
        for x in s[s.find(",") + 1 :].split(","):
            split = x.split("=")
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == "adadelta":
        optim_fn = optim.Adadelta
    elif method == "adagrad":
        optim_fn = optim.Adagrad
    elif method == "adam":
        optim_fn = Adam
        optim_params["betas"] = (
            optim_params.get("beta1", 0.9),
            optim_params.get("beta2", 0.999),
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_warmup":
        optim_fn = AdamWithWarmup
        optim_params["betas"] = (
            optim_params.get("beta1", 0.9),
            optim_params.get("beta2", 0.999),
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_inverse_sqrt":
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params["betas"] = (
            optim_params.get("beta1", 0.9),
            optim_params.get("beta2", 0.999),
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_cosine":
        optim_fn = AdamCosineWithWarmup
        optim_params["smooth"] = False
        optim_params["betas"] = (
            optim_params.get("beta1", 0.9),
            optim_params.get("beta2", 0.999),
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_smooth_cosine":
        optim_fn = AdamCosineWithWarmup
        optim_params["smooth"] = True
        optim_params["betas"] = (
            optim_params.get("beta1", 0.9),
            optim_params.get("beta2", 0.999),
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adamax":
        optim_fn = optim.Adamax
    elif method == "asgd":
        optim_fn = optim.ASGD
    elif method == "rmsprop":
        optim_fn = optim.RMSprop
    elif method == "rprop":
        optim_fn = optim.Rprop
    elif method == "sgd":
        optim_fn = optim.SGD
        assert "lr" in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getfullargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ["self", "params"]
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception(
            'Unexpected parameters: expected "%s", got "%s"'
            % (str(expected_args[2:]), str(optim_params.keys()))
        )

    return optim_fn(parameters, **optim_params)
