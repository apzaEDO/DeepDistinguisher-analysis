from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.model import get_model


# ============================================================
# Saliency functions
# ============================================================

def low_vs_nonlow_row_saliency(saliency, A_bin, t, mode="abs_mean"):
    """
    Compare row-level saliency between Goppa-incompatible rows and
    Goppa-compatible rows.

    A row is considered Goppa-incompatible if its Hamming weight satisfies

        0 < w_H(row) < 2t.

    A row is considered Goppa-compatible if

        w_H(row) >= 2t.

    Parameters
    ----------
    saliency : array-like
        Entry-wise saliency map with the same shape as A_bin.

    A_bin : array-like
        Binary input matrix A.

    t : int
        Goppa polynomial degree.

    mode : str
        Row aggregation mode. Supported values are:
        - "abs_sum"
        - "abs_mean"
        - "signed_sum"
        - "signed_mean"

    Returns
    -------
    dict
        Statistics comparing saliency scores on incompatible and compatible rows.
    """
    S = np.asarray(saliency)
    A = np.asarray(A_bin)

    if mode == "abs_sum":
        row_saliency = np.abs(S).sum(axis=1)
    elif mode == "abs_mean":
        row_saliency = np.abs(S).mean(axis=1)
    elif mode == "signed_sum":
        row_saliency = S.sum(axis=1)
    elif mode == "signed_mean":
        row_saliency = S.mean(axis=1)
    else:
        raise ValueError(
            "Unknown mode. Expected one of: "
            "'abs_sum', 'abs_mean', 'signed_sum', 'signed_mean'."
        )

    row_weights = A.sum(axis=1)

    # Zero rows are ignored, since they may correspond to padding.
    valid = row_weights > 0
    low = (row_weights < 2 * t) & valid
    nonlow = (row_weights >= 2 * t) & valid

    stats = {
        "n_low": int(low.sum()),
        "n_nonlow": int(nonlow.sum()),
        "mean_low": float(np.nan) if low.sum() == 0 else float(row_saliency[low].mean()),
        "mean_nonlow": float(np.nan) if nonlow.sum() == 0 else float(row_saliency[nonlow].mean()),
        "max_low": float(np.nan) if low.sum() == 0 else float(row_saliency[low].max()),
        "max_nonlow": float(np.nan) if nonlow.sum() == 0 else float(row_saliency[nonlow].max()),
    }

    if stats["n_low"] > 0 and stats["n_nonlow"] > 0:
        stats["ratio_mean_low_nonlow"] = (
            stats["mean_low"] / (stats["mean_nonlow"] + 1e-12)
        )
    else:
        stats["ratio_mean_low_nonlow"] = float(np.nan)

    return stats


def input_saliency_map(model, inputs, example_idx=0, use_grad_times_input=True):
    """
    Compute a gradient-based saliency map for one input example.

    The saliency is computed from the gradient of the model logit with respect
    to the input entries. If use_grad_times_input is enabled, the returned score is

        grad * (input - 0.5),

    which gives a centered signed attribution for binary inputs.
    """
    model.eval()

    device = next(model.parameters()).device

    x = inputs.detach().clone().to(device).float()
    x.requires_grad_(True)

    # Non-zero rows are treated as valid rows.
    valid_rows = x.abs().sum(dim=-1) != 0

    cache = {}

    def hook_head(module, inp, out):
        cache["logits"] = out

    handle = model.head.register_forward_hook(hook_head)

    model.zero_grad(set_to_none=True)

    _ = model(
        x,
        labels=None,
        key_padding_mask=valid_rows,
    )

    logits = cache["logits"]

    if logits.ndim == 2:
        logit = logits[example_idx, 0]
    else:
        logit = logits[example_idx]

    logit.backward()
    handle.remove()

    grad = x.grad[example_idx]
    inp = x.detach()[example_idx]

    if use_grad_times_input:
        saliency = grad * (inp - 0.5)
    else:
        saliency = grad

    return saliency.detach().cpu(), float(logit.item())


# ============================================================
# Model configuration
# ============================================================

def build_params(n, m, t, representation, checkpoint_kind, device="cuda"):
    """
    Build the parameter namespace expected by get_model(...).

    Parameters
    ----------
    n : int
        Code length.

    m : int
        Extension degree.

    t : int
        Goppa polynomial degree.

    representation : str
        Input representation used by the model.

    checkpoint_kind : str
        Type of checkpoint to load. Supported values are:
        - "standard"
        - "all"

    device : str
        Device used for model evaluation.
    """
    import argparse

    k = n - m * t

    if k <= 0:
        raise ValueError(
            f"Invalid parameters: k = n - m*t = {n} - {m}*{t} = {k}. "
            "Expected k > 0."
        )

    if checkpoint_kind == "standard":
        task = "code-dist-goppa"
    else:
        task = "code-dist-all-goppa"

    return argparse.Namespace(
        seed=15388,
        resume="",
        log_every=10,
        val_every=500,
        save_every=10000,
        data_path="",
        random_data_path=None,
        dump_path="./checkpoint",
        exp_name="debug_pretrain",
        resume_from_checkpoint=None,

        enc_emb_dim=1024,
        n_enc_layers=4,
        n_enc_heads=4,
        dropout=0,
        attention_dropout=0,
        angular_emb=False,
        compile=False,

        optimizer="adam_warmup,lr=0.00001,warmup_updates=1000,weight_decay=0.001",
        timescale=40,
        dtype="float16",
        clip_grad_norm=5.0,

        train_batch_size=32,
        val_batch_size=48,
        eval_samples=1000,
        train_samples=20000,
        num_train_epochs=3,
        shuffle=True,
        workers=8,

        master_port=10035,
        local_rank=-1,
        device=device,
        is_master=True,
        multi_gpu=False,

        task=task,
        model="encoder",
        Q=2,
        B=1,
        K=1,
        max_hours=72,

        exp_id="",
        checkpoint_model=False,
        wandb=False,
        wandb_primary_key="exp_id",
        tqdm=True,
        copy_data=False,
        tag="",

        code_len=n,
        standard_only=True,
        col_periods="",
        row_periods="",

        representation=representation,
        m_alt=m,
        t_alt=t,

        beta_dist="uniform",
        alpha_dist="uniform",
        data_bundle_size=100,

        model_input_dim=m * t,
        model_input_len=k,
        model_output_dim=1,
        model_output_len=1,
        output_vocab_size=1,

        param_sets=None,
    )


def get_checkpoint_path(n, m, t, representation, checkpoint_kind="standard"):
    """
    Return the checkpoint path for a given model configuration.
    """
    if checkpoint_kind == "standard":
        folder = f"{representation}_model_Goppa_N{n}_T{t}_M{m}"
    elif checkpoint_kind == "all":
        folder = f"{representation}_model_GoppaAll_Nmax{n}_T{t}_M{m}"
    else:
        raise ValueError("checkpoint_kind must be either 'standard' or 'all'.")

    return Path("./checkpoint/debug_pretrain") / folder / "checkpoint.pth"


def resolve_dataset_path(dataset_path, dataset_kind="random"):
    """
    Resolve the dataset path from a reference Goppa dataset path.

    The convention is:
      - for Goppa: use dataset_path as given;
      - for random: replace every occurrence of 'goppa' by 'random'.

    Example
    -------
    Input:
        ./data/dataset_goppa_64_H5/A_goppa_nmt_64_6_5/dataset.h5

    Random counterpart:
        ./data/dataset_random_64_H5/A_random_nmt_64_6_5/dataset.h5
    """
    if dataset_kind not in {"goppa", "random"}:
        raise ValueError("dataset_kind must be either 'goppa' or 'random'.")

    dataset_path = str(dataset_path)

    if dataset_kind == "goppa":
        resolved_path = Path(dataset_path)
    else:
        resolved_path = Path(dataset_path.replace("goppa", "random"))

    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_path}")

    return resolved_path


def load_model(n, m, t, representation, device="cuda", checkpoint_kind="standard"):
    """
    Build a model and load its checkpoint.
    """
    params = build_params(
        n=n,
        m=m,
        t=t,
        representation=representation,
        checkpoint_kind=checkpoint_kind,
        device=device,
    )

    model = get_model(params)

    ckpt_path = get_checkpoint_path(
        n=n,
        m=m,
        t=t,
        representation=representation,
        checkpoint_kind=checkpoint_kind,
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = {
        key.replace("module.", ""): value
        for key, value in ckpt["model"].items()
    }

    #If the model is train on full generator matrix, field pos_emb isn't attribuated
    if hasattr(model, "pos_emb") and "pos_emb.weight" not in state_dict:
        print("Warning: pos_emb.weight missing from checkpoint. Initializing it to zero.")
        state_dict["pos_emb.weight"] = torch.zeros_like(model.pos_emb.weight)

    model.load_state_dict(state_dict)
    model.to(params.device)

    return model


# ============================================================
# Experiment
# ============================================================

def run_experiment(
    n,
    m,
    t,
    representation,
    mode,
    dataset_path,
    dataset_kind="random",
    checkpoint_kind="standard",
    nb_examples=1000,
    device="cuda",
):
    """
    Run the saliency experiment for one parameter configuration.

    The experiment computes gradient-based saliency maps on several examples,
    aggregates saliency scores at the row level, and compares rows that violate
    the Goppa row-weight constraint against compatible rows.
    """
    if mode not in {"abs_sum", "abs_mean", "signed_sum", "signed_mean"}:
        raise ValueError(
            "mode must be one of: "
            "'abs_sum', 'abs_mean', 'signed_sum', 'signed_mean'."
        )

    k = n - m * t

    if k <= 0:
        raise ValueError(f"Invalid parameters: k = n - m*t = {k}. Expected k > 0.")

    model = load_model(
        n=n,
        m=m,
        t=t,
        representation=representation,
        device=device,
        checkpoint_kind=checkpoint_kind,
    )

    data_path = resolve_dataset_path(
        dataset_path=dataset_path,
        dataset_kind=dataset_kind,
    )

    print(f"Using dataset: {data_path}")

    all_ratios = []
    all_mean_low = []
    all_mean_nonlow = []

    with h5py.File(data_path, "r") as f:
        total_available = len(f["G"])
        nb = min(nb_examples, total_available)

        print(f"Evaluating {nb} examples...")

        for i in range(nb):
            G = f["G"][i].astype(np.float32)
            inputs = torch.from_numpy(G).unsqueeze(0)

            saliency, logit = input_saliency_map(
                model=model,
                inputs=inputs,
                example_idx=0,
            )

            stats = low_vs_nonlow_row_saliency(
                saliency=saliency,
                A_bin=G,
                t=t,
                mode=mode,
            )

            if not np.isnan(stats["mean_low"]):
                all_mean_low.append(stats["mean_low"])

            if not np.isnan(stats["mean_nonlow"]):
                all_mean_nonlow.append(stats["mean_nonlow"])

            if not np.isnan(stats["ratio_mean_low_nonlow"]):
                all_ratios.append(stats["ratio_mean_low_nonlow"])

    results = {
        "n": n,
        "m": m,
        "t": t,
        "k": k,
        "representation": representation,
        "mode": mode,
        "dataset_kind": dataset_kind,
        "checkpoint_kind": checkpoint_kind,
        "dataset_path": str(data_path),
        "nb_examples": nb,

        "mean_low": float(np.mean(all_mean_low)) if all_mean_low else float(np.nan),
        "std_low": float(np.std(all_mean_low)) if all_mean_low else float(np.nan),

        "mean_nonlow": float(np.mean(all_mean_nonlow)) if all_mean_nonlow else float(np.nan),
        "std_nonlow": float(np.std(all_mean_nonlow)) if all_mean_nonlow else float(np.nan),

        "mean_ratio": float(np.mean(all_ratios)) if all_ratios else float(np.nan),
        "median_ratio": float(np.median(all_ratios)) if all_ratios else float(np.nan),
    }

    return results


# ============================================================
# Plot
# ============================================================

def plot_results(results_by_t, mode, save_path=None, show=True):
    """
    Display a bar plot comparing saliency on incompatible and compatible rows.

    Parameters
    ----------
    results_by_t : dict
        Dictionary of the form:
            {
                t: {
                    "mean_low": ...,
                    "std_low": ...,
                    "mean_nonlow": ...,
                    "std_nonlow": ...
                }
            }

    mode : str
        Saliency aggregation mode.

    save_path : str or None
        If not None, save the figure to this path.

    show : bool
        If True, display the figure with plt.show().
    """
    t_values = sorted(results_by_t.keys())

    mean_low = [results_by_t[t]["mean_low"] for t in t_values]
    mean_nonlow = [results_by_t[t]["mean_nonlow"] for t in t_values]

    std_low = [results_by_t[t]["std_low"] for t in t_values]
    std_nonlow = [results_by_t[t]["std_nonlow"] for t in t_values]

    x = np.arange(len(t_values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.bar(
        x - width / 2,
        mean_low,
        width,
        label="Goppa-incompatible rows",
        color="#e03e61",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.bar(
        x + width / 2,
        mean_nonlow,
        width,
        label="Goppa-compatible rows",
        color="#2dbd75",
        edgecolor="black",
        linewidth=0.5,
    )

    if mode in {"signed_sum", "signed_mean"}:
        all_values = np.array(mean_low + mean_nonlow, dtype=float)
        max_abs = np.nanmax(np.abs(all_values))

        if max_abs == 0 or np.isnan(max_abs):
            max_abs = 1.0

        ax.set_ylim(-1.10 * max_abs, 1.10 * max_abs)

    ax.axhline(0, color="black", linewidth=1.0, zorder=1)

    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.spines["bottom"].set_color("black")

    ax.spines["left"].set_linewidth(1.1)
    ax.spines["left"].set_color("black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in t_values], fontsize=12)

    ax.tick_params(axis="x", bottom=True, length=4, width=1, labelsize=15)
    ax.tick_params(axis="y", left=True, length=4, width=1, labelsize=15)

    ax.set_xlabel("Degree of the Goppa polynomial", fontsize=20)

    if mode == "abs_mean":
        ylabel = "Mean absolute row attribution"
    elif mode == "abs_sum":
        ylabel = "Sum absolute row attribution"
    elif mode == "signed_mean":
        ylabel = "Mean signed row attribution"
    elif mode == "signed_sum":
        ylabel = "Sum signed row attribution"
    else:
        ylabel = "Row attribution"

    ax.set_ylabel(ylabel, fontsize=20)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    ax.legend(prop={"size": 20, "weight": "bold"})

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    """
    Run the saliency analysis for a selected set of parameters.
    """
    n = 64
    m = 6

    representation = "A"
    mode = "abs_mean"

    dataset_kind = "random"
    checkpoint_kind = "standard"

    nb_examples = 1000
    device = "cuda"

    save_figure = True
    show_figure = True

    figure_path = (
        f"./out/saliency_maps/"
        f"summary_{representation}_N{n}_M{m}_{dataset_kind}_{mode}.png"
    )

    results_by_t = {}

    for t in range(2, 3):
        dataset_path = (
            f"./data/dataset_goppa_{n}_H5/"
            f"{representation}_goppa_nmt_{n}_{m}_{t}/"
            f"dataset_10K.h5"
        )

        print(f"\nRunning t={t}")
        print(f"Reference dataset path: {dataset_path}")

        results = run_experiment(
            n=n,
            m=m,
            t=t,
            representation=representation,
            mode=mode,
            dataset_path=dataset_path,
            dataset_kind=dataset_kind,
            checkpoint_kind=checkpoint_kind,
            nb_examples=nb_examples,
            device=device,
        )

        results_by_t[t] = results

    plot_results(
        results_by_t=results_by_t,
        mode=mode,
        save_path=figure_path if save_figure else None,
        show=show_figure,
    )


if __name__ == "__main__":
    main()