from pathlib import Path
import os
import argparse

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from matplotlib.ticker import MaxNLocator
from src.model import get_model


# ============================================================
# Drawing utilities
# ============================================================

def draw_low_weight_rows(
    ax,
    A_bin,
    t,
    color="red",
    linewidth=1.5,
    ignore_zero_rows=True,
):
    """
    Highlight rows whose Hamming weight is < 2t.
    """
    A_bin = np.asarray(A_bin)
    k, n = A_bin.shape
    row_weights = A_bin.sum(axis=1)

    for i, w in enumerate(row_weights):
        if w < 2 * t and (not ignore_zero_rows or w > 0):
            rect = patches.Rectangle(
                (-0.5, i - 0.5),
                n,
                1,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)



# ============================================================
# Attention extraction
# ============================================================

def collect_attention_maps_with_hooks(model, inputs):
    """
    Collect attention maps from all encoder layers using forward hooks.

    Expected attention shape per layer:
        (B, H, k, k)

    Returns
    -------
    list[torch.Tensor]
        One tensor per layer.
    """
    model.eval()

    device = next(model.parameters()).device
    x = inputs.to(device).float()

    valid_rows = x.abs().sum(dim=-1) != 0

    attn_list = [None] * len(model.layers)
    hooks = []

    for layer_idx, layer in enumerate(model.layers):

        def make_hook(idx):
            def hook(module, inp, out):
                # In your original code, inp[0] contains the attention tensor.
                attn_list[idx] = inp[0].detach().cpu()
            return hook

        hooks.append(
            layer.attn.attn_dropout.register_forward_hook(
                make_hook(layer_idx)
            )
        )

    with torch.no_grad():
        _ = model(
            x,
            labels=None,
            key_padding_mask=valid_rows,
        )

    for hook in hooks:
        hook.remove()

    if any(attn is None for attn in attn_list):
        raise RuntimeError(
            "Some attention maps were not collected. "
            "Check that model.layers[*].attn.attn_dropout exists."
        )

    return attn_list


# ============================================================
# Plotting
# ============================================================

def save_attention_map_with_matrix(
    n,
    m,
    representation,
    attn_list,
    A_bin,
    output_dir,
    t,
    layer_idx=-1,
    head="mean",
    example_idx=0,
    show_every=8,
    label="goppa",
    dpi=125,
    ignore_zero_rows=True,
    show=True,
):
    """
    Save and optionally display a figure with:
      - left: binary matrix seen by the model
      - right: attention map

    Parameters
    ----------
    attn_list : list[torch.Tensor]
        Attention maps from collect_attention_maps_with_hooks(...).
        Each element has shape (B, H, k, k).

    A_bin : np.ndarray, shape (k, n)
        Binary matrix seen by the model.

    sample_idx : int
        Index used in the output filename.

    output_dir : str or Path
        Output directory.

    t : int
        Goppa degree. Used to highlight rows with weight < 2t.

    layer_idx : int
        Layer index to visualize.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Extract attention
    # ----------------------------

    attention = attn_list[layer_idx][example_idx]  # shape: (H, k, k)

    if head == "mean":
        attention = attention.mean(dim=0)
        head_label = "mean"
    else:
        attention = attention[int(head)]
        head_label = str(head)

    attention = attention.detach().cpu().numpy()
    A_bin = np.asarray(A_bin)

    k_attn = attention.shape[0]
    k_mat, n_mat = A_bin.shape

    if attention.shape != (k_mat, k_mat):
        raise ValueError(
            f"Incompatible shapes: attention={attention.shape}, "
            f"matrix={A_bin.shape}. Expected attention shape ({k_mat}, {k_mat})."
        )

    # ----------------------------
    # Build figure
    # ----------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 12),
        constrained_layout=True,
    )

    ax0, ax1 = axes

    # ============================
    # Left: binary matrix
    # ============================

    ax0.set_title("Matrix seen by the model", fontsize=20)
    ax0.matshow(
        A_bin,
        interpolation="none",
        vmin=0,
        vmax=1,
        cmap="gray_r",
    )

    draw_low_weight_rows(
        ax0,
        A_bin,
        t,
        ignore_zero_rows=ignore_zero_rows,
    )

    ax0.set_xlabel("Column")
    ax0.set_ylabel("Row")

    ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax0.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax0.set_xticks(np.arange(0, n_mat, show_every))
    ax0.set_yticks(np.arange(0, k_mat, show_every))

    ax0.set_xticks(np.arange(-0.5, n_mat, 1), minor=True)
    ax0.set_yticks(np.arange(-0.5, k_mat, 1), minor=True)

    ax0.grid(which="minor", linestyle="-", linewidth=0.5)
    ax0.tick_params(which="minor", bottom=False, left=False)

    # ============================
    # Right: attention map
    # ============================

    white_red = mcolors.LinearSegmentedColormap.from_list(
        "white_red",
        ["white", "red"],
    )

    vmax = np.nanmax(attention)
    if vmax <= 0 or np.isnan(vmax):
        vmax = 1.0

    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    im = ax1.matshow(
        attention,
        interpolation="none",
        aspect="auto",
        cmap=white_red,
        norm=norm,
    )

    cbar = fig.colorbar(im, ax=ax1)
    cbar.ax.set_title("Attention", pad=8, fontsize=10)

    ax1.set_title(
        f"Attention map",
        fontsize=20,
    )

    ax1.set_xlabel("Key row")
    ax1.set_ylabel("Query row")

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_xticks(np.arange(0, k_attn, show_every))
    ax1.set_yticks(np.arange(0, k_attn, show_every))

    ax1.set_xticks(np.arange(-0.5, k_attn, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, k_attn, 1), minor=True)

    ax1.grid(which="minor", linestyle="-", linewidth=0.5)
    ax1.tick_params(which="minor", bottom=False, left=False)

    # ----------------------------
    # Save and show
    # ----------------------------

    output_file = output_dir / (
        f"attn_{representation}_{label}_N{n}_M{m}_T{t}_layer{layer_idx}.png"
    )

    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {output_file}")

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# Model configuration
# ============================================================

def build_params(
    n,
    m,
    t,
    representation,
    device="cuda",
    task="code-dist-all-goppa",
):
    """
    Build the parameter object expected by get_model(...).

    This replaces the large Namespace string from the original script.
    """
    k = n - m * t

    if k <= 0:
        raise ValueError(
            f"Invalid parameters: k = n - m*t = {n} - {m}*{t} = {k}. "
            "Expected k > 0."
        )

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


def load_model_from_path(
    model_path,
    n,
    m,
    t,
    representation,
    device="cuda",
    task="code-dist-all-goppa",
):
    """
    Build the model and load the checkpoint from an explicit path.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    params = build_params(
        n=n,
        m=m,
        t=t,
        representation=representation,
        device=device,
        task=task,
    )

    model = get_model(params)

    ckpt = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = {
        key.replace("module.", ""): value
        for key, value in ckpt["model"].items()
    }

    model.load_state_dict(state_dict)
    model.to(params.device)
    model.eval()

    return model


# ============================================================
# Dataset utilities
# ============================================================

def random_path_from_goppa_path(goppa_dataset_path):
    """
    Build the random dataset path by replacing every occurrence of 'goppa'
    by 'random'.

    Example:
        ./data/dataset_goppa_64_H5/AT_goppa_nmt_64_6_2/dataset.h5

    becomes:
        ./data/dataset_random_64_H5/AT_random_nmt_64_6_2/dataset.h5
    """
    return Path(str(goppa_dataset_path).replace("goppa", "random"))


def check_dataset_shape(dataset_path, n, m, t):
    """
    Check that the first matrix has the expected shape:
        (k, m*t), where k = n - m*t.
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    expected_shape = (n - m * t, m * t)

    with h5py.File(dataset_path, "r") as f:
        if "G" not in f:
            raise KeyError(f"Dataset {dataset_path} does not contain key 'G'.")

        actual_shape = f["G"][0].shape

    if actual_shape != expected_shape:
        raise ValueError(
            f"Shape mismatch for dataset {dataset_path}: "
            f"actual shape={actual_shape}, expected shape={expected_shape}. "
            "Check n, m, t, representation and dataset path."
        )

    # print(f"Dataset OK: {dataset_path}")
    # print(f"Matrix shape: {actual_shape}")


# ============================================================
# Experiment
# ============================================================

def run_attention_experiment(
    model,
    dataset_path,
    label,
    n,
    m,
    t,
    representation,
    output_dir,
    batch_size,
    sample_idx,
    layer_indices=(0, 1, 2, 3),
    head="mean",
    show_every=8,
    show=True,
):
    """
    Run one attention experiment on one sample, for several layers.

    Parameters
    ----------
    sample_idx : int
        Single sample index to visualize.

    layer_indices : iterable[int]
        Layers to plot, e.g. (0, 1, 2, 3).
    """
    dataset_path = Path(dataset_path)

    check_dataset_shape(
        dataset_path=dataset_path,
        n=n,
        m=m,
        t=t,
    )

    if batch_size <= sample_idx:
        raise ValueError(
            f"batch_size={batch_size} is too small for sample index {sample_idx}. "
            f"Use batch_size >= {sample_idx + 1}."
        )

    with h5py.File(dataset_path, "r") as f:
        total_available = len(f["G"])
        effective_batch_size = min(batch_size, total_available)

        if effective_batch_size <= sample_idx:
            raise ValueError(
                f"Dataset contains only {total_available} examples, "
                f"but sample index {sample_idx} was requested."
            )

        Gs = f["G"][:effective_batch_size].astype(np.float32)

    inputs = torch.from_numpy(Gs)

    # One forward pass collects all layers.
    attn_list = collect_attention_maps_with_hooks(
        model=model,
        inputs=inputs,
    )

    for layer_idx in layer_indices:
        save_attention_map_with_matrix(
            n=n,
            m=m,
            representation=representation,
            attn_list=attn_list,
            A_bin=Gs[sample_idx],
            output_dir=output_dir,
            t=t,
            layer_idx=layer_idx,
            head=head,
            label=label,
            example_idx=sample_idx,
            show_every=show_every,
            show=show,
        )


# ============================================================
# Main
# ============================================================

def main():
    # ----------------------------
    # Main parameters
    # ----------------------------

    n = 64
    m = 6
    t = 2
    checkpoint_type = "standard"
    representation = "A"

    goppa_dataset_path = (
        f"./data/dataset_goppa_{n}_H5/"
        f"{representation}_goppa_nmt_{n}_{m}_{t}/"
        f"dataset_10K.h5"
    )

    random_dataset_path = random_path_from_goppa_path(
        goppa_dataset_path
    )

    checkpoint_type = "standard"

    if checkpoint_type == "standard":
        model_path = (
            f"./checkpoint/debug_pretrain/"
            f"{representation}_model_Goppa_N{n}_T{t}_M{m}/"
            f"checkpoint.pth"
        )
    else :
        model_path = (
            f"./checkpoint/debug_pretrain/"
            f"{representation}_model_GoppaAll_Nmax{n}_T{t}_M{m}/"
            f"checkpoint.pth"
        )

    task = "code-dist-all-goppa"

    device = "cuda"

    output_dir = "./out/attention_maps/"

    # ----------------------------
    # Figure parameters
    # ----------------------------

    layer_idx = [0,1,2,3]
    head = "mean"
    show_every = 8
    show_figures = False

    # ----------------------------
    # Experiment parameters
    # ----------------------------

    goppa_batch_size = 1
    random_batch_size = 1

    samples_to_plot = 0

    # ----------------------------
    # Load model
    # ----------------------------

    print(f"Loading model: {model_path}")

    model = load_model_from_path(
        model_path=model_path,
        n=n,
        m=m,
        t=t,
        representation=representation,
        device=device,
        task=task,
    )

    # ----------------------------
    # Run Goppa experiment
    # ----------------------------

    run_attention_experiment(
        model=model,
        dataset_path=goppa_dataset_path,
        label="goppa",
        n=n,
        m=m,
        t=t,
        representation=representation,
        output_dir=output_dir,
        batch_size=goppa_batch_size,
        sample_idx=samples_to_plot,
        layer_indices=layer_idx,
        head=head,
        show_every=show_every,
        show=show_figures,
    )

    # ----------------------------
    # Run random experiment
    # ----------------------------

    run_attention_experiment(
        model=model,
        dataset_path=random_dataset_path,
        label="random",
        n=n,
        m=m,
        t=t,
        representation=representation,
        output_dir=output_dir,
        batch_size=random_batch_size,
        sample_idx=samples_to_plot,
        layer_indices=layer_idx,
        head=head,
        show_every=show_every,
        show=show_figures,
    )


if __name__ == "__main__":
    main()