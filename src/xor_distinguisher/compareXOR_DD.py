from pathlib import Path
import multiprocessing as mp
import argparse
import copy
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.model import get_model
from src.xor_distinguisher.distinguisher import *


# ============================================================
# Model configuration / loading
# ============================================================

def infer_model_input_shape(n, m, t, representation):
    """
    For a GoppaAll checkpoint, k_max = n_max - m*t. Smaller n are evaluated
    with a shorter sequence length k = n - m*t; the last dimension stays m*t.
    """
    k = n - m * t

    if k <= 0:
        raise ValueError(
            f"Invalid parameters: k = n - m*t = {n} - {m}*{t} = {k}. "
            "Expected k > 0."
        )

    return k, m * t


def build_params(
    n,
    m,
    t,
    representation,
    checkpoint_kind="all",
    device="cuda",
):
    """
    Build the parameter object expected by get_model(...)."""
    k = n - m * t

    if k <= 0:
        raise ValueError(
            f"Invalid parameters: k = n - m*t = {n} - {m}*{t} = {k}. "
            "Expected k > 0."
        )

    model_input_len, model_input_dim = infer_model_input_shape(
        n=n,
        m=m,
        t=t,
        representation=representation,
    )

    if checkpoint_kind == "standard":
        task = "code-dist-goppa"
        param_sets = None
    elif checkpoint_kind == "all":
        task = "code-dist-all-goppa"
        param_sets = ";".join(
            f"{n_i}, {m}, {t}"
            for n_i in (32, 40, 48, 56, 64)
            if n_i > m * t and n_i <= n
        )
    else:
        raise ValueError("checkpoint_kind must be either 'standard' or 'all'.")

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

        code="goppa",
        code_len=n,
        k=k,
        standard_only=True,
        col_periods="",
        row_periods="",

        representation=representation,
        m_alt=m,
        t_alt=t,

        beta_dist="uniform",
        alpha_dist="uniform",
        data_bundle_size=100,

        model_input_dim=model_input_dim,
        model_input_len=model_input_len,
        model_output_dim=1,
        model_output_len=1,
        output_vocab_size=1,

        param_sets=param_sets,
    )


def get_checkpoint_path(n, m, t, representation, checkpoint_kind="all"):
    """
    Resolve the checkpoint path using the naming convention.
    """
    if checkpoint_kind == "standard":
        folder = f"{representation}_model_Goppa_N{n}_T{t}_M{m}"
    elif checkpoint_kind == "all":
        folder = f"{representation}_model_NGoppa_N{n}_T{t}_M{m}"
    else:
        raise ValueError("checkpoint_kind must be either 'standard' or 'all'.")
    print(Path("./checkpoint/debug_pretrain") / folder / "checkpoint.pth")
    return Path("./checkpoint/debug_pretrain") / folder / "checkpoint.pth"


def namespace_from_pickle_object(obj):
    """
    Training code usually stores an argparse.Namespace in params.pkl, but this
    also accepts plain dictionaries and dictionaries containing a 'params' key.
    """
    if isinstance(obj, dict) and "params" in obj:
        obj = obj["params"]

    if isinstance(obj, argparse.Namespace):
        return copy.deepcopy(obj)

    if isinstance(obj, dict):
        return argparse.Namespace(**copy.deepcopy(obj))

    if hasattr(obj, "__dict__"):
        return copy.deepcopy(obj)

    raise TypeError(
        f"Unsupported params.pkl object type: {type(obj)!r}. "
        "Expected argparse.Namespace, dict, or namespace-like object."
    )


def load_params_from_checkpoint_dir(checkpoint_path):
    """Load params.pkl located next to checkpoint.pth."""
    params_path = Path(checkpoint_path).with_name("params.pkl")

    if not params_path.exists():
        raise FileNotFoundError(f"params.pkl not found next to checkpoint: {params_path}")

    with params_path.open("rb") as f:
        obj = pickle.load(f)

    params = namespace_from_pickle_object(obj)
    print(f"Loaded training params from: {params_path}")
    return params


def set_missing_param(params, name, value):
    """Set a default only when params.pkl does not define the field."""
    if not hasattr(params, name):
        setattr(params, name, value)


def patch_runtime_params(
    params,
    n,
    m,
    t,
    representation,
    checkpoint_kind="all",
    device="cuda",
):
    """
    These fields are therefore intentionally overwritten here. They are not
    treated as immutable architecture fields from params.pkl, because the
    original evaluation code also overwrote them before get_model(params).
    """
    if hasattr(params, "m_alt") and str(getattr(params, "m_alt")) != str(m):
        raise ValueError(
            f"params.pkl mismatch for m_alt: got {getattr(params, 'm_alt')!r}, expected {m!r}."
        )

    if hasattr(params, "t_alt") and str(getattr(params, "t_alt")) != str(t):
        raise ValueError(
            f"params.pkl mismatch for t_alt: got {getattr(params, 't_alt')!r}, expected {t!r}."
        )

    if hasattr(params, "representation") and str(getattr(params, "representation")) != str(representation):
        raise ValueError(
            "params.pkl mismatch for representation: "
            f"got {getattr(params, 'representation')!r}, expected {representation!r}."
        )

    k = n - m * t
    if k <= 0:
        raise ValueError(f"Invalid parameters: k = {n} - {m}*{t} = {k}.")

    # Runtime-only overrides.
    params.device = device
    params.local_rank = -1
    params.multi_gpu = False
    params.is_master = True
    params.compile = False
    params.wandb = False
    params.checkpoint_model = False
    params.resume = ""
    params.resume_from_checkpoint = None

    params.task = "code-dist-all-goppa" if checkpoint_kind == "all" else "code-dist-goppa"
    params.model = "encoder"
    params.code = "goppa"
    params.code_len = n
    params.k = k
    params.m_alt = m
    params.t_alt = t
    params.representation = representation
    params.model_input_dim = m * t
    params.model_input_len = k
    params.model_output_dim = 1
    params.model_output_len = 1
    params.output_vocab_size = 1

    # Keep params.param_sets from params.pkl when it exists. 
    if not hasattr(params, "param_sets"):
        params.param_sets = None

    return params


def build_or_load_params(
    n,
    m,
    t,
    representation,
    checkpoint_path,
    checkpoint_kind="all",
    device="cuda",
    prefer_params_pkl=True,
):
    """
    Build params from params.pkl when available; otherwise fall back to the
    local hand-written configuration.
    """
    params_path = Path(checkpoint_path).with_name("params.pkl")

    if prefer_params_pkl and params_path.exists():
        params = load_params_from_checkpoint_dir(checkpoint_path)
        return patch_runtime_params(
            params=params,
            n=n,
            m=m,
            t=t,
            representation=representation,
            checkpoint_kind=checkpoint_kind,
            device=device,
        )

    if prefer_params_pkl:
        print(f"params.pkl not found at {params_path}; falling back to build_params().")

    return build_params(
        n=n,
        m=m,
        t=t,
        representation=representation,
        checkpoint_kind=checkpoint_kind,
        device=device,
    )


def load_model(
    n,
    m,
    t,
    representation,
    device="cuda",
    checkpoint_kind="all",
):
    """
    Build the model and load its checkpoint from a parameterized configuration.
    """
    ckpt_path = get_checkpoint_path(
        n=n,
        m=m,
        t=t,
        representation=representation,
        checkpoint_kind=checkpoint_kind,
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    params = build_or_load_params(
        n=n,
        m=m,
        t=t,
        representation=representation,
        checkpoint_path=ckpt_path,
        checkpoint_kind=checkpoint_kind,
        device=device,
        prefer_params_pkl=True,
    )

    print(
        "Model input shape parameters: "
        f"model_input_len={params.model_input_len}, "
        f"model_input_dim={params.model_input_dim}"
    )

    model = get_model(params)

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = {
        key.replace("module.", ""): value
        for key, value in ckpt["model"].items()
    }

    #Compatibility with model trained on G matrix. (full generator matrix)
    if hasattr(model, "pos_emb") and "pos_emb.weight" not in state_dict:
        print(
            "Warning: pos_emb.weight missing from checkpoint. "
            "Initializing it to zero."
        )
        state_dict["pos_emb.weight"] = torch.zeros_like(model.pos_emb.weight)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(params.device)

    model.eval_params = params

    return model


def code_rate(n, m, t):
    """
    Compute code rate : R = k/n = (n - mt)/n.
    """
    return (n - m * t) / n


def reorder_table(table, class_order=(1, 0)):
    table = np.asarray(table, dtype=int)
    return table[np.ix_(class_order, class_order)]


def agreement_from_table(table):
    table = np.asarray(table, dtype=int)
    return np.trace(table) / table.sum()



def load_same_order_samples(path_goppa, path_random, key="G", n_samples=1000):
    """
    Build a balanced dataset :
        X[0:n] = Goppa
        X[n:2n] = Random
        y_model(prediction) = 1 for Goppa, 0 for Random
    """
    with h5py.File(path_goppa, "r") as f:
        Ag = f[key][:n_samples]

    with h5py.File(path_random, "r") as f:
        Ar = f[key][:n_samples]

    X = np.concatenate([Ag, Ar], axis=0)

    y = np.concatenate([
        np.ones(len(Ag), dtype=np.int64),
        np.zeros(len(Ar), dtype=np.int64),
    ])

    source = np.concatenate([
        np.full(len(Ag), "goppa"),
        np.full(len(Ar), "random"),
    ])

    return X, y, source


def prepare_model_numpy_input(model, X):
    """
    Validate and return raw HDF5 matrices for the model.

    the AI model is evaluated on matrices shaped (batch, k, m*t). For a checkpoint,
    k can vary with n as long as k <= model_input_len; the feature dimension
    must remain m*t.
    """
    X = np.asarray(X)

    if X.ndim != 3:
        raise ValueError(f"Expected X with shape (N, k, m*t), got {X.shape}.")

    params = getattr(model, "eval_params", None)
    if params is None:
        return X.astype(np.float32, copy=False)

    expected_len = int(getattr(params, "model_input_len"))
    expected_dim = int(getattr(params, "model_input_dim"))

    if X.shape[2] != expected_dim:
        raise ValueError(
            "Input feature dimension is incompatible with the loaded model: "
            f"got {X.shape[2]}, expected {expected_dim}. Full shape: {X.shape}."
        )

    if X.shape[1] > expected_len:
        raise ValueError(
            "Input sequence length is larger than the loaded GoppaAll model allows: "
            f"got k={X.shape[1]}, expected at most {expected_len}. Full shape: {X.shape}."
        )

    return X.astype(np.float32, copy=False)


def predict_model_numpy(model, X, device, batch_size=256):
    model.eval()

    X_model = prepare_model_numpy_input(model, X)

    param_dtype = next(model.parameters()).dtype

    X_tensor = torch.tensor(X_model, dtype=param_dtype)
    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    preds = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)

            out = model(xb)

            if isinstance(out, dict):
                pred = out["output"]
            else:
                pred = out

            pred = pred.detach().cpu().numpy()
            pred = np.asarray(pred).reshape(-1).astype(np.int64)
            preds.append(pred)

    return np.concatenate(preds, axis=0)


def predict_xor(X, t):
    pred_xor = binary_distinguisher_on_batch(X, t=t)
    return np.asarray(pred_xor).reshape(-1).astype(np.int64)


def compare_model_vs_xor(
    model,
    path_goppa,
    path_random,
    t,
    device,
    key="G",
    n_samples=1000,
    batch_size=256,
):
    X, y_true, source = load_same_order_samples(
        path_goppa=path_goppa,
        path_random=path_random,
        key=key,
        n_samples=n_samples,
    )

    pred_ai = predict_model_numpy(
        model=model,
        X=X,
        device=device,
        batch_size=batch_size,
    )

    pred_xor = predict_xor(
        X=X,
        t=t,
    )

    same = pred_ai == pred_xor
    table = np.zeros((2, 2), dtype=np.int64)
    for a, x in zip(pred_ai, pred_xor):
        table[a, x] += 1

    disagreement_idx = np.where(~same)[0]

    acc_ia = float(np.mean(pred_ai == y_true))
    acc_xor = float(np.mean(pred_xor == y_true))
    agreement = float(np.mean(same))
    print(path_random,acc_ia,acc_xor)
    random_mask = source == "random"
    goppa_mask = source == "goppa"

    p_ai_goppa_on_random = float(np.mean(pred_ai[random_mask] == 1))
    p_ai_random_on_random = float(np.mean(pred_ai[random_mask] == 0))

    p_xor_goppa_on_random = float(np.mean(pred_xor[random_mask] == 1))
    p_xor_random_on_random = float(np.mean(pred_xor[random_mask] == 0))

    p_ai_goppa_on_goppa = float(np.mean(pred_ai[goppa_mask] == 1))
    p_xor_goppa_on_goppa = float(np.mean(pred_xor[goppa_mask] == 1))

    return {
        "X": X,
        "y_true": y_true,
        "source": source,
        "pred_ai": pred_ai,
        "pred_xor": pred_xor,
        "same": same,
        "disagreement_idx": disagreement_idx,

        "table": table,
        "acc_ia": acc_ia,
        "acc_xor": acc_xor,
        "agreement": agreement,
        "agreement_goppa": float(np.mean(same[source == "goppa"])),
        "agreement_random": float(np.mean(same[source == "random"])),

        "p_ai_goppa_on_random": p_ai_goppa_on_random,
        "p_ai_random_on_random": p_ai_random_on_random,
        "p_xor_goppa_on_random": p_xor_goppa_on_random,
        "p_xor_random_on_random": p_xor_random_on_random,
        "p_ai_goppa_on_goppa": p_ai_goppa_on_goppa,
        "p_xor_goppa_on_goppa": p_xor_goppa_on_goppa,
    }

def max_rate(m, t):
    n_max = 2**m
    return 1 - (m * t / n_max)


def n_from_target_rate(m, t, R_target, rounding="floor"):
    """
    Compute code length necessary to attain the target rate.
    """
    x = (m * t) / (1 - R_target)

    if rounding == "floor":
        return int(x)
    elif rounding == "round":
        return round(x)
    else:
        raise ValueError("rounding doit être 'floor', 'ceil' ou 'round'")


def target_rate_pairs(m, t, target_rates, rounding="floor", add_full=True):
    """
    Return the list of code rate we want the distinguishers to be compared on depending on m and t.
    """
    n_max = 2**m
    pairs = []

    for R_target in target_rates:
        n = n_from_target_rate(m, t, R_target, rounding=rounding)

        if n <= m * t:
            continue

        if n > n_max:
            continue

        pairs.append((R_target, n))

    if add_full:
        pairs.append(("full", n_max))

    return pairs

def build_agreement_results(
    model,
    ns,
    m,
    t,
    device,
    representation="AT",
    n_samples=1000,
    batch_size=256,
    key="G",
    random_filename="dataset_10K.h5",
    xorfree_filename="dataset_10K_xorfree.h5",
):
    """
    Build 'results' dictionnary for plot functions.
    

    results[n]["raw"]["table"]     = matrix IA/XOR for random corrected distribution
    results[n]["xorfree"]["table"] = matrix IA/XOR for XOR-free distribution
    """
    results = {}

    for n in ns:


        path_goppa = (
            f"./data/dataset_goppa_{n}_H5/"
            f"{representation}_goppa_nmt_{n}_{m}_{t}/dataset_10K.h5"
        )

        path_random = (
            f"./data/dataset_random_{n}_H5/"
            f"{representation}_random_nmt_{n}_{m}_{t}/{random_filename}"
        )

        path_random_xorfree = (
            f"./data/dataset_random_{n}_H5/"
            f"AT_random_nmt_{n}_{m}_{t}/{xorfree_filename}"
        )

        #compare results between DeepDistinguisher and XOR-distinguisher on corrected random distribution
        res_raw = compare_model_vs_xor(
            model=model,
            path_goppa=path_goppa,
            path_random=path_random,
            t=t,
            device=device,
            key=key,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        #compare results between DeepDistinguisher and XOR-distinguisher on XOR-free distribution
        res_xorfree = compare_model_vs_xor(
            model=model,
            path_goppa=path_goppa,
            path_random=path_random_xorfree,
            t=t,
            device=device,
            key=key,
            n_samples=n_samples,
            batch_size=batch_size,
        )

        results[n] = {
            "raw": {
                "table": res_raw["table"],
                "acc_ia": res_raw["acc_ia"],
                "acc_xor": res_raw["acc_xor"],
                "agreement": res_raw["agreement"],

                "p_ai_goppa_on_random": res_raw["p_ai_goppa_on_random"],
                "p_ai_random_on_random": res_raw["p_ai_random_on_random"],
                "p_xor_goppa_on_random": res_raw["p_xor_goppa_on_random"],
                "p_xor_random_on_random": res_raw["p_xor_random_on_random"],
                "p_ai_goppa_on_goppa": res_raw["p_ai_goppa_on_goppa"],
                "p_xor_goppa_on_goppa": res_raw["p_xor_goppa_on_goppa"],
            },
            "xorfree": {
                "table": res_xorfree["table"],
                "acc_ia": res_xorfree["acc_ia"],
                "acc_xor": res_xorfree["acc_xor"],
                "agreement": res_xorfree["agreement"],

                "p_ai_goppa_on_random": res_xorfree["p_ai_goppa_on_random"],
                "p_ai_random_on_random": res_xorfree["p_ai_random_on_random"],
                "p_xor_goppa_on_random": res_xorfree["p_xor_goppa_on_random"],
                "p_xor_random_on_random": res_xorfree["p_xor_random_on_random"],
                "p_ai_goppa_on_goppa": res_xorfree["p_ai_goppa_on_goppa"],
                "p_xor_goppa_on_goppa": res_xorfree["p_xor_goppa_on_goppa"],
            },
        }

    return results

def plot_agreement_curves(
    results,
    m,
    t,
    output_path="agreement_vs_rate.pdf",
):
    """
    Plot agreement curves between DeepDistinguisher and XOR-distinguisher for corrected random
    distribution and XOR-free distribution.
    """

    items = sorted(
        results.items(),
        key=lambda kv: code_rate(int(kv[0]), m, t)
    )

    ns = [int(n) for n, _ in items]
    rates = [code_rate(n, m, t) for n in ns]

    agreement_raw = [results[n]["raw"]["agreement"] for n in ns]
    agreement_xorfree = [results[n]["xorfree"]["agreement"] for n in ns]

    def setup_axis(ax):
        ax.set_xlabel(r"Code rate", fontsize=30)
        ax.set_ylabel("Agreement", fontsize=30)
        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

        for r, n in zip(rates, ns):
            ax.text(
                r,
                0.02,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=25,
                rotation=0,
            )

    fig1, ax1 = plt.subplots(figsize=(16, 12))

    ax1.plot(
        rates,
        agreement_raw,
        marker="o",
        linewidth=2,
        label="random correct-weight",
        color="#2dbd75"
    )

    setup_axis(ax1)
    ax1.legend(prop={"size": 20, "weight": "bold"})
    fig1.tight_layout()
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(16, 12))

    ax2.plot(
        rates,
        agreement_raw,
        marker="o",
        linewidth=2,
        label="random correct-weight",
        color="#2dbd75"
    )

    ax2.plot(
        rates,
        agreement_xorfree,
        marker="s",
        linewidth=2,
        label="random XOR-free",
        color="#e03e61"
    )

    setup_axis(ax2)
    ax2.legend(prop={"size": 20, "weight": "bold"})
    fig2.tight_layout()

    fig2.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved in : {output_path}")

def plot_goppa_prediction_barplot(
    results,
    m,
    t,
    output_path="goppa_prediction_rate_barplot.pdf",
):
    """
    False positive rate of DeepDistinguisher on corrected random distribution and XOR-free distribution
    """
    items = sorted(
        results.items(),
        key=lambda kv: code_rate(int(kv[0]), m, t)
    )

    ns = [int(n) for n, _ in items]
    rates = [code_rate(n, m, t) for n in ns]

    p_raw = [
        results[n]["raw"]["p_ai_goppa_on_random"]
        for n in ns
    ]

    p_xorfree = [
        results[n]["xorfree"]["p_ai_goppa_on_random"]
        for n in ns
    ]

    x = np.arange(len(ns))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12,8))

    bars_raw = ax.bar(
        x - width / 2,
        p_raw,
        width,
        label="random correct weigth",
        color="#2dbd75",    
        edgecolor="black",
        linewidth=0.5,
    )

    bars_xorfree = ax.bar(
        x + width / 2,
        p_xorfree,
        width,
        label="random XOR-free",
        color="#e03e61",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel(r"Code length (code rate)",fontsize="20",fontweight="bold")
    ax.set_ylabel(r"False Positive Rate",fontsize="20",fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"n = {n} (r={r:.1f})"
        for r, n in zip(rates, ns)
    ],fontsize=16,fontweight="bold")

    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(prop={"size": 20, "weight": "bold"})

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved in : {output_path}")


def main():
    n_max = 64
    m = 6
    t = 4
    representation = "AT"
    checkpoint_kind = "all"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ns = [n for n in range(8*t,65,8)]
    n_samples = 1000
    batch_size = 256

    print(
        "Loading model with "
        f"n_max={n_max}, m={m}, t={t}, "
        f"representation={representation}, checkpoint_kind={checkpoint_kind}, "
        f"device={device}"
    )

    model = load_model(
        n=n_max,
        m=m,
        t=t,
        representation=representation,
        device=device,
        checkpoint_kind=checkpoint_kind,
    )

    print(f"Evaluating n values: {ns}")

    results = build_agreement_results(
        model=model,
        ns=ns,
        m=m,
        t=t,
        device=device,
        representation=representation,
        n_samples=n_samples,
        batch_size=batch_size,
        key="G",
        random_filename="dataset_10K.h5",
        xorfree_filename="dataset_10K_xorfree.h5",
    )

    plot_agreement_curves(
        results,
        m=m,
        t=t,
        output_path="./out/XOR/agreement.png",
    )

    plot_goppa_prediction_barplot(
        results,
        m=m,
        t=t,
        output_path="./out/XOR/false_positive.png",
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()