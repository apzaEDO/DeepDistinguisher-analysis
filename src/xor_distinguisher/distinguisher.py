import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import multiprocessing as mp


def _worker_pair_batch(args):
    """
    Worker for the pairwise XOR distinguisher.

    It returns the predictions for one chunk of matrices.
    """
    A_chunk, t = args

    # Prevent BLAS/OpenMP oversubscription inside multiprocessing workers.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    preds = np.empty(len(A_chunk), dtype=np.int64)

    for i, A in enumerate(A_chunk):
        # Prediction convention:
        #   0 = Random
        #   1 = Goppa
        preds[i] = 0 if has_pair_violation(A, t) else 1

    return preds


def _worker_triple_batch(args):
    """
    Worker for the triple XOR distinguisher.

    It returns the predictions for one chunk of matrices.
    """
    A_chunk, t = args

    # Prevent BLAS/OpenMP oversubscription inside multiprocessing workers.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    preds = np.empty(len(A_chunk), dtype=np.int64)

    for i, A in enumerate(A_chunk):
        # Prediction convention:
        #   0 = Random
        #   1 = Goppa
        preds[i] = 0 if has_triple_violation(A, t) else 1

    return preds


def binary_distinguisher_on_batch_parallel(A_batch, t, mode="pair", num_workers=4):
    """
    Apply a binary XOR-based distinguisher in parallel.

    Parameters
    ----------
    A_batch : np.ndarray
        Batch of binary matrices.

    t : int
        Goppa polynomial degree.

    mode : str
        Distinguisher type:
        - "pair": use pairwise XOR violations.
        - "triple": use triple XOR violations.

    num_workers : int
        Number of multiprocessing workers.

    Returns
    -------
    np.ndarray
        Predicted labels with convention:
        - 0 = Random
        - 1 = Goppa
    """
    if len(A_batch) == 0:
        return np.empty(0, dtype=np.int64)

    chunks = [
        chunk
        for chunk in np.array_split(A_batch, num_workers)
        if len(chunk) > 0
    ]

    if mode == "pair":
        worker = _worker_pair_batch
    elif mode == "triple":
        worker = _worker_triple_batch
    else:
        raise ValueError(f"Unknown mode: {mode}")

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=num_workers) as pool:
        preds_list = pool.map(worker, [(chunk, t) for chunk in chunks])

    return np.concatenate(preds_list, axis=0)


def pair_xor_features(A, t, q=0.01):
    """
    Compute three pairwise XOR features for one binary matrix A:

    - pair_xor_min:
        minimum Hamming weight among all pairwise row XORs.

    - pair_violation_count:
        number of pairs (i, j) such that
            w_H(A_i + A_j) < 2t - 1.

    - pair_xor_q01:
        low quantile of the pairwise XOR weight distribution.

    Parameters
    ----------
    A : np.ndarray
        Binary matrix of shape (k, r).

    t : int
        Goppa polynomial degree.

    q : float
        Quantile to compute.

    Returns
    -------
    dict
        Dictionary containing the three features.
    """
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A must be binary")

    threshold = 2 * t - 1
    k, r = A.shape

    A = A.astype(np.int16, copy=False)
    row_w = A.sum(axis=1)

    pair_weights = []

    for i in range(k - 1):
        ai = A[i]
        wi = row_w[i]

        inter = A[i + 1:] @ ai
        xor_w = wi + row_w[i + 1:] - 2 * inter
        pair_weights.append(xor_w)

    pair_weights = np.concatenate(pair_weights, axis=0)

    pair_xor_min = int(pair_weights.min())
    pair_violation_count = int(np.count_nonzero(pair_weights < threshold))
    pair_xor_q01 = float(np.quantile(pair_weights, q))

    return {
        "pair_xor_min": pair_xor_min,
        "pair_violation_count": pair_violation_count,
        "pair_xor_q01": pair_xor_q01,
    }


def extract_pair_feature_matrix(A_batch, t, q=0.01):
    """
    Compute pairwise XOR features for a batch of matrices.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (N, 3).

    feature_names : list[str]
        Names of the computed features.
    """
    X = np.empty((len(A_batch), 3), dtype=np.float64)

    for n, A in enumerate(A_batch):
        feats = pair_xor_features(A, t=t, q=q)

        X[n, 0] = feats["pair_xor_min"]
        X[n, 1] = feats["pair_violation_count"]
        X[n, 2] = feats["pair_xor_q01"]

    feature_names = [
        "pair_xor_min",
        "pair_violation_count",
        "pair_xor_q01",
    ]

    return X, feature_names


def has_pair_violation(A, t):
    """
    Return True if there exists a pair of rows (i, j) such that

        w_H(A_i + A_j) < 2t - 1.

    Here + denotes XOR over F_2.
    """
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A must be binary")

    threshold = 2 * t - 1
    k, r = A.shape

    A = A.astype(np.int16, copy=False)
    row_w = A.sum(axis=1)

    for i in range(k - 1):
        ai = A[i]
        wi = row_w[i]

        inter = A[i + 1:] @ ai
        xor_w = wi + row_w[i + 1:] - 2 * inter

        if np.any(xor_w < threshold):
            return True

    return False


def pair_violation_count_early(A, t, stop_at_first=False):
    """
    Count pairwise XOR violations.

    If stop_at_first=True, return 1 as soon as one violation is found.
    """
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A must be binary")

    threshold = 2 * t - 1
    k, r = A.shape

    A = A.astype(np.int16, copy=False)
    row_w = A.sum(axis=1)

    count = 0

    for i in range(k - 1):
        ai = A[i]
        wi = row_w[i]

        inter = A[i + 1:] @ ai
        xor_w = wi + row_w[i + 1:] - 2 * inter

        c = np.count_nonzero(xor_w < threshold)

        if c:
            if stop_at_first:
                return 1

            count += c

    return count


def binary_distinguisher_on_batch(A_batch, t):
    """
    Apply the pairwise XOR distinguisher to a batch of matrices.

    Prediction convention:
    - 0 = Random
    - 1 = Goppa

    A matrix is predicted as Random if it contains at least one pairwise XOR
    violation. Otherwise, it is predicted as Goppa.
    """
    preds = np.empty(len(A_batch), dtype=np.int64)

    for n in range(len(A_batch)):
        preds[n] = 0 if has_pair_violation(A_batch[n], t) else 1

    return preds


def evaluate_binary_distinguisher(path_goppa, path_random, t, key="G", n_samples=1000):
    """
    Evaluate the pairwise XOR distinguisher on balanced Goppa and random datasets.
    """
    with h5py.File(path_goppa, "r") as f:
        Ag = f[key][:n_samples]

    with h5py.File(path_random, "r") as f:
        Ar = f[key][:n_samples]

    pred_g = binary_distinguisher_on_batch(Ag, t=t)
    pred_r = binary_distinguisher_on_batch(Ar, t=t)

    y_true = np.concatenate([
        np.ones(len(Ag), dtype=np.int64),
        np.zeros(len(Ar), dtype=np.int64),
    ])

    y_pred = np.concatenate([pred_g, pred_r])

    acc = (y_true == y_pred).mean()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    print("Accuracy:", acc)
    print("Confusion matrix:")
    print(np.array([[tn, fp], [fn, tp]]))
    print()
    print("Random predicted Goppa:", fp, "/", len(Ar))
    print("Goppa predicted Random:", fn, "/", len(Ag))

    return Ag, Ar, y_true, y_pred


def evaluate_binary_distinguisher_parallel(
    path_goppa,
    path_random,
    t,
    key="G",
    n_samples=1000,
    num_workers=4,
):
    """
    Evaluate the pairwise XOR distinguisher in parallel on balanced datasets.
    """
    with h5py.File(path_goppa, "r") as f:
        Ag = f[key][:n_samples]

    with h5py.File(path_random, "r") as f:
        Ar = f[key][:n_samples]

    pred_g = binary_distinguisher_on_batch_parallel(
        Ag,
        t=t,
        mode="pair",
        num_workers=num_workers,
    )

    pred_r = binary_distinguisher_on_batch_parallel(
        Ar,
        t=t,
        mode="pair",
        num_workers=num_workers,
    )

    y_true = np.concatenate([
        np.ones(len(Ag), dtype=np.int64),
        np.zeros(len(Ar), dtype=np.int64),
    ])

    y_pred = np.concatenate([pred_g, pred_r])

    acc = (y_true == y_pred).mean()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    print("Accuracy:", acc)

    return Ag, Ar, y_true, y_pred


def plot_feature_distribution(
    Xg,
    Xr,
    feature_names,
    feature_name,
    title=None,
    density=True,
):
    """
    Plot the empirical distribution of one feature for Goppa and random samples.
    """
    idx = feature_names.index(feature_name)

    vg = Xg[:, idx]
    vr = Xr[:, idx]

    vmin = int(min(vg.min(), vr.min()))
    vmax = int(max(vg.max(), vr.max()))

    bins = np.arange(vmin - 0.5, vmax + 1.5, 1)

    plt.figure(figsize=(9, 5))

    plt.hist(
        vg,
        bins=bins,
        alpha=0.6,
        label="Goppa",
        density=density,
    )

    plt.hist(
        vr,
        bins=bins,
        alpha=0.6,
        label="Random",
        density=density,
    )

    plt.xlabel(feature_name)
    plt.ylabel("Density" if density else "Count")
    plt.title(title if title is not None else f"Distribution of {feature_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(
        f"{feature_name} — Goppa: "
        f"min={vg.min()}, mean={vg.mean():.4f}, "
        f"median={np.median(vg)}, max={vg.max()}"
    )

    print(
        f"{feature_name} — Random: "
        f"min={vr.min()}, mean={vr.mean():.4f}, "
        f"median={np.median(vr)}, max={vr.max()}"
    )


def summarize_feature_distribution(X, feature_names, label):
    """
    Print summary statistics for each feature.
    """
    print(f"\n=== {label} ===")

    for j, name in enumerate(feature_names):
        col = X[:, j]

        print(
            f"{name:22s} "
            f"min={col.min():.4f} "
            f"mean={col.mean():.4f} "
            f"median={np.median(col):.4f} "
            f"max={col.max():.4f}"
        )


def has_triple_violation(A, t):
    """
    Return True if there exists a triplet of rows (i, j, l) such that

        w_H(A_i + A_j + A_l) < 2t - 2.

    Here + denotes XOR over F_2.
    """
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A must be binary")

    threshold = 2 * t - 2
    k, r = A.shape

    A = A.astype(np.uint8, copy=False)

    for i in range(k - 2):
        ai = A[i]

        for j in range(i + 1, k - 1):
            xij = np.bitwise_xor(ai, A[j])

            for l in range(j + 1, k):
                w = np.bitwise_xor(xij, A[l]).sum()

                if w < threshold:
                    return True

    return False


def triple_violation_count_early(A, t, stop_at_first=False):
    """
    Count triple XOR violations.

    If stop_at_first=True, return 1 as soon as one violation is found.
    """
    A = np.asarray(A)

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")

    if not np.all((A == 0) | (A == 1)):
        raise ValueError("A must be binary")

    threshold = 2 * t - 2
    k, r = A.shape

    A = A.astype(np.uint8, copy=False)

    count = 0

    for i in range(k - 2):
        ai = A[i]

        for j in range(i + 1, k - 1):
            xij = np.bitwise_xor(ai, A[j])

            for l in range(j + 1, k):
                w = np.bitwise_xor(xij, A[l]).sum()

                if w < threshold:
                    if stop_at_first:
                        return 1

                    count += 1

    return count


def binary_distinguisher_on_batch_triple(A_batch, t):
    """
    Apply the triple XOR distinguisher to a batch of matrices.

    Prediction convention:
    - 0 = Random
    - 1 = Goppa

    A matrix is predicted as Random if it contains at least one triple XOR
    violation. Otherwise, it is predicted as Goppa.
    """
    preds = np.empty(len(A_batch), dtype=np.int64)

    for n in range(len(A_batch)):
        preds[n] = 0 if has_triple_violation(A_batch[n], t) else 1

    return preds


def evaluate_binary_distinguisher_triple_parallel(
    path_goppa,
    path_random,
    t,
    key="G",
    n_samples=1000,
    num_workers=4,
):
    """
    Evaluate the triple XOR distinguisher in parallel on balanced datasets.
    """
    with h5py.File(path_goppa, "r") as f:
        Ag = f[key][:n_samples]

    with h5py.File(path_random, "r") as f:
        Ar = f[key][:n_samples]

    pred_g = binary_distinguisher_on_batch_parallel(
        Ag,
        t=t,
        mode="triple",
        num_workers=num_workers,
    )

    pred_r = binary_distinguisher_on_batch_parallel(
        Ar,
        t=t,
        mode="triple",
        num_workers=num_workers,
    )

    y_true = np.concatenate([
        np.ones(len(Ag), dtype=np.int64),
        np.zeros(len(Ar), dtype=np.int64),
    ])

    y_pred = np.concatenate([pred_g, pred_r])

    acc = (y_true == y_pred).mean()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    print("Accuracy:", acc)

    return Ag, Ar, y_true, y_pred


# def compare_pair_vs_triple_parallel(
#     path_goppa,
#     path_random,
#     t,
#     key="G",
#     n_samples=1000,
#     num_workers=4,
# ):
#     """
#     Compare the pairwise and triple XOR distinguishers on the same datasets.
#     """
#     print("=== Pairwise XOR distinguisher ===")

#     _, _, y_true2, y_pred2 = evaluate_binary_distinguisher_parallel(
#         path_goppa,
#         path_random,
#         t=t,
#         key=key,
#         n_samples=n_samples,
#         num_workers=num_workers,
#     )

#     acc2 = (y_true2 == y_pred2).mean()

#     print("\n=== Triple XOR distinguisher ===")

#     _, _, y_true3, y_pred3 = evaluate_binary_distinguisher_triple_parallel(
#         path_goppa,
#         path_random,
#         t=t,
#         key=key,
#         n_samples=n_samples,
#         num_workers=num_workers,
#     )

#     acc3 = (y_true3 == y_pred3).mean()

#     print("\n=== Final comparison ===")
#     print(f"Accuracy s=2: {acc2:.6f}")
#     print(f"Accuracy s=3: {acc3:.6f}")


if __name__ == "__main__":
    n = 64
    m = 6

    for t in range(2, 9):
        path_goppa = (
            f"./data/dataset_goppa_{n}_H5/"
            f"AT_goppa_nmt_{n}_{m}_{t}/"
            f"dataset_10K.h5"
        )

        path_random = (
            f"./data/dataset_random_{n}_H5/"
            f"AT_random_nmt_{n}_{m}_{t}/"
            f"dataset_10K.h5"
        )

        _, _, y_true2, y_pred2 = evaluate_binary_distinguisher_triple_parallel(
            path_goppa,
            path_random,
            t=t,
            key="G",
            n_samples=5000,
            num_workers=10,
        )

        # To evaluate the triple XOR distinguisher instead, uncomment this block.
        #
        # _, _, y_true2, y_pred2 = evaluate_binary_distinguisher_triple_parallel(
        #     path_goppa,
        #     path_random,
        #     t=t,
        #     key="G",
        #     n_samples=5000,
        #     num_workers=10,
        # )