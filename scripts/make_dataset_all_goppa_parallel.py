import struct
import subprocess
import numpy as np
import h5py
from sage.all import GF, PolynomialRing, set_random_seed
from time import time
import os
import multiprocessing as mp
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="longueur du code")
    parser.add_argument("m", type=int, help="degré d'extension")
    parser.add_argument("t", type=int, help="degré de Goppa")
    parser.add_argument("total_samples", type=int, help="nb samples")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=10)
    return parser.parse_args()



def poly_lowbits_to_int(mod, m: int) -> int:
    coeffs = mod.list()
    if len(coeffs) < m + 1:
        coeffs += [0] * (m + 1 - len(coeffs))

    val = 0
    for i in range(m):
        ci = int(coeffs[i])
        if ci not in (0, 1):
            raise ValueError("Le polynôme du corps doit être sur F2.")
        val |= (ci << i)
    return val


def make_field(m: int):
    Ktmp = GF(2**m, modulus="primitive", names=("a",))
    mod = Ktmp.modulus()
    #print(mod)
    K = GF(2**m, modulus=mod, names=("a",))
    a = K.gen()
    alpha_m = poly_lowbits_to_int(mod, m)
    return K, a, mod, alpha_m


def sample_irreducible_goppa_poly(K, t: int):
    R = PolynomialRing(K, "x")
    while True:
        g = R.random_element(degree=t, monic=True)
        if g.degree() == t and g.is_irreducible():
            #print(g)
            return g


def coeffs_for_c(g):
    return [int(c.to_integer()) for c in g.list()]


def build_payload(K, t: int, batch_size: int) -> bytes:
    lines = []
    for _ in range(batch_size):
        g = sample_irreducible_goppa_poly(K, t)
        coeffs = coeffs_for_c(g)

        lines.append(" ".join(map(str, coeffs[:-1])))
    payload = "\n".join(lines) + "\n"
    return payload.encode()


def run_c_batch(binary_path: str, payload: bytes, n: int, m: int, t: int, alpha_m: int, c_seed: int):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["ASAN_OPTIONS"] = "abort_on_error=1:halt_on_error=1"

    res = subprocess.run(
        [binary_path, str(m), str(t), str(n), str(alpha_m), str(c_seed)],
        input=payload,
        capture_output=True,
        check=False,
        env=env,
    )

    if res.returncode != 0:
        print("=== STDOUT ===")
        print(res.stdout.decode(errors="replace"))
        print("=== STDERR ===")
        print(res.stderr.decode(errors="replace"))
        raise RuntimeError(f"{binary_path} failed with return code {res.returncode}")

    return res.stdout, res.stderr

def parse_matrices_from_stdout(data: bytes):
    offset = 0
    matrices = []

    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("Flux tronqué: header incomplet")

        k, nA = struct.unpack_from("<II", data, offset)
        offset += 8

        size = k * nA
        if offset + size > len(data):
            raise ValueError("Flux tronqué: matrice incomplète")

        A = np.frombuffer(data, dtype=np.uint8, count=size, offset=offset)
        A = A.reshape((k, nA)).copy()
        offset += size

        matrices.append(A)

    return matrices


def append_to_h5(h5_path: str, matrices, expected_shape=None, attrs=None):
    if not matrices:
        return

    first_shape = matrices[0].shape

    for A in matrices:
        if A.shape != first_shape:
            raise ValueError(f"Shapes différentes dans le batch: {A.shape} vs {first_shape}")

    if expected_shape is not None and first_shape != expected_shape:
        raise ValueError(f"Shape inattendue {first_shape}, attendu {expected_shape}")

    batch = np.stack(matrices, axis=0).astype(np.uint8)

    with h5py.File(h5_path, "a") as f:
        if "G" not in f:
            maxshape = (None, batch.shape[1], batch.shape[2])
            f.create_dataset(
                "G",
                data=batch,
                maxshape=maxshape,
                chunks=(min(256, batch.shape[0]), batch.shape[1], batch.shape[2]),
                dtype="uint8",
            )
            if attrs:
                for key, value in attrs.items():
                    f.attrs[key] = value
        else:
            dset = f["G"]
            if dset.shape[1:] != batch.shape[1:]:
                raise ValueError(
                    f"Shape H5 existante {dset.shape[1:]} incompatible avec {batch.shape[1:]}"
                )

            old_n = dset.shape[0]
            new_n = old_n + batch.shape[0]
            dset.resize((new_n, dset.shape[1], dset.shape[2]))
            dset[old_n:new_n] = batch


def format_samples(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}B"
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)

def make_c_seed(base_seed: int, worker_id: int, batch_idx: int, n: int, m: int, t: int) -> int:
    x = (
        (base_seed & 0xFFFFFFFFFFFFFFFF)
        ^ ((worker_id + 1) * 0x9E3779B97F4A7C15)
        ^ ((batch_idx + 1) * 0xBF58476D1CE4E5B9)
        ^ ((n + 1) * 0x94D049BB133111EB)
        ^ ((m + 1) << 17)
        ^ ((t + 1) << 33)
    ) & 0xFFFFFFFFFFFFFFFF

    # mélange type splitmix64
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 31)

    # srand() prend un unsigned int -> 32 bits
    return x & 0xFFFFFFFF

def worker(worker_id, samples_to_generate, n, m, t, batch_size, binary_path, out_dir):
    seed = 123456 + worker_id
    set_random_seed(seed)
    np.random.seed(seed)

    K, a, mod, alpha_m = make_field(m)

    expected_k = n - m * t
    expected_nA = n - expected_k
    expected_shape = (expected_k, expected_nA)

    part_path = os.path.join(out_dir, f"dataset_part_{worker_id:02d}.h5")
    if os.path.exists(part_path):
        os.remove(part_path)

    written = 0
    batch_idx=0
    while written < samples_to_generate:
        current_batch = min(batch_size, samples_to_generate - written)

        payload = build_payload(K, t, current_batch)
        c_seed = make_c_seed(
        base_seed=123456789,
        worker_id=worker_id,
        batch_idx=batch_idx,
        n=n,
        m=m,
        t=t,
    )
        stdout_data, stderr_data = run_c_batch(binary_path, payload, n, m, t, alpha_m,c_seed)

        if stderr_data:
            print(f"[worker {worker_id}] stderr:\n{stderr_data.decode(errors='replace')}")

        matrices = parse_matrices_from_stdout(stdout_data)
        matrices = [A for A in matrices if A.shape == expected_shape]

        append_to_h5(
            part_path,
            matrices,
            expected_shape=expected_shape,
            attrs={
                "n": n,
                "m": m,
                "t": t,
                "field_modulus": str(mod),
                "alpha_m": alpha_m,
                "worker_id": worker_id,
            },
        )

        written += len(matrices)
        batch_idx+=1
        print(f"[worker {worker_id}] {written}/{samples_to_generate}")

    return part_path


def merge_h5_files(part_files, final_h5_path, delete_parts=True):
    if os.path.exists(final_h5_path):
        os.remove(final_h5_path)

    for i, part in enumerate(part_files):
        with h5py.File(part, "r") as src:
            data = src["G"][:]
            attrs = dict(src.attrs)

        append_to_h5(
            final_h5_path,
            list(data),
            expected_shape=data.shape[1:],
            attrs=attrs if i == 0 else None,
        )

    print(f"Fusion terminée dans {final_h5_path}")

    if delete_parts:
        for part in part_files:
            if os.path.exists(part):
                os.remove(part)
        print("Fichiers temporaires supprimés.")


def split_counts(total, num_workers):
    q, r = divmod(total, num_workers)
    return [q + (1 if i < r else 0) for i in range(num_workers)]


def main():
    args = get_args()

    n = args.n
    m = args.m
    t = args.t
    total_samples = args.total_samples
    batch_size = 1000         
    num_workers = 10
    binary_path = "./generationC/gen_goppa"

    out_dir = f"../ai4code/data/dataset_goppa_{n}_H5/AT_goppa_nmt_{n}_{m}_{t}"
    os.makedirs(out_dir, exist_ok=True)

    final_h5_path = os.path.join(out_dir, f"dataset_{format_samples(total_samples)}.h5")

    counts = split_counts(total_samples, num_workers)

    args = [
        (wid, counts[wid], n, m, t, batch_size, binary_path, out_dir)
        for wid in range(num_workers)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        part_files = pool.starmap(worker, args)

    merge_h5_files(part_files, final_h5_path, delete_parts=True)

    print("Terminé.")


if __name__ == "__main__":
    start = time()
    main()
    print(time() - start)