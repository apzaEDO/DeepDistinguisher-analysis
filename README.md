# DeepDistinguisher Analysis for Goppa Codes

This repository extends the experimental framework of **AI for Code-Based Cryptography** by Malhou, Perret and Lauter.

The original project introduces **DeepDistinguisher**, a Transformer-based neural distinguisher trained to classify public generator matrices of structured codes, such as binary Goppa codes, from random linear codes.

This repository focuses on the analysis of DeepDistinguisher for binary Goppa codes. In particular, it studies whether the model learns structural properties of Goppa codes or whether part of its behavior can be explained by simpler combinatorial rules.

The main additions are:

* a C implementation for faster generation of Goppa and random matrices;
* a modified representation using the systematic block (A) instead of the full generator matrix $$(G=[I_k \mid A])$$;
* attention-map and saliency-map analysis tools;
* an explicit XOR-based distinguisher;
* comparisons between DeepDistinguisher and the XOR-distinguisher;

---

## Relation to AI4Code

This project is based on the public implementation associated with:

> Mohamed Malhou, Ludovic Perret, Kristin Lauter,
> *AI for Code-based Cryptography*, Cryptology ePrint Archive, Paper 2025/440.

The original AI4Code repository provides:

* code generation tools;
* dataset construction scripts;
* the DeepDistinguisher architecture;
* the training and evaluation pipeline.

This repository keeps the general setting of AI4Code but modifies the experimental pipeline in order to analyze the behavior of the model.

The original README is kept in:

```text
README_AI4CODE.md
```

---


## Repository Structure

Only source files and scripts are intended to be versioned.

```text
.
├── generationC/
│   ├── gen_goppa.c                  # C generator for Goppa samples
│   ├── gen_random.c                 # C generator for corrected random samples
│   ├── gen_random_xorfree.c         # C generator for XOR-free random samples
│   ├── Makefile                     # compilation rules
│   ├── make_dataset_goppa_parallel.py      #parallel Python wrapper
│   ├── make_dataset_random_parallel.py     #parallel Python wrapper
│   └── make_dataset_xorfree.py             #parallel Python wrapper
│
├── scripts/
│   ├── gen_dataset.sh               # original AI4Code dataset generation wrapper
│   ├── gen_all.sh                   # generation for all parameters needed to reproduce training performed in the article
│   ├── gen_code.sh                  # code generation for only goppa or random codes
│   ├── collect_data.py              # collection of generated shards into HDF5
│   ├── generate_data.py             # Python/Sage generation script
│   ├── make_dataset_all_goppa_parallel.py
│   ├── make_dataset_all_random_parallel.py
│   ├── train_all.sh                 # launch several training runs
│   ├── 1train.sh                    # training wrapper for one code length
│   ├── ntrain.sh                    # training wrapper for several code lengths    
│   └── viewTrain.sh                 
│
├── src/
│   ├── data/                        # datasets, generators and tokenizers from AI4CODE
│   ├── model/                       # Transformer models from AI4CODE and modified structure for A representation
│   ├── attention/
│   │   └── attention_mapAnalysis.py # attention-map extraction
│   ├── saliency/
│   │   └── saliencyAnalysis.py      # saliency-map analysis
│   ├── xor_distinguisher/
│   │   ├── distinguisher.py         # XOR-distinguisher implementation
│   │   └── compareXOR_DD.py         # comparison with DeepDistinguisher
│   ├── trainer.py
│   ├── metrics.py
│   ├── optim.py
│   ├── logger.py
│   └── utils.py
│
├── Experimental-proba/
│   └── estimate_xor_proba.py        # experimental probability estimates
│
├── train.py                         # main training entry point
├── evaln.py                         # evaluation script 
├── README_AI4CODE.md                # original AI4Code README
└── README.md
```

---

## Requirements

The project requires :

* Linux or macOS;
* Python (\geq 3.10);
* PyTorch;
* SageMath;
* NumPy;
* h5py;
* a C compiler such as `gcc`;
* optionally, an NVIDIA GPU for training.
* PyTorch, installed separately according to the user's hardware;

SageMath is required by the original AI4Code generation pipeline.
The C generation pipeline requires a standard C compiler and `make`.

## PyTorch Installation

PyTorch is not pinned to a specific CUDA version in `requirements.txt`, because the correct PyTorch build depends on the user's hardware, operating system, NVIDIA driver and CUDA version.

By default, a CPU-only installation can be used:

```bash
pip install torch
```

This is sufficient to run the code on CPU, although training DeepDistinguisher will be significantly slower than on GPU.

For GPU support, install PyTorch separately by following the official installation instructions:

```text
https://pytorch.org/get-started/locally/
```

For example, on a machine using CUDA 13.0, the installation command may look like:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

After installing PyTorch, check the installation with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The output should display the installed PyTorch version. If `torch.cuda.is_available()` returns `True`, PyTorch can access a CUDA-compatible GPU.


---

## Installation

Create a Python environment:

```bash
conda create -n ai4code-analysis python=3.10 -y
conda activate ai4code-analysis
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install SageMath if needed:

```bash
conda install -c conda-forge sage
```

or, on Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install sagemath
```

Check that SageMath is available:

```bash
sage -python - <<'PY'
import sage.all
print("SageMath available")
PY
```

Compile the C generators from the project root:

```bash
make -C ./generationC/
```

This should create the executables:

```text
generationC/gen_goppa
generationC/gen_random
generationC/gen_xorfree
```

---

## Dataset Generation

Datasets can be generated either with the original Python/Sage pipeline or with the C-based pipeline.

### C-based Goppa generation

The C generator is located in:

```text
generationC/gen_goppa.c
```

A parallel Python wrapper is provided:

```text
generationC/make_dataset_goppa_parallel.py
```

Example:

```bash
python3 ./generationC/make_dataset_goppa_parallel.py n m t n_samples
```

`n` is the code length, `m` is the extension degree, `t` is the degree of the Goppa polynomial, and `n_samples` is the number of samples to generate.

### Corrected random generation

Corrected random matrices, where the low-weight XOR artifact is removed, are generated using:

```text
generationC/gen_random.c
```

with the wrapper:

```text
generationC/make_dataset_random_parallel.py
```

Example:

```bash
python generationC/make_dataset_random_parallel.py n m t n_samples
```
Where the parameters illustrate that the random generator matrices produced are similar to Goppa generator matrix of those (n,m,t) parameters.


### XOR-free random generation
XOR-free random matrices are generated using:

```text
generationC/gen_random_xorfree.c
```

with the wrapper:

```text
generationC/make_dataset_xorfree_parallel.py
```

Example:

```bash
python generationC/make_dataset_xorfree.py n m t n_samples
```

Where the parameters illustrate that the random generator matrices produced are similar to Goppa generator matrix of those (n,m,t) parameters.


These random datasets are used to test whether DeepDistinguisher still distinguishes Goppa matrices after removing the XOR signal.

---

## Data Layout

Generated datasets follow the naming convention:

```text
<representation>_<code>_nmt_<n>_<m>_<t>
```

For example:

```text
A_goppa_nmt_64_6_2
AT_random_nmt_64_6_3
G_goppa_nmt_64_6_3
```

where:

* `A` denotes the systematic block (A) generated with AI4CODE generation process;
* `AT` * `AT` denotes the systematic block `A` generated with the C pipeline. For random matrices, this pipeline imposes row-weight constraints matching the Goppa-generated matrices;
* `G` denotes the full systematic generator matrix (G=[I_k \mid A]) generated with AI4CODE generation process;
* `goppa` denotes the Goppa distribution;
* `random` denotes the random distribution;
* `nmt_64_6_2` corresponds to (n=64), (m=6), (t=2).

The generated HDF5 datasets are stored under directories such as:

```text
data/dataset_goppa_64_H5/
data/dataset_random_64_H5/
```

These files are not versioned.

---

## Training DeepDistinguisher

The main training entry point is:

```text
train.py
```

Training wrappers are available in:

```text
scripts/
```

For example:

```bash
bash scripts/1train.sh n t representation
```
`n` is the code length, `t` is the degree of the Goppa polynomial, and `representation` is the matrix representation used to train the model.


The model checkpoints are written to:

```text
checkpoint/
```

For example:

```text
checkpoint/debug_pretrain/A_model_Goppa_N64_T2_M6
checkpoint/debug_pretrain/AT_model_NGoppa_N64_T3_M6
```

---

## XOR-Distinguisher

The XOR-distinguisher is implemented in:

```text
src/xor_distinguisher/distinguisher.py
```

Given a matrix

$$
A \in \mathbb{F}_2^{k \times (n-k)}.
$$

the distinguisher computes all pairwise XORs between rows:

$$
A_i + A_j, \qquad i < j.
$$

It predicts `random` if there exists a pair of rows such that

$$
w_H(A_i + A_j) < 2t-1.
$$

Otherwise, it predicts `Goppa`.

Pseudo-code:

```text
Input: matrix A, Goppa degree t

for i < j:
    if HammingWeight(A[i] XOR A[j]) < 2t - 1:
        return Random

return Goppa
```

The comparison between DeepDistinguisher and the XOR-distinguisher is implemented in:

```text
src/xor_distinguisher/compareXOR_DD.py
```

The generated figures are stored in:

```text
out/XOR/
```

For example:

```text
out/XOR/agreement.png
out/XOR/false_positive.png
```

---

## Attention and Saliency Analysis

The repository contains tools to analyze the internal behavior of DeepDistinguisher.

### Attention maps

Attention-map extraction is implemented in:

```text
src/attention/attention_mapAnalysis.py
```

Generated attention maps are stored in:

```text
out/attention_maps/
```

Typical outputs include:

```text
attn_A_goppa_N64_M6_T2_layer0.png
attn_A_random_N64_M6_T2_layer0.png
attn_AT_goppa_N64_M6_T2_layer0.png
attn_AT_random_N64_M6_T2_layer0.png
```

### Saliency maps

Saliency-map analysis is implemented in:

```text
src/saliency/saliencyAnalysis.py
```

Generated saliency summaries are stored in:

```text
out/saliency_maps/
```

For example:

```text
summary_A_N64_M6_random_abs_mean.png
summary_AT_N64_M6_random_abs_mean.png
```

---

## Experimental Probability Estimates

The script

```text
Experimental-proba/estimate_xor_proba.py
```

is used to estimate the probability that a random binary matrix contains at least one pair of rows whose XOR has low Hamming weight.

This supports the analysis of the XOR-distinguisher and helps compare empirical results with the expected behavior of random matrices.

---

## Example Workflow

An example of training DeepDistinguisher consists of the following steps.

### 1. Compile the C generators

```bash
make -C ./generationC/
```

### 2. Generate Goppa and random datasets

```bash
python generationC/make_dataset_goppa_parallel.py 64 6 2 10000
python generationC/make_dataset_random_parallel.py 64 6 2 10000
```

### 3. Train DeepDistinguisher

```bash
bash scripts/1train.sh 64 2 AT
```

An other example of the complete workflow (longer to compute) is :

---

### 1. Compile the C generators

```bash
make -C ./generationC/
```

### 2. Generate Goppa and random datasets

```bash
for n in $(seq 32 8 64); do
    python3 generationC/make_dataset_goppa_parallel.py "$n" 6 4 100000 && \
    python3 generationC/make_dataset_random_parallel.py "$n" 6 4 100000
done
```

### 3. Generate XOR-free random datasets

```bash
python generationC/make_dataset_xorfree.py 64 6 4 10000
```

### 4. Train DeepDistinguisher

```bash
bash scripts/ntrain.sh 64 4 AT
```


### 5. Evaluate the XOR-distinguisher

```bash
python src/xor_distinguisher/distinguisher.py
```

### 6. Compare DeepDistinguisher and XOR-distinguisher

```bash
python src/xor_distinguisher/compareXOR_DD.py
```

### 7. Generate attention and saliency analyses

```bash
python src/attention/attention_mapAnalysis.py
python src/saliency/saliencyAnalysis.py
```

---

## Outputs

The main generated outputs are:

```text
data/        datasets
checkpoint/ trained model checkpoints
out/         figures and analysis results
```

These directories are intentionally excluded from version control.

---

## Citation

If you use this repository, please cite the original AI4Code paper:

```bibtex
@misc{AI4code,
      author = {Mohamed Malhou and Ludovic Perret and Kristin Lauter},
      title = {{AI} for Code-based Cryptography},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/440},
      year = {2025},
      url = {https://eprint.iacr.org/2025/440}
}
```


---

## License

This repository is based on AI4Code. The original code is distributed under a CC-BY-NC license.

Check the license of the original repository before redistributing modified versions or trained models.
