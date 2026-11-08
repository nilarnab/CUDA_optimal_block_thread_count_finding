# Modular CUDA launch-configuration selector

This Python package implements the supervised model pipeline discussed in
*Autotuning CUDA: Applying NLP Techniques to LS-CAT* while keeping the modules
generic enough for other kernel datasets and future DPO/RL trainers.

## Implemented pipeline

```text
CUDA source -> LLVM IR -> tokens -> frozen FastText vectors -> LSTM
                                                        |
problem dimensions -> context encoder ------------------+
                                                        |
                                      configuration scorer
                                                        |
                                one score per catalog entry
```

`ConfigurationCatalog` is configurable. Its `lscat()` factory contains the 16
one-dimensional thread-count actions from 64 through 1024 used by the NLP
baseline. `lscat_all_shapes()` retains the 20-shape raw-dataset catalog for
separate experiments.

These are CUDA **threads-per-block actions**, not grid/block counts. The pilot
adapter maps native `(N,1)` and `(1,N)` measurements to the same abstract
action `N`, while preserving the physical per-kernel launch mapping in
`launch_mappings.json`. The application still derives a correctness-preserving
grid and restores the appropriate orientation.

The model is trained against a configurable soft performance vector:

```text
target[i] = (fastest runtime / runtime[i]) ** exponent
```

The default exponent is 12, inspired by the paper's best reported exponent-12
target transformation. The paper calls its exact transformation a modified
Softmax but does not provide enough detail to claim this ratio transform is an
exact reproduction. BCE and MSE training losses are available. Inference
returns the catalog entry with the highest score.

## Install

Use Python 3.9-3.13 with PyTorch support:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,fasttext]'
```

To enable optional Weights & Biases tracking, include the `tracking` extra:

```bash
python -m pip install -e '.[dev,fasttext,tracking]'
```

The current local macOS Python is 3.14, for which PyTorch wheels may not yet be
available. A CUDA machine should use the PyTorch package appropriate for its
installed CUDA toolkit.

## Runtime table format

The default loader expects:

```csv
kernel_id,problem_size,block_x,block_y,block_z,runtime_us
k1,1024,64,1,1,18.42
k1,1024,128,1,1,15.17
```

Repeated rows are allowed and are reduced to their median. Column mappings are
configurable in `RuntimeTable.from_csv()`.

## Run tests

The data, tokenizer, catalog, and split tests use only the Python standard
library:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Prepare the collected pilot

Convert a complete pilot run into the common 16-action runtime table and copy
the six kernels under stable semantic IDs:

```bash
PYTHONPATH=src python scripts/prepare_pilot_dataset.py \
  --run-dir data/full-20261103T225641Z-cr8Us0 \
  --output-dir data/processed \
  --verbose
```

For the current collection this produces 672 runtime rows:

```text
6 kernels × 7 problem sizes × 16 actions = 672
```

It excludes the four square shapes recorded only for `matrix_add`. The raw run
directory is never modified.

Next prepare LLVM IR, the vocabulary, token IDs, and the FastText corpus on a
machine with Clang CUDA support:

```bash
PYTHONPATH=src python scripts/prepare_frontend.py \
  --cuda-dir data/processed/cuda \
  --output-dir data/processed \
  --gpu-arch sm_75 \
  --verbose
```

Train the frozen 40-dimensional skip-gram frontend and export the embedding
matrix:

```bash
PYTHONPATH=src python scripts/train_fasttext.py \
  --corpus data/processed/fasttext_corpus.txt \
  --vocabulary data/processed/vocabulary.json \
  --output-model data/processed/fasttext.bin \
  --output-matrix data/processed/fasttext_embeddings.pt \
  --verbose
```

Then train the selector:

```bash
PYTHONPATH=src python scripts/train_supervised.py \
  --runtimes data/processed/runtimes.csv \
  --token-ids data/processed/token_ids.json \
  --embedding-matrix data/processed/fasttext_embeddings.pt \
  --checkpoint checkpoints/supervised.pt \
  --validation-kernels 2 \
  --verbose
```

`--verbose` is optional on every pipeline command. It prints input and output
artifacts, per-kernel frontend progress, model dimensions, split membership,
and metrics after every training epoch.

With six kernels, the default two-way, kernel-disjoint split contains four
training kernels (28 samples) and two unseen validation kernels (14 samples).
Every problem size belonging to a kernel stays in the same partition. The
checkpoint records the exact kernel IDs used by each partition.

## Weights & Biases tracking

Pass a project name to enable W&B for a training run:

```bash
PYTHONPATH=src python scripts/train_supervised.py \
  --runtimes data/processed/runtimes.csv \
  --token-ids data/processed/token_ids.json \
  --embedding-matrix data/processed/fasttext_embeddings.pt \
  --checkpoint checkpoints/supervised.pt \
  --validation-kernels 2 \
  --wandb-project cuda-block-selector \
  --wandb-run-name lscat-baseline \
  --verbose
```

The supplied `--wandb-run-name` is used as a prefix. The script appends the
epochs, batch size, loss, target exponent, seed, device, validation-kernel
count, and latency-benchmark settings. For example:

```text
lscat-baseline-ep20-bs256-loss-bce-exp12-seed0-dev-cpu-valk2-lw3-lr5
```

Input and output paths, split membership, action values, model sizes, and all
other settings remain available in the W&B configuration panel instead of
making the displayed run name excessively long.

The run records the model and split configuration plus training loss,
validation loss, performance score, mean regret, and exact-optimal rate for
every epoch. After training it also benchmarks batch-size-one selector
inference and reports mean, median, and p95 model latency plus throughput. The
timing includes the complete neural model and `argmax`, but excludes LLVM
compilation, tokenization, tensor transfer, and CUDA kernel execution. Use
`--latency-warmup` and `--latency-repetitions` to control the benchmark.

Use `--wandb-entity` for a team account. To collect data without
network access, add `--wandb-mode offline`; it can later be uploaded with
`wandb sync <run-directory>`. If `--wandb-project` is omitted, W&B is neither
imported nor initialized.

The LS-CAT files are intentionally not hard-coded into the model. This keeps
the implementation usable with our own GPU measurements and with future
representation modules. Split kernels before training FastText when strict
evaluation requires the embedding vocabulary to see training kernels only.

