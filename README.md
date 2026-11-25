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

The project uses `uv` with Python 3.11 and a committed lockfile. Install `uv`,
then create the environment with every optional feature:

```bash
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv sync --python 3.11 --all-extras --locked
```

`uv` downloads Python 3.11 when it is not already available and creates the
project `.venv`. The explicit `--python 3.11` also replaces an incompatible
pre-existing environment. For an installation without FastText or W&B, use
`uv sync --python 3.11 --locked`.
To verify that the lockfile is current without changing it:

```bash
uv lock --check
```

On a CUDA machine, verify the resolved PyTorch build before training:

```bash
uv run --locked python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

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
uv run --locked python -m unittest discover -s tests -v
```

## Prepare the collected pilot

Convert a complete pilot run into the common 16-action runtime table and copy
the six kernels under stable semantic IDs:

```bash
uv run --locked python -c 'from data.pilot_adapter import prepare_pilot_dataset; print(prepare_pilot_dataset("data/full-20261103T225641Z-cr8Us0", "data/processed"))'
```

For the current collection this produces 672 runtime rows:

```text
6 kernels × 7 problem sizes × 16 actions = 672
```

It excludes the four square shapes recorded only for `matrix_add`. The raw run
directory is never modified.

## Import the full public LS-CAT dataset

Clone the official NTNU repository under the ignored raw-data directory and
convert its archived T4 measurements to the same 16-action schema:

```bash
git clone --depth 1 https://github.com/NTNU-HPC-Lab/LS-CAT.git \
  data/raw/ls-cat-official

PYTHONPATH=src uv run --locked python scripts/import_lscat.py \
  --repository data/raw/ls-cat-official \
  --output-dir data/processed/lscat_t4_1d
```

The importer keeps only `(1,N)` block shapes for `N=64,...,1024`, averages
duplicate archived T4 measurements, rejects failure placeholders, retains only
contexts containing all 16 actions, and groups kernels by original repository
for leakage-resistant splitting. The generated `sources.json` maps every
runtime kernel ID to its official CUDA source.

On a CUDA/Clang machine, process only those sources rather than the repository's
generated benchmark harnesses:

```bash
uv run --locked python scripts/prepare_frontend.py \
  --source-manifest data/processed/lscat_t4_1d/sources.json \
  --source-root data/raw/ls-cat-official \
  --output-dir data/processed/lscat_t4_1d \
  --cuda-path "$CONDA_PREFIX" \
  --gpu-arch sm_75 \
  --keep-going \
  --verbose
```

This frontend step cannot run on macOS because it requires a CUDA toolkit.
After producing `token_ids.json` and `fasttext_corpus.txt`, use the filtering
step in [readme.txt](readme.txt) to keep the runtimes and groups for kernels that
compiled. Train FastText using paths under `data/processed/lscat_t4_1d`. Training
uses `--problem-columns m n` and the filtered split groups.

Next prepare LLVM IR, the vocabulary, token IDs, and the FastText corpus on a
machine with Clang CUDA support:

```bash
uv run --locked python scripts/prepare_frontend.py \
  --cuda-dir data/processed/cuda \
  --output-dir data/processed \
  --gpu-arch sm_75 \
  --verbose
```

Train the frozen 40-dimensional skip-gram frontend and export the embedding
matrix:

```bash
uv run --locked python scripts/train_fasttext.py \
  --corpus data/processed/fasttext_corpus.txt \
  --vocabulary data/processed/vocabulary.json \
  --output-model data/processed/fasttext.bin \
  --output-matrix data/processed/fasttext_embeddings.pt \
  --verbose
```

Then train the selector:

```bash
uv run --locked python scripts/train_sequence.py --architecture baseline \
  --runtimes data/processed/runtimes.csv \
  --token-ids data/processed/token_ids.json \
  --embedding-matrix data/processed/fasttext_embeddings.pt \
  --checkpoint checkpoints/supervised.pt \
  --validation-kernels 2 \
  --verbose
```

`--verbose` prints extra progress in frontend preparation, FastText training,
and sequence training.

With six kernels, the default two-way, kernel-disjoint split contains four
training kernels (28 samples) and two unseen validation kernels (14 samples).
Every problem size belonging to a kernel stays in the same partition. The
checkpoint records the exact kernel IDs used by each partition.

## Weights & Biases tracking

Pass a project name to enable W&B for a training run:

```bash
uv run --locked python scripts/train_sequence.py --architecture baseline \
  --runtimes data/processed/runtimes.csv \
  --token-ids data/processed/token_ids.json \
  --embedding-matrix data/processed/fasttext_embeddings.pt \
  --checkpoint checkpoints/supervised.pt \
  --validation-kernels 2 \
  --wandb-project cuda-block-selector \
  --wandb-run-name lscat-baseline \
  --verbose
```

`--wandb-run-name` sets the displayed run name.

Input and output paths, split membership, action values, model sizes, and all
other settings remain available in the W&B configuration panel instead of
making the displayed run name excessively long.

The run records the model and split configuration plus training loss,
validation loss, performance score, mean regret, and exact-optimal rate for
every epoch. After training it caches each kernel representation and benchmarks
the batch-size-one runtime predictor. It reports mean, median, and p95 latency
plus throughput. Timing includes problem-size tensor creation/transfer, the
context encoder, configuration scorer, `argmax`, and returning the action to
the host. It excludes the offline LLVM, tokenization, FastText and LSTM stages,
as well as CUDA kernel execution. Use `--latency-warmup` and
`--latency-repetitions` to control the benchmark.

Use `--wandb-entity` for a team account. To collect data without
network access, add `--wandb-mode offline`; it can later be uploaded with
`wandb sync <run-directory>`. If `--wandb-project` is omitted, W&B is neither
imported nor initialized.

The LS-CAT files are intentionally not hard-coded into the model. This keeps
the implementation usable with our own GPU measurements and with future
representation modules. Split kernels before training FastText when strict
evaluation requires the embedding vocabulary to see training kernels only.

