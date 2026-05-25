# AGENTS.md

This is a Hydra + PyTorch Lightning research repository for anomaly detection at
L1 trigger. Prefer small, config-driven changes and keep experiments reproducible.

## Environment

- Use `uv`, not Poetry or `pip install -r requirements.txt`.
- Python is pinned by `.python-version` to `3.10`.
- Install the development environment with `uv sync --group dev`.
- Install optional quantisation dependencies with `uv sync --group dev --extra quant`.
- Regenerate local paths with `bash scripts/setup.sh`; `.env` is intentionally ignored.

Important `.env` variables:

- `PROJECT_ROOT`: repository root.
- `DATA_DIR`: local data root.
- `RAW_DATA_DIR`: raw L1 input data root.
- `LOG_DIR`, `OUTPUT_DIR`, `CHECKPOINT_DIR`: runtime artifacts.
- `WANDB_MODE=offline`: default safe WandB behavior.

## Common Commands

- `uv run python tests/train.py experiment=demo/cifar10_ae`
- `uv run python tests/train.py experiment=demo/cchamber_ae`
- `KERAS_BACKEND=torch uv run python tests/train.py experiment=demo/l1_vae algorithm=qvae`
- `make train-demo`
- `uv run pytest -k "not slow"`
- `uv run pre-commit run -a`
- `uv lock`

## Repository Map

- `src/train.py`: main Hydra entry point and training/evaluation orchestration.
- `configs/train.yaml`: base composition. Defaults to CIFAR-10 + `image_ae`.
- `configs/data`: datamodule configs for L1 AD, CIFAR-10, RobustAD, and Causal Chamber.
- `configs/algorithm`: model configs. Use `algorithm`, not the old `model` key.
- `configs/experiment`: paper experiment overrides.
- `tests/configs/experiment/demo`: smoke-test/demo experiment overrides.
- `src/data`: Lightning datamodules and L1 preprocessing components.
- `src/algorithms`: Lightning modules, model components, losses, optimizers, schedulers.
- `src/evaluation`: post-training evaluator and evaluation callbacks.
- `src/callbacks`: training-time callbacks.

## Development Rules

- Keep large artifacts out of git: `data/`, `logs/`, `outputs/`, `results/`,
  `checkpoints/`, and `.env` are ignored.
- Use the CIFAR-10 demo for smoke tests; it downloads automatically under
  `data/cifar10`.
- Causal Chamber demos download `lt_interventions_standard_v1` automatically under
  `data/causal_chamber` and should be run through `tests/train.py`.
- L1 experiments require external raw data. Point `RAW_DATA_DIR` or
  `paths.raw_data_dir` at the unpacked files before running physics configs.
- For quick CLI training/debug runs, use `logger=none`, `trainer=cpu`, small
  `+trainer.limit_*_batches` values, and quote Hydra deletions in zsh, for example
  `'~evaluation' '~evaluation/callbacks' '~callbacks'`.
- If enabling evaluation, configure both `evaluation` and `evaluation/callbacks`
  and make sure the requested checkpoints exist.
- Update tests and configs together. Stale config keys are a common failure mode.
