# Confirmation runtime recovery — 2026-08-22

## Scope

This receipt records a runtime-only recovery during the frozen physics background
confirmation campaign. The training code, configuration, data, pairing artifacts,
Optuna selections, checkpoints, and campaign commit were not changed.

- Campaign commit: `ddfcfbb46d7fed7308b2066328b7b898a6316cc8`
- Campaign root: `/iopsstor/scratch/cscs/vjimenez/adatl1/campaigns/physics_background_confirmation_ddfcfbb`
- Frozen execution worktree: `/iopsstor/scratch/cscs/vjimenez/adatl1/worktrees/physics_background_confirmation_ddfcfbb`
- Lockfile SHA-256: `7e97bf062a6ba5f6a8d442aaab6dea3dbee6c8b8abb0ea2c98e6f4fac3686726`
- `pyproject.toml` SHA-256: `98cd659844f182725ffedc12a9e051539b40d83e4e724ff1ca9ff7c4ad93bcd5`
- uv: `0.11.32 (aarch64-unknown-linux-gnu)`

## Incident

At approximately 13:17 CEST, the shared Clariden environment lost Python package
source files during a broad interrupted reinstall/removal. This caused:

- retrain index 51 to reach epoch 200 and then fail while lazily loading Pillow's
  JPEG plugin for an end-of-fit plot;
- retrain index 1 to fail in the same end-of-fit path after reaching epoch 200;
- retrain index 6's successor to fail at startup because `MlflowClient` could no
  longer be imported.

The failure was external to model training. Earlier segments of the same runs had
successfully written the same plots, and exact epoch-200 checkpoints plus contiguous
MLflow epoch histories remained available for the two end-of-fit failures.

## Recovery

A replacement environment was built off-path from the frozen worktree using:

```bash
UV_PROJECT_ENVIRONMENT=/iopsstor/scratch/cscs/vjimenez/adatl1/.venv-clariden-recovery-ddfcfbb-v2 \
UV_CACHE_DIR=/iopsstor/scratch/cscs/vjimenez/adatl1/uv-cache-confirmation-recovery-ddfcfbb \
uv sync --locked --group dev
```

The cache was new and isolated because the former shared cache also contained an
incomplete `adl1t_datamaker` archive. After verification, the replacement directory
was moved to the unchanged runtime path `.venv-clariden`; the damaged directory was
preserved as `.venv-clariden-corrupt-20260822T1317`.
Because virtual-environment console-script shebangs retain the off-path build prefix,
`.venv-clariden-recovery-ddfcfbb-v2` is a symlink to `.venv-clariden`; direct Python
and installed console-script entry points therefore resolve the same verified files.

## Verification gates

- 204 installed-distribution `RECORD` files were present.
- All 41,726 `RECORD` paths existed; zero were missing.
- `MlflowClient`, Pillow/JPEG, Torch, PyTorch Lightning, `adl1t_datamaker`, and the
  L1 data extractor imported successfully.
- The frozen runner's exact-RNG validator accepted index 51's epoch-200 checkpoint.
- Torch was `2.12.0+cu130`, Lightning was `2.6.4`, and Optuna was `2.10.1`.

The sole `uv pip check` warning is a vendor metadata issue in
`nvidia-cusparselt-cu13==0.8.1`: the locked and downloaded wheel is AArch64 and its
library is an AArch64 ELF, but its internal `WHEEL` file uses the nonstandard tag
`manylinux2014_sbsa`. Its binary and metadata hashes match the previously working
installation. This is not a runtime or scientific compatibility change.

At 13:45 CEST, the failed indices were reconciled and submitted as mapped recovery
job array `3148945`. The campaign's existing exact-resume checks remain responsible
for authenticating live checkpoints, RNG state, contiguous epoch history, resolved
configuration, and final markers.
