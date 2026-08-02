# JetCLR AAD L1 encoder: final model card

## Selected encoder

- Architecture: two-layer `ObjectTransformer`, CLS pooling, `d_model=128`,
  `out_dim=128`, 8 attention heads, 512-wide feed-forward layers, pre-norm and
  post-pooling normalization.
- Projection head used only for contrastive training: `[256, 256] -> 128`, batch
  normalization and GELU.
- Objective: JetCLR contrastive loss at temperature 0.1, augmented with encoder
  VICReg variance/covariance penalties of 0.5/0.005.
- Augmentations: detector smearing `(p=0.5, strength=0.3)` and object masking
  `(p=0.5, per-object p=0.025)`; Lorentz rotation disabled.
- Optimization: AdamW, learning rate `2e-4`, weight decay `1e-4`, batch size
  8192, gradient clipping 0.1.
- Training stop: epoch 1 / 1,530 optimizer steps. The cosine schedule retains
  the 24,480-step horizon used by the 16-epoch trajectory study.
- Frozen deployment seed: 456, chosen before test as the checkpoint closest to
  the eight-seed validation median—not from test performance.

## Evidence for the stopping point

Across six eligible hybrid runs, epoch 2 gave the highest median validation
worst-quartile AUROC (0.7462). The one-standard-error rule selected the earliest
statistically competitive checkpoint, epoch 1 (threshold 0.7346; epoch-1 median
0.7370). From epoch 8 to 16 the paired score declined by 0.00922 on average
(SE 0.00330), so the preregistered epoch-32 extension criterion failed.

An independent five-seed paired confirmation promoted the VICReg-regularized
recipe. Relative to the unregularized control, encoder effective rank improved
35.1%, participation rank improved 38.2%, top-PC fraction fell 0.0623, raw
pairing score improved 17.0%, macro mean AUROC improved 0.0203, and
worst-quartile AUROC improved 0.0197 on average. Across all eight validation
seeds, macro mean AUROC was 0.8610 +/- 0.0147 SD and worst-quartile AUROC was
0.7527 +/- 0.0095 SD.

## Sealed held-out test

The seven completed frozen checkpoints achieved:

| Metric | Mean +/- SD | Median |
|---|---:|---:|
| Macro mean AUROC | 0.8319 +/- 0.0187 | 0.8366 |
| Macro median AUROC | 0.8272 +/- 0.0227 | 0.8327 |
| Worst-quartile mean AUROC | 0.7382 +/- 0.0114 | 0.7436 |
| Macro median AUPRC | 0.7750 +/- 0.0265 | 0.7799 |
| Encoder effective rank | 15.9037 +/- 1.0433 | 16.2212 |

The preregistered seed-456 checkpoint completed with macro mean AUROC 0.8561,
macro median AUROC 0.8596, worst-quartile AUROC 0.7451, macro median AUPRC
0.8165, and encoder effective rank 14.7379. It remains the deployment encoder;
no post-test model selection was performed.

One of eight planned processes (seed 1337) was SIGKILLed by a node OOM after
validation began but before any test-phase marker. The one-shot policy forbade a
retry. Its absence and authenticated chronology are recorded in the Stage-9
summary; the reported aggregate therefore has n=7 and is explicitly not a
complete eight-seed population.

## Artifacts

The release directory is
`/iopsstor/scratch/cscs/vjimenez/jetclr/final/jetclr_aad_best_20260801`.

- `jetclr_aad_seed456_epoch1.ckpt`: full Lightning checkpoint, SHA-256
  `8ce02d89201f130b5c97526ab95ae68badb295928da2f10768b0ea40e9993ecf`.
- `jetclr_aad_seed456_encoder.pt`: encoder-only state dict with the `model.`
  prefix removed, SHA-256
  `7d73823b74445e822e4d97495223b287a90a8559e0a97704380c8636a23db7ec`.
- `jetclr_aad_best.yaml`: frozen Hydra recipe.
- `results/`: authenticated milestone, confirmation, and sealed-test summaries.
- `MANIFEST.json`: hashes and provenance for every packaged artifact.

The full checkpoint reports `epoch=0`, meaning one completed epoch in Lightning,
and `global_step=1530`; its scheduler state records `total_steps=24480` and
`last_epoch=1530`.
