# Deterministic JetCLR pairing of 2025 ZeroBias runs

## Sources

- Target: `ZB_run396102`, 10,363,545 events.
- Reference: `ZB_run398183`, 10,523,634 events.
- Event order: unchanged row order of the processed, event-aligned Parquet tables.
- Encoder: frozen JetCLR seed-456 epoch-1 checkpoint, SHA-256
  `8ce02d89201f130b5c97526ab95ae68badb295928da2f10768b0ea40e9993ecf`.

The processed features are normalized and padded with the same fitted parameters
and object-feature map used during JetCLR training. A replayed shuffled training
row matched the original cached value and mask tensors bit-for-bit.

## Pairing contract

The release is under:

```text
/iopsstor/scratch/cscs/vjimenez/jetclr/pairings/zb_runs_full_20260802_e1b728c/pairing
```

`ZB_run396102_to_ZB_run398183_index.pt` is the primary standalone tensor. It is a
`torch.int64` tensor with shape `[10_363_545]`; for an event index `i` in run
396102, `pairing[i]` is its unique partner's row index in run 398183. Its SHA-256
is `f7ba5370fd314aae07150b7cf1791aa5a183ae841274e0e305474dd7cbfa7ca6`.

`ZB_run398183_to_ZB_run396102_index.pt` is the inverse tensor with shape
`[10_523_634]`. Its 160,089 unused reference rows contain `-1`. Its SHA-256 is
`a41a1a0c8a500ab32e40b335ae709543cb607297a17ba956dbd147ba164b0053`.

`ZB_run396102_to_ZB_run398183.pt` is the complete versioned artifact. In addition
to both index tensors, it contains distance, validity, candidate-rank, dataset,
strategy, and provenance metadata. Its SHA-256 is
`9e1acf0b696c6c9a0af80dca5293d57a31c925f32f7a327e1d8c06796966260f`.

## Validation

- Complete target coverage: 10,363,545 / 10,363,545.
- Unique reference indices: 10,363,545 / 10,363,545.
- Inverse-map consistency: exact for every assigned row.
- Mean embedding distance: 0.17452.
- Median / p95 / p99 distance: 0.14790 / 0.37745 / 0.65064.
- Maximum distance: 1.45548.
- A 10,000-pair recomputation matched stored distances within `5.97e-8`.
- On the 32,768-event exact-search audit, scalable IVF pairing retained 84.7% of
  exact pair identities and increased mean distance by 0.94%.

The scalable search is deterministic IVF candidate retrieval followed by
deterministic one-to-one proposal resolution. It is not a globally optimal
bipartite assignment. Every smaller-run event is paired; no caliper rejection or
index-order fallback was used.
