# Publishing the L1AD physics data

Builds the anonymous Zenodo deposition, and the HuggingFace mirror that goes up later.

## Order

```bash
export PROJECT_ROOT=/data/deodagiu/adl1t
RAW=/data/deodagiu/adl1t_data/parquet_files
OUT=/data/deodagiu/adl1t_publication

python scripts/publish_l1data/build_split_map.py --raw-data-dir $RAW --out $OUT
python scripts/publish_l1data/export_zenodo.py   --out $OUT --tar
python scripts/publish_l1data/dataset_card.py    --out $OUT
python scripts/publish_l1data/export_hf.py       --out $OUT
python scripts/publish_l1data/anonymise.py $OUT/adl1t-l1ad-v1
python scripts/publish_l1data/anonymise.py $OUT/tarballs
python scripts/publish_l1data/anonymise.py $OUT/huggingface
```

`run_export.sh` does all of it with the scans as blocking gates. Add `--only <dataset>`
to the two exporters to rehearse on one sample first; the smallest is
`SUSYGluGluToBBHTo2B_Par-M-1200` at 49,264 events, which takes a few seconds.

The split map depends on the frozen splits under
`<data_dir>/mlready/<name>/splits/`, written the first time
`L1DataMLReady.prepare()` runs. They are already in place for 2025E+G.

## What each piece does

| file | role |
|---|---|
| `anonymise.py` | The scrubbing rules, in one place. Tar flags, config filter, stray-file prune, and `scan()`, which walks the tree and every finished tar for anything identifying. Exits non-zero on a hit. |
| `build_split_map.py` | Maps the frozen split indices, which are positions in the *processed* rows, back onto raw row numbers through each dataset's cached event mask. Also writes the metadata the deposition ships. |
| `export_zenodo.py` | Consolidates each object's raw shards into one file, then takes each split's rows in the pipeline's permutation order and writes them back out sharded. Packs one tar per dataset. |
| `dataset_card.py` | Generates the deposition's README and LICENSE, with every number computed from the split summary rather than typed. |
| `export_hf.py` | Joins the per-object directories into one row-per-event table for HuggingFace. |
| `hf_assets/adl1t_l1ad.py` | Ships inside the deposition. Rename, cuts, normalise, pad -- reproduces the training tensor. |

## Resources

About 60 GB of scratch: the consolidated copy, the split tree and the HuggingFace tree
are each roughly the size of the source. Budget a few hours; the consolidation pass is
sequential I/O over 19.5 GB and the take pass is random access over it.

Output goes to `$OUT`. Nothing under `adl1t_data/`, `data/` or `logs/` is written, with
one exception: the frozen `splits/*.npz` live beside the mlready cache.

## Things that will bite

- **Config keys are not raw folder names.** `ZB_run396102` lives in
  `EphZB_2025E_run396102`. `build_split_map.py` records the mapping in
  `_splitmap/index.json`, which stays out of the published tree because it holds
  absolute paths.
- **Zero-bias order.** The pipeline concatenates the two runs in *sorted* order, which
  is not the order `configs/paths/data_2025E+G.yaml` lists them in. The split indices
  are global over that concatenation.
- **Zenodo files are frozen at publish.** Metadata stays editable, files do not. Run the
  scans before uploading, not after.
- **Do not add a repo link to the card.** The deposition is deliberately self-contained
  until the paper is accepted.
