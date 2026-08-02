#!/bin/bash
# Full export of the L1AD physics data for the anonymous Zenodo deposition.
#
# Run from the repository root on olqti. Takes a few hours and about 60 GB of scratch.
# Every leak scan is a blocking gate: nothing is packaged or uploaded if one fails,
# because Zenodo freezes a record's files at publish.
#
#   nohup bash scripts/publish_l1data/run_export.sh > export.log 2>&1 &
#
# Pass a dataset name to rehearse on one sample first, e.g.
#   bash scripts/publish_l1data/run_export.sh SUSYGluGluToBBHTo2B_Par-M-1200
set -euo pipefail

RAW=${RAW:-/data/deodagiu/adl1t_data/parquet_files}
OUT=${OUT:-/data/deodagiu/adl1t_publication}
export PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}

ONLY=()
if [ $# -gt 0 ]; then
    ONLY=(--only "$@")
    echo "Rehearsing on: $*"
fi

PUBLISH=scripts/publish_l1data
step () { echo; echo "=== $* ==="; date; }

step "1/5 split map and metadata"
python $PUBLISH/build_split_map.py --raw-data-dir "$RAW" --out "$OUT"

step "2/5 partition the raw parquet by split"
python $PUBLISH/export_zenodo.py --out "$OUT" "${ONLY[@]+"${ONLY[@]}"}"

step "3/5 dataset card"
python $PUBLISH/dataset_card.py --out "$OUT"

step "4/5 leak scan before packaging"
python $PUBLISH/anonymise.py "$OUT/adl1t-l1ad-v1"
python $PUBLISH/anonymise.py "$OUT/metadata"

step "5/5 assemble the upload payload, then scan it"
python $PUBLISH/export_zenodo.py --out "$OUT" --pack-only
( cd "$OUT/payload" && sha256sum ./* > sha256sums.txt 2>/dev/null || true )
python $PUBLISH/anonymise.py "$OUT/payload"

step "done"
echo "Upload everything in: $OUT/payload"
echo "  $(ls "$OUT/payload" | wc -l) files, $(du -sh "$OUT/payload" | cut -f1) -- Zenodo allows 100 files / 50 GB"
rm -rf "$OUT/_work" && echo "  removed the consolidation scratch"
echo
echo "The HuggingFace mirror is NOT built here; it is published only after acceptance:"
echo "  python $PUBLISH/export_hf.py --out $OUT"
