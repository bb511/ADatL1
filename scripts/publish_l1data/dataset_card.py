# Generates the README and LICENSE that ship inside the deposition.
#
# The deposition is self-contained on purpose: it links to no repository, so everything
# a reader needs to interpret the files has to be stated here. Nothing written by this
# module may name a person, an institution, an account or a machine.
import argparse
import json
import sys
from pathlib import Path

LICENCE = """Creative Commons Attribution 4.0 International (CC BY 4.0)

You are free to share and adapt this material for any purpose, including
commercially, provided you give appropriate credit, link to the licence, and
indicate if changes were made.

Licence deed:  https://creativecommons.org/licenses/by/4.0/
Legal code:    https://creativecommons.org/licenses/by/4.0/legalcode

Attribution details will be completed on publication of the accompanying data
descriptor. Until then, cite this record by its DOI.
"""

CARD = """# L1 trigger anomaly-detection dataset

Level-1 trigger objects from the CMS experiment at the LHC: {zb_events:,} zero-bias
events recorded in 2025, plus {n_signal} simulated signal samples and one simulated
zero-bias-like background sample. It is built for unsupervised anomaly detection --
train on the zero-bias data, validate against signals the model never saw.

Every event and every object is here. The saturation cuts the accompanying study
applied are *documented below, not applied to the files*, so you can reproduce that
study exactly or make different choices.

## Layout

```
adl1t-l1ad-v1/
  zerobias/<run>/{{train,valid,test}}/<object>/*.parquet
  signal/<sample>/{{valid,test}}/<object>/*.parquet
  background/<sample>/{{valid,test}}/<object>/*.parquet
  metadata/
  README.md  LICENSE
```

One directory per object collection, each holding sharded parquet. All objects of a
split have the same number of rows in the same order, so row *i* of `jets/` and row *i*
of `muons/` are the same event.

Objects: `ET`, `FET`, `FHT`, `HT`, `MET`, `MHT`, `cica`, `egammas`, `event_info`,
`jets`, `muons`, `seeds`, `taus`. Not every sample has every object.

`seeds/` is the full ~190-bit L1 trigger menu decision per event and is about a third of
the total volume; skip it if you only want kinematics.

## Splits

The zero-bias data is split 60/20/20. The simulated samples are validation-only and are
split 60/40 between `valid` and `test`.

| split | events |
|---|---|
{split_table}

The split was drawn once with NumPy's PCG64 generator seeded with **{seed}**, over the
events passing the event cut below, with the two zero-bias runs concatenated in the
order `{zb_order}`. `metadata/split_indices/` carries the raw index arrays if you want
to audit it.

## Reproducing the study's preprocessing

Four steps, in this order. `metadata/data_config_resolved.yaml` records the exact
configuration.

**1. Rename.** The files carry the original ntuple field names. The study renames them:

| object | raw | renamed |
|---|---|---|
| muons | `muonIEt`, `muonIEta`, `muonIPhi` | `Et`, `eta`, `phi` |
| jets | `jetIEt`, `jetIEta`, `jetIPhi` | `Et`, `eta`, `phi` |
| egammas | `egIEt`, `egIEta`, `egIPhi` | `Et`, `eta`, `phi` |
| taus | `tauIEt`, `tauIEta`, `tauIPhi` | `Et`, `eta`, `phi` |

**2. Saturation cuts.** These are hardware counter limits, not physics selections.

| kind | cut | effect |
|---|---|---|
| event | `ET.Et < 4095` | drops the event entirely |
| object | `Et < 511` on muons, egammas, jets, taus | removes that object, keeps the event |
| object | `FET.Et < 4095` | removes that entry, keeps the event |

The event cut removes {dropped:,} of {total_raw:,} zero-bias events ({dropped_pct}).
Those events **are published**; they carry `order = -1` in `event_info` and sit at the
end of their split.

**3. Normalise.** Median and interquartile range fitted on the training split only.
`metadata/norm_params_robust.json` holds the shift and scale used by the study;
`norm_params_standard.json` is a mean/standard-deviation alternative.

**4. Pad to a fixed shape.** Keep {nconst}, padding with zeros, which gives
`(N, 39, 3)` -- flatten for the 117 features the models take. A companion boolean mask
marks the real constituents.

`adl1t_l1ad.py` in this record does all four:

```python
import adl1t_l1ad as l1

objects = l1.read_split("adl1t-l1ad-v1/zerobias/{zb_first}/train")
norms = l1.load_norm_params("metadata/norm_params_robust.json")
x, mask = l1.to_model_tensor(objects, norms)       # (N, 39, 3)
```

Pass `apply_cuts=False` to keep everything, or call `l1.apply_saturation_cuts` with your
own thresholds.

## Units

Values are **integer hardware units**, as the trigger produces them. Nothing in the
files is scaled. `metadata/l1_scales.yaml` gives the conversion factors -- transverse
energy is 0.5 GeV per unit, jet/e-gamma/tau azimuth 0.043633231299858 rad, muon azimuth
0.010908307824965 rad, muon pseudorapidity 0.010875.

## Reading the row order

Within a split, rows are in the order the study consumed them, so reading front to back
after applying the event cut reproduces its input row for row. `event_info` carries two
extra columns: `split`, so a file separated from its directory is still self-describing,
and `order`, the position in that ordering (`-1` for events the study's cut removed).

## Notes for comparison

If you are comparing against the accompanying study: it evaluated each simulated sample
on only the first 163,840 events of that sample's split, while using the zero-bias split
in full. Sample labels there are assigned by sorted sample name, with zero bias 0, the
simulated background -1, and the signals 1 upward.

## Provenance

Zero-bias data: CMS, 2025, runs {runs}. Simulated samples: CMS Run 3 Winter25 campaign.
Values are the Level-1 trigger's own reconstructed objects, not offline reconstruction.

## Licence

CC BY 4.0. See `LICENSE`.
"""


def render(summary: dict) -> str:
    """Fill the card from the measured split summary."""
    datasets = summary["datasets"]
    zb = {k: v for k, v in datasets.items() if v["category"] == "zerobias"}
    zb_order = " then ".join(sorted(zb))
    total_raw = sum(v["raw_events"] for v in zb.values())
    kept = sum(v["events_passing_filter"] for v in zb.values())
    dropped = total_raw - kept

    rows = []
    for split in ("train", "valid", "test"):
        n = sum(v["counts"].get(split, 0) for v in zb.values())
        if n:
            rows.append(f"| zero-bias {split} | {n:,} |")
    for name in sorted(k for k, v in datasets.items() if v["category"] != "zerobias"):
        counts = datasets[name]["counts"]
        rows.append(
            f"| {name} valid / test | {counts.get('valid', 0):,} / {counts.get('test', 0):,} |"
        )

    return CARD.format(
        zb_events=total_raw,
        n_signal=sum(1 for v in datasets.values() if v["category"] == "signal"),
        split_table="\n".join(rows),
        seed=summary["split_seed"],
        zb_order=zb_order,
        dropped=dropped,
        total_raw=total_raw,
        dropped_pct=f"{100 * dropped / total_raw:.4f}%",
        nconst="4 muons, 10 jets, 12 e-gammas, 12 taus and the 1 FET entry",
        zb_first=sorted(zb)[0],
        runs=", ".join(sorted(k.replace("ZB_run", "") for k in zb)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="publication root")
    args = parser.parse_args()

    summary = json.loads((args.out / "metadata" / "split_summary.json").read_text())
    tree = args.out / "adl1t-l1ad-v1"
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "README.md").write_text(render(summary))
    (tree / "LICENSE").write_text(LICENCE)
    print(f"wrote {tree}/README.md and LICENSE")

    return 0


if __name__ == "__main__":
    sys.exit(main())
