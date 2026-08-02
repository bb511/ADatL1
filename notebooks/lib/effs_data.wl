(* ::Package:: *)

(* effs_data.wl -- part of the notebooks/lib shared library for the
   effs_*.nb model-picking notebooks. Loaded with Get[]; all
   symbols live in Global`.

   Harvest-CSV ingestion: one record per retrained run (strategy,
   Q', Q'', per-signal test efficiencies), grouped into fronts.
   Expects the notebook to set paretoEffsDir before any call. *)

ClearAll[bkgDatasetQ, originalPicks, numOrMissing, strategyContextQ, perSignalCols, loadRuns, frontsOf, rebuttalRuns, dropEmptyModels];

(* Background/reference datasets excluded from the signal mean. No bare
   "shifted": robustad signals are shifted_anomaly_*, its background
   shifted_normal_all is caught via "normal". *)
bkgDatasetQ[ds_String] :=
  StringContainsQ[ds, "normal" | "SingleNeutrino" | "ZB_" | "reference"];

(* Original 250 Hz picks, from ORIGINAL_PICKS/VERBATIM_FALLBACKS in
   scripts/optuna/make_pareto_scripts.py (cvar25eff -> cvar25, drift -> stability). *)
originalPicks = <|
   "physics" -> <|
     "ae" -> <|"cap" -> 175, "cvar25" -> 169, "stability" -> 564, "wasserstein" -> 584|>,
     "vae" -> <|"cap" -> 179, "cvar25" -> 345, "stability" -> 529, "wasserstein" -> 539|>,
     "dsae" -> <|"cap" -> 570, "cvar25" -> 599, "stability" -> 565, "wasserstein" -> 383|>,
     "dsvae" -> <|"cap" -> 324, "cvar25" -> 372, "stability" -> 445, "wasserstein" -> 503|>,
     "svdd" -> <|"cap" -> 536, "cvar25" -> 739, "stability" -> 419, "wasserstein" -> 545|>,
     "realnvp" -> <|"cap" -> 376, "cvar25" -> 523, "stability" -> 505, "wasserstein" -> 383|>|>,
   "cifar10" -> <|
     "ae" -> <|"cap" -> 211, "cvar25" -> 592, "stability" -> 241, "wasserstein" -> 279|>,
     "vae" -> <|"cap" -> 520, "cvar25" -> 568, "stability" -> 587, "wasserstein" -> 596|>,
     "svdd" -> <|"cap" -> 599, "cvar25" -> 284, "stability" -> 191, "wasserstein" -> 266|>,
     "realnvp" -> <|"cap" -> 211, "cvar25" -> 535, "stability" -> 104, "wasserstein" -> 475|>|>,
   "robustad" -> <|
     "ae" -> <|"cap" -> 350, "cvar25" -> 526, "stability" -> 568, "wasserstein" -> 299|>,
     "vae" -> <|"cap" -> 402, "cvar25" -> 62, "stability" -> 389, "wasserstein" -> 587|>,
     "svdd" -> <|"cap" -> 546, "cvar25" -> 518, "stability" -> 525, "wasserstein" -> 581|>,
     "realnvp" -> <|"cap" -> 174, "cvar25" -> 333, "stability" -> 478, "wasserstein" -> 591|>|>|>;

numOrMissing[x_] := If[NumericQ[x], N[x], Missing["NA"]];

strategyContextQ[strat_String, ctx_String] :=
  StringStartsQ[ctx, "summary/eff/"] &&
   With[{m = StringDrop[ctx, StringLength["summary/eff/"]]},
    Switch[strat,
     "cvar25", m === "cvar25_ema/max",
     "cvar10", m === "cvar10_ema/max",
     "cap", StringStartsQ[m, "cap_ema"] && StringEndsQ[m, "/max"],
     "consistency", StringStartsQ[m, "consistency_ema"] && StringEndsQ[m, "/max"],
     "stability", StringContainsQ[m, "drift_ema"] && StringEndsQ[m, "/min"],
     "wasserstein", StringContainsQ[m, "w1dist_ema"] && StringEndsQ[m, "/min"],
     _, False]];

(* Per-signal eval columns: eff_<label>_<dataset>, excluding the eff_med_ /
   eff_min_ summaries; the plain eff_<label> scalar is the shortest prefix. *)
perSignalCols[hdr_List] := Module[{eff, base, cols},
  eff = Select[hdr, StringQ[#] && StringStartsQ[#, "eff_"] &&
      !StringStartsQ[#, "eff_med_"] && !StringStartsQ[#, "eff_min_"] &];
  base = SelectFirst[SortBy[eff, StringLength],
    Function[b, Count[eff, c_ /; StringStartsQ[c, b <> "_"]] >= 3]];
  cols = Select[eff, StringStartsQ[#, base <> "_"] &];
  <|"cols" -> cols,
    "names" -> (StringDrop[#, StringLength[base] + 1] & /@ cols)|>];

(* One record per run: strategy, Q' (optimized_main), Q'' (optimized_sec),
   per-signal TEST efficiencies at the run's own strategy checkpoint, and
   their mean over signal datasets. *)
loadRuns[exp_String] := Module[
  {raw, hdr, idx, rows, runs = <||>, eraw, ehdr, eidx, erows, ps, rn, strat,
   arr, sig},
  raw = Import[FileNameJoin[{paretoEffsDir, exp <> ".csv"}], "CSV"];
  hdr = First[raw]; rows = Rest[raw];
  idx = AssociationThread[hdr -> Range[Length[hdr]]];
  Do[If[row[[idx["ckpt"]]] === "strategy",
    runs[row[[idx["run_name"]]]] = <|
      "strategy" -> row[[idx["strategy"]]],
      "q0" -> numOrMissing[row[[idx["optimized_main"]]]],
      "q1" -> numOrMissing[row[[idx["optimized_sec"]]]]|>],
   {row, rows}];
  eraw = Import[FileNameJoin[{paretoEffsDir, exp <> "_eval.csv"}], "CSV"];
  ehdr = First[eraw]; erows = Rest[eraw];
  eidx = AssociationThread[ehdr -> Range[Length[ehdr]]];
  ps = perSignalCols[ehdr];
  Do[
   rn = row[[eidx["run_name"]]];
   If[KeyExistsQ[runs, rn] && row[[eidx["split"]]] === "test" &&
     strategyContextQ[runs[rn]["strategy"], row[[eidx["context"]]]],
    arr = AssociationThread[ps["names"] ->
       (numOrMissing[row[[eidx[#]]]] & /@ ps["cols"])];
    sig = DeleteMissing[KeySelect[arr, !bkgDatasetQ[#] &]];
    If[Length[sig] > 0,
     runs[rn] = Join[runs[rn],
       <|"testArr" -> sig, "eff" -> Mean[Values[sig]]|>]]],
   {row, erows}];
  runs];

frontsOf[runs_Association] := GroupBy[runs, #["strategy"] &];

(* A model whose campaign has not run yet has no harvest CSVs. Degrade
   gracefully: note it once and treat the architecture as absent, so every
   table and plot below simply omits it instead of failing on a ragged row.
   Both files of the pair are required -- loadRuns imports <exp>_eval.csv
   unconditionally, so a half-harvested pair would throw, not degrade. The
   miss is deliberately not memoised: dropping the CSVs into place mid-session
   is picked up by the next evaluation without re-Get-ing the library. *)
rebuttalRuns[exp_String] := Module[{f, fe},
   f = FileNameJoin[{paretoEffsDir, exp <> ".csv"}];
   fe = FileNameJoin[{paretoEffsDir, exp <> "_eval.csv"}];
   If[!FileExistsQ[f] || !FileExistsQ[fe],
     Print["note: no harvest CSV pair for ", exp, " -- omitting this model."];
     <||>,
     rebuttalRuns[exp] = loadRuns[exp]]];
dropEmptyModels[groups_Association] := Select[groups, Length[#] > 0 &];
