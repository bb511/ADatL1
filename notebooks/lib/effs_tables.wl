(* ::Package:: *)

(* effs_tables.wl -- part of the notebooks/lib shared library for the
   effs_*.nb model-picking notebooks. Loaded with Get[]; all
   symbols live in Global`.

   LaTeX/markdown table builders over the picked efficiencies:
   legacy compact tables with bootstrap CIs and Holm-adjusted
   p-values, rule tables, Spearman tables/matrices, and the
   quantile/metric comparison axes. *)

ClearAll[legacyStratKey, toLegacyGroups, ArchitectureRowDataCompactCollapse, LatexRowCompactAsymCollapse,
  BuildArchitectureRowsCompactCollapse, MarkdownRowCompactAsymCollapse, BuildLatexTableCompactAsymCollapse,
  BuildMarkdownTableCompactAsymCollapse, LatexTableFromRows, MarkdownTableFromRows, ruleTableRows,
  ruleTableLatex, ruleTableMarkdown, spearmanTableLatex, spearmanTableMarkdown, spearmanDisplayOrder,
  spearmanCellString, spearmanMatrixCells, spearmanMatrixLatex, spearmanMatrixMarkdown, axisPair,
  QuantileAxisCells, QuantileAxisLatex, QuantileAxisMarkdown, MetricAxisSummary];

(* Key translation between the rebuttal strategy names and the legacy
   association keys used by BuildLatexTableCompactAsymCollapse. Missing
   strategies are zero-filled (the legacy collapsed-method convention). *)
legacyStratKey = <|"cvar25" -> "semisupervised", "stability" -> "stable",
   "cap" -> "CAP", "wasserstein" -> "wasserstein", "consistency" -> "consistency"|>;
toLegacyGroups[groups_Association] := Association @@ KeyValueMap[
   Function[{m, sub}, Module[
     {len = If[Length[sub] > 0, Length[First[Values[sub]]], 1]},
     m -> Association @@ Table[
        legacyStratKey[s] -> If[KeyExistsQ[sub, s], N[sub[s]],
          ConstantArray[-1., len]], {s, boxStrategyOrder}]]],
   dropEmptyModels[groups]];

ArchitectureRowDataCompactCollapse[archName_String, methods_Association, nBoot_:10000, alpha_:0.05] := Module[{semi,
    cap, cons, stable, w1, semiCollapsed, capCollapsed, consCollapsed, stableCollapsed, w1Collapsed,
    dCC, dCS, dCW, ciCC, ciCS, ciCW, pCC, pCS, pCW}, semi = methods["semisupervised"]; cap = methods["CAP"];
  stable = methods["stable"]; w1 = methods["wasserstein"]; cons = Lookup[methods, "consistency",
    ConstantArray[0., Length[cap]]]; If[ !Length[semi] == Length[cap] == Length[cons] == Length[stable] == Length[w1],
    Return[Association["Architecture" -> archName, "Error" -> "All method arrays must have equal lengths."]]];
  semiCollapsed = CollapsedMethodQ[semi]; capCollapsed = CollapsedMethodQ[cap]; consCollapsed = CollapsedMethodQ[cons];
  stableCollapsed = CollapsedMethodQ[stable]; w1Collapsed = CollapsedMethodQ[w1]; dCC = If[capCollapsed || consCollapsed,
    Missing["Collapsed"], Mean[cap - cons]]; dCS = If[capCollapsed || stableCollapsed, Missing["Collapsed"],
    Mean[cap - stable]]; dCW = If[capCollapsed || w1Collapsed, Missing["Collapsed"], Mean[cap - w1]];
  ciCC = If[capCollapsed || consCollapsed, Missing["Collapsed"], BootstrapPairedDiffCI[cap,
      cons, nBoot, alpha]]; ciCS = If[capCollapsed || stableCollapsed, Missing["Collapsed"],
    BootstrapPairedDiffCI[cap, stable, nBoot, alpha]]; ciCW = If[capCollapsed || w1Collapsed,
    Missing["Collapsed"], BootstrapPairedDiffCI[cap, w1, nBoot, alpha]]; pCC = If[capCollapsed || consCollapsed,
    Missing["Collapsed"], SafeWilcoxonP[cap, cons]]; pCS = If[capCollapsed || stableCollapsed,
    Missing["Collapsed"], SafeWilcoxonP[cap, stable]]; pCW = If[capCollapsed || w1Collapsed,
    Missing["Collapsed"], SafeWilcoxonP[cap, w1]]; Association["Architecture" -> archName, "Semi" -> semi,
    "CAP" -> cap, "Cons" -> cons, "Stable" -> stable, "W1" -> w1, "CAPMinusCons" -> dCC, "CAPMinusConsCI" -> ciCC,
    "CAPMinusConsP" -> pCC, "CAPMinusStable" -> dCS, "CAPMinusStableCI" -> ciCS, "CAPMinusStableP" -> pCS,
    "CAPMinusW1" -> dCW, "CAPMinusW1CI" -> ciCW, "CAPMinusW1P" -> pCW]];

LatexRowCompactAsymCollapse[row_Association, digits_:2] := StringJoin[StringRiffle[{row["Architecture"],
      FormatMeanOrDash[row["Semi"], digits], FormatMeanOrDash[row["CAP"], digits], FormatMeanOrDash[Lookup[row,
        "Cons", {0.}], digits], FormatMeanOrDash[row["Stable"], digits], FormatMeanOrDash[row["W1"],
        digits], FormatAsymPMOrDash[Lookup[row, "CAPMinusCons", Missing["Collapsed"]], Lookup[row,
        "CAPMinusConsCI", Missing["Collapsed"]], digits], FormatP[Lookup[row, "CAPMinusConsP",
        Missing["Collapsed"]]], FormatAsymPMOrDash[row["CAPMinusStable"], row["CAPMinusStableCI"],
        digits], FormatP[row["CAPMinusStableP"]], FormatAsymPMOrDash[row["CAPMinusW1"], row["CAPMinusW1CI"],
        digits], FormatP[row["CAPMinusW1P"]]}, " & "], " \\\\"];

BuildArchitectureRowsCompactCollapse[allArchitectures_Association, nBoot_:10000, alpha_:0.05] := KeyValueMap[ArchitectureRowDataCompactCollapse[#1, #2, nBoot, alpha] & , allArchitectures]; 

(* Markdown twins of the table builders. The legacy-format pair works from
   pre-built rows (BuildArchitectureRowsCompactCollapse) so LaTeX and
   markdown show identical bootstrap CIs. *)
MarkdownRowCompactAsymCollapse[row_Association, digits_:2] :=
  StringJoin["| ", StringRiffle[{row["Architecture"],
      FormatMeanOrDash[row["Semi"], digits],
      FormatMeanOrDash[row["CAP"], digits],
      FormatMeanOrDash[Lookup[row, "Cons", {0.}], digits],
      FormatMeanOrDash[row["Stable"], digits],
      FormatMeanOrDash[row["W1"], digits],
      FormatAsymPMOrDash[Lookup[row, "CAPMinusCons", Missing["Collapsed"]],
       Lookup[row, "CAPMinusConsCI", Missing["Collapsed"]], digits],
      FormatP[Lookup[row, "CAPMinusConsP", Missing["Collapsed"]]],
      FormatAsymPMOrDash[row["CAPMinusStable"], row["CAPMinusStableCI"],
       digits], FormatP[row["CAPMinusStableP"]],
      FormatAsymPMOrDash[row["CAPMinusW1"], row["CAPMinusW1CI"], digits],
      FormatP[row["CAPMinusW1P"]]}, " | "], " |"];

ruleTableMarkdown[rows_List, ruleLabel_String] := StringJoin[
   "| Arch. | Strategy | n | Pick | eff | Oracle | eff_or | d_or | Orig. | ",
   "eff_o | d_o |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
   StringRiffle[
    "| " <> StringRiffle[{#["model"], strategyPretty[#["strategy"]],
         ToString[#["n"]], fmtTrial[#["pick"]], fmtEffPct[#["pickEff"]],
         fmtTrial[#["oracle"]], fmtEffPct[#["oracleEff"]],
         fmtDelta[#["pickEff"], #["oracleEff"]], fmtTrial[#["orig"]],
         fmtEffPct[#["origEff"]],
         fmtDelta[#["pickEff"], #["origEff"]]}, " | "] <> " |" & /@ rows,
    "\n"]];

spearmanTableMarkdown[modelExps_Association, domain_String] :=
  Module[{rows = {}},
   Do[Module[{model = modelKey, runs, fronts},
     runs = rebuttalRuns[modelExps[modelKey]];
     fronts = frontsOf[runs];
     Do[If[KeyExistsQ[fronts, strat], Module[{front = fronts[strat], rho},
        rho = frontSpearman[front, strat];
        AppendTo[rows, {model, strategyPretty[strat],
          ToString[Length[front]],
          If[NumericQ[rho], fmtNum[rho, 2], "-"]}]]],
      {strat, strategyOrder}]],
    {modelKey, Keys[modelExps]}];
   StringJoin["| Arch. | Strategy | n | rho |\n|---|---|---|---|\n",
    StringRiffle[("| " <> StringRiffle[#, " | "] <> " |") & /@ rows,
     "\n"]]];

(* Compact Spearman matrix: one row per architecture, one column per training
   strategy (legacy order); cell = rho (n), with n the number of front points
   entering the ranking. "-" = strategy absent; "- (n)" = rho undefined
   (n < 3 or tied/zero-variance efficiencies). *)
spearmanDisplayOrder = {"cvar25", "cap", "consistency", "stability", "wasserstein"};
spearmanMatrixCells[modelExps_Association] := Table[
   Prepend[Table[spearmanCellString[frontsOf[rebuttalRuns[modelExps[m]]], s],
     {s, spearmanDisplayOrder}], m], {m, Keys[modelExps]}];
spearmanMatrixLatex[modelExps_Association] := StringJoin[
   "\\begin{tabular}{lccccc}\n\\toprule\n",
   "Arch. & Semi & CAP & Cons & Stable & W1 \\\\\n\\midrule\n",
   StringRiffle[StringRiffle[#, " & "] <> " \\\\" & /@
     spearmanMatrixCells[modelExps], "\n"],
   "\n\\bottomrule\n\\end{tabular}"];
spearmanMatrixMarkdown[modelExps_Association] := StringJoin[
   "| Arch. | Semi | CAP | Cons | Stable | W1 |\n|---|---|---|---|---|---|\n",
   StringRiffle[("| " <> StringRiffle[#, " | "] <> " |") & /@
     spearmanMatrixCells[modelExps], "\n"]];

(* Corrected builders: every table generated through these prints
   Holm-adjusted p-values (overrides the raw legacy definition). *)
BuildLatexTableCompactAsymCollapse[allArchitectures_Association,
   nBoot_:10000, alpha_:0.05, digits_:2] :=
  LatexTableFromRows[HolmAdjustRows[
    BuildArchitectureRowsCompactCollapse[allArchitectures, nBoot, alpha]],
   digits];
BuildMarkdownTableCompactAsymCollapse[allArchitectures_Association,
   nBoot_:10000, alpha_:0.05, digits_:2] :=
  MarkdownTableFromRows[HolmAdjustRows[
    BuildArchitectureRowsCompactCollapse[allArchitectures, nBoot, alpha]],
   digits];

(* One rule-table row per (model, strategy) front. pickF[front, strat] must
   return a run name (or Missing). *)
ruleTableRows[modelExps_Association, domain_String, pickF_] := Module[{rows = {}},
  Do[Module[{model = modelKey, exp = modelExps[modelKey], runs, fronts},
    runs = rebuttalRuns[exp];
    fronts = frontsOf[runs];
    Do[If[KeyExistsQ[fronts, strat], Module[
       {front = fronts[strat], pick, oracle, orig, origTrial, effOf},
       effOf[name_] := If[StringQ[name] && KeyExistsQ[front, name],
         front[name]["eff"], Missing["NA"]];
       pick = pickF[front, strat, model];
       oracle = oraclePick[front];
       origTrial = Query[ToLowerCase[model], strat][
         Lookup[originalPicks, domain, <||>]];
       orig = If[IntegerQ[origTrial],
         strat <> "_t" <> ToString[origTrial], Missing["NA"]];
       AppendTo[rows, <|"model" -> model, "strategy" -> strat,
         "n" -> Length[front], "pick" -> pick, "pickEff" -> effOf[pick],
         "pickQ" -> If[StringQ[pick], front[pick]["q0"], Missing["NA"]],
         "oracle" -> oracle, "oracleEff" -> effOf[oracle],
         "orig" -> orig, "origEff" -> effOf[orig]|>]]],
     {strat, strategyOrder}]],
   {modelKey, Keys[modelExps]}];
  rows];

ruleTableLatex[rows_List, ruleLabel_String] := Module[{body},
  body = StringRiffle[
    StringRiffle[{#["model"], strategyPretty[#["strategy"]],
        ToString[#["n"]], fmtTrial[#["pick"]], fmtEffPct[#["pickEff"]],
        fmtTrial[#["oracle"]], fmtEffPct[#["oracleEff"]],
        fmtDelta[#["pickEff"], #["oracleEff"]],
        fmtTrial[#["orig"]], fmtEffPct[#["origEff"]],
        fmtDelta[#["pickEff"], #["origEff"]]}, " & "] <> " \\\\" & /@ rows,
    "\n"];
  StringJoin["% rule: ", ruleLabel, "\n",
   "\\begin{tabular}{llccccccccc}\n\\toprule\n",
   "Arch. & Strategy & $n$ & Pick & $\\langle\\varepsilon\\rangle$ & ",
   "Oracle & $\\langle\\varepsilon\\rangle_{\\mathrm{or}}$ & $\\Delta_{\\mathrm{or}}$ & ",
   "Orig. & $\\langle\\varepsilon\\rangle_{\\mathrm{o}}$ & $\\Delta_{\\mathrm{o}}$ \\\\\n",
   "\\midrule\n", body, "\n\\bottomrule\n\\end{tabular}"]];

spearmanTableLatex[modelExps_Association, domain_String] := Module[{rows = {}},
  Do[Module[{model = modelKey, runs, fronts},
    runs = rebuttalRuns[modelExps[modelKey]];
    fronts = frontsOf[runs];
    Do[If[KeyExistsQ[fronts, strat], Module[{front = fronts[strat], rho},
       rho = frontSpearman[front, strat];
       AppendTo[rows, <|"model" -> model, "strategy" -> strat,
         "n" -> Length[front],
         "rho" -> If[NumericQ[rho], fmtNum[rho, 2], "-"]|>]]],
     {strat, strategyOrder}]],
   {modelKey, Keys[modelExps]}];
  StringJoin["\\begin{tabular}{llcc}\n\\toprule\n",
   "Arch. & Strategy & $n$ & $\\rho_{\\mathrm{Spearman}}$ \\\\\n\\midrule\n",
   StringRiffle[
    StringRiffle[{#["model"], strategyPretty[#["strategy"]], ToString[#["n"]],
        #["rho"]}, " & "] <> " \\\\" & /@ rows, "\n"],
   "\n\\bottomrule\n\\end{tabular}"]];

(* Rebuttal axes over the legacy data.
   Axis 1 (LHC): both operating points side by side in one table; the Holm
   family is per operating point (12 tests), consistent with the main tables.
   Axis 2 (CIFAR10/RobustAD): compact per-model summary + counts over the
   AUPRC comparisons, backing a text paragraph. Stars mark Holm-adjusted
   p < 0.05; collapsed comparisons render as n/a. *)
QuantileAxisCells[assocTail_Association, assocQ99_Association, latexQ_,
   nBoot_:10000, alpha_:0.05] := Module[
  {rT = HolmAdjustRows[
     BuildArchitectureRowsCompactCollapse[assocTail, nBoot, alpha]],
   rQ = HolmAdjustRows[
     BuildArchitectureRowsCompactCollapse[assocQ99, nBoot, alpha]]},
  MapThread[Function[{a, b}, Join[{a["Architecture"]},
     axisPair[a, "CAPMinusCons", "CAPMinusConsP", latexQ],
     axisPair[a, "CAPMinusStable", "CAPMinusStableP", latexQ],
     axisPair[a, "CAPMinusW1", "CAPMinusW1P", latexQ],
     axisPair[b, "CAPMinusCons", "CAPMinusConsP", latexQ],
     axisPair[b, "CAPMinusStable", "CAPMinusStableP", latexQ],
     axisPair[b, "CAPMinusW1", "CAPMinusW1P", latexQ]]], {rT, rQ}]];

QuantileAxisLatex[assocTail_Association, assocQ99_Association,
   nBoot_:10000, alpha_:0.05] := StringJoin[
  "\\begin{tabular}{lcccccccccccc}\n\\toprule\n",
  " & \\multicolumn{6}{c}{$q \\approx 0.999991$ (250\\,Hz)} & ",
  "\\multicolumn{6}{c}{$q = 0.99$} \\\\\n",
  "Arch. & $\\Delta_{\\mathrm{C-C}}$ & $p$ & $\\Delta_{\\mathrm{C-S}}$ & $p$ & $\\Delta_{\\mathrm{C-W}}$ & $p$ & ",
  "$\\Delta_{\\mathrm{C-C}}$ & $p$ & $\\Delta_{\\mathrm{C-S}}$ & $p$ & $\\Delta_{\\mathrm{C-W}}$ & $p$ \\\\\n\\midrule\n",
  StringRiffle[StringRiffle[#, " & "] <> " \\\\" & /@
    QuantileAxisCells[assocTail, assocQ99, True, nBoot, alpha], "\n"],
  "\n\\bottomrule\n\\end{tabular}"];

QuantileAxisMarkdown[assocTail_Association, assocQ99_Association,
   nBoot_:10000, alpha_:0.05] := StringJoin[
  "| Arch. | d(C-C) tail | p | d(C-S) tail | p | d(C-W) tail | p | ",
  "d(C-C) q99 | p | d(C-S) q99 | p | ",
  "d(C-W) q99 | p |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
  StringRiffle[("| " <> StringRiffle[#, " | "] <> " |") & /@
    QuantileAxisCells[assocTail, assocQ99, False, nBoot, alpha], "\n"]];

MetricAxisSummary[assoc_Association, label_String:"AUPRC",
   nBoot_:10000, alpha_:0.05] := Module[
  {rows = HolmAdjustRows[
     BuildArchitectureRowsCompactCollapse[assoc, nBoot, alpha]],
   lines = {}, sigFor = 0, sigAgainst = 0, na = 0, bestCells = 0,
   tally, fmtDP},
  tally[d_, p_] := Which[!NumericQ[d] || !NumericQ[p], na++,
    d > 0 && p < 0.05, sigFor++, d < 0 && p < 0.05, sigAgainst++,
    True, Null];
  fmtDP[d_, p_] := If[!NumericQ[d] || !NumericQ[p], "n/a",
    fmtNum[d, 2] <> " (" <> FormatP[p] <>
     If[p < 0.05, "*", ""] <> ")"];
  Do[Module[{cap = Mean[r["CAP"]], st = Mean[r["Stable"]],
     w1 = Mean[r["W1"]], consArr = Lookup[r, "Cons", {0.}], consOK,
     best},
    (* A collapsed/absent consistency column must not make CAP look 'best'
       by comparing against an all-zero array. *)
    consOK = !CollapsedMethodQ[consArr];
    best = cap >= st && cap >= w1 && (!consOK || cap >= Mean[consArr]);
    If[best, bestCells++];
    tally[Lookup[r, "CAPMinusCons"], Lookup[r, "CAPMinusConsP"]];
    tally[Lookup[r, "CAPMinusStable"], Lookup[r, "CAPMinusStableP"]];
    tally[Lookup[r, "CAPMinusW1"], Lookup[r, "CAPMinusW1P"]];
    AppendTo[lines, StringJoin["| ", r["Architecture"], " | ",
      fmtNum[cap, 2], If[best, " (best)", ""], " | ",
      fmtDP[Lookup[r, "CAPMinusCons"], Lookup[r, "CAPMinusConsP"]],
      " | ",
      fmtDP[Lookup[r, "CAPMinusStable"], Lookup[r, "CAPMinusStableP"]],
      " | ",
      fmtDP[Lookup[r, "CAPMinusW1"], Lookup[r, "CAPMinusW1P"]], " |"]]],
   {r, rows}];
  StringJoin["Axis 2 (", label, ") summary -- ",
   "star = Holm-adjusted p < 0.05, family = this table:\n",
   "| Arch. | CAP | d CAP-Cons (p) | d CAP-Stable (p) | d CAP-W1 (p) |\n",
   "|---|---|---|---|---|\n", StringRiffle[lines, "\n"], "\n",
   "counts: CAP best point estimate in ", ToString[bestCells], "/",
   ToString[Length[rows]], " cells; significant for CAP: ",
   ToString[sigFor], ", against: ", ToString[sigAgainst], ", n/a: ",
   ToString[na], " (of ", ToString[2*Length[rows]], " comparisons)."]];

(* Every table carries its own legend: a trailing line in markdown, a LaTeX
   comment after the tabular so it survives copy-paste into the paper. *)
LatexTableFromRows[rows_List, digits_:2] := StringJoin[
   "\\begin{tabular}{lcccccccc}\n\\toprule\n",
   "Arch. & Semi & CAP & Stable & W1 & CAP$-$Stable & $p$ & CAP$-$W1 & $p$ \\\\\n",
   "\\midrule\n",
   StringRiffle[LatexRowCompactAsymCollapse[#, digits] & /@ rows, "\n"],
   "\n\\bottomrule\n\\end{tabular}\n% legend: ", tableLegendText[]];
MarkdownTableFromRows[rows_List, digits_:2] := StringJoin[
   "| Arch. | Semi | CAP | Stable | W1 | CAP-Stable | p | CAP-W1 | p |\n",
   "|---|---|---|---|---|---|---|---|---|\n",
   StringRiffle[MarkdownRowCompactAsymCollapse[#, digits] & /@ rows, "\n"],
   "\n\nlegend: ", tableLegendText[]];

(* Spearman cells use the same vocabulary: a strategy with no front at all is
   a no-pick; an undefined rho (n < 3, or tied/zero-variance effs) is n/a. *)
spearmanCellString[fronts_Association, strat_String] :=
  Module[{front, usable, rho},
   If[!KeyExistsQ[fronts, strat], Return[$markNoPick]];
   front = fronts[strat];
   usable = Select[front, !MissingQ[#["q0"]] && !MissingQ[#["eff"]] &];
   rho = frontSpearman[front, strat];
   StringJoin[If[NumericQ[rho], fmtNum[rho, 2], $markNA], " (",
    ToString[Length[usable]], ")"]];

(* The quantile-axis pair used a bare "-" for its p-cell; use the shared
   not-computable marker so no ambiguous dash survives anywhere. *)
axisPair[row_Association, dkey_String, pkey_String, latexQ_] := Module[
  {d = Lookup[row, dkey], p = Lookup[row, pkey], star},
  If[!NumericQ[d] || !NumericQ[p], Return[{$markNA, $markNA}]];
  star = If[p < 0.05, If[latexQ, "$^{*}$", "*"], ""];
  {fmtNum[d, 2], FormatP[p] <> star}];
