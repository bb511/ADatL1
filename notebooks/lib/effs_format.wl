(* ::Package:: *)

(* effs_format.wl -- part of the notebooks/lib shared library for the
   effs_*.nb model-picking notebooks. Loaded with Get[]; all
   symbols live in Global`.

   Number/cell formatting and the dash-marker vocabulary. The
   fire-on-everything guard reads $degenerateGuard at call time;
   that flag is per-notebook configuration and is set (with its
   domain-specific rationale) in each notebook, not here. *)

ClearAll[FormatNum, fmtNum, fmtTrial, degenerateArrQ, degenerateEffQ, $markNoPick, $markColl, $markNA, noPickQ, zeroCollapsedQ, CollapsedMethodQ, FormatMeanOrDash, FormatAsymPMOrDash, FormatP, fmtEffPct, fmtDelta, tableLegendText, printTable];

FormatNum[x_, digits_:2] := ToString[NumberForm[x, {Infinity, digits}, NumberPadding -> {"", "0"}, NumberPoint -> "."]]; 

fmtNum[x_, digits_:2] := If[NumericQ[x],
   ToString[NumberForm[N[x], {Infinity, digits}, NumberPadding -> {"", "0"},
     NumberPoint -> "."]], "-"];
fmtTrial[name_] := If[StringQ[name],
   Last[StringSplit[name, "_t"]], "-"];

degenerateArrQ[x_List] := TrueQ[$degenerateGuard] && Length[x] > 0 &&
   Min[x] >= 99.;  (* percent scale *)
degenerateEffQ[e_] := TrueQ[$degenerateGuard] && NumericQ[e] &&
   e >= 0.99;  (* fraction scale *)

(* Dash legend. A bare "-" previously stood for three different situations.
   Give each its own marker so a reader can tell them apart:
     no-pick  the rule could not select a point from this front (kneePick
              needs >= 3 usable/distinct Pareto points; bestQPick needs >= 1)
     coll     method collapsed (all-zero) or excluded by the saturation guard
     n/a      comparison not computable (a required side is missing)
   toLegacyGroups fills a strategy the rule could not pick with the sentinel
   -1 rather than 0, so no-pick stays distinguishable from a genuine collapse
   while the downstream statistics are suppressed either way. *)
$markNoPick = "no-pick";
$markColl = "coll";
$markNA = "n/a";
noPickQ[x_List] := Length[x] > 0 && AllTrue[x, # == -1. &];
zeroCollapsedQ[x_List] := AllTrue[x, # == 0. &];
CollapsedMethodQ[x_List] :=
   zeroCollapsedQ[x] || degenerateArrQ[x] || noPickQ[x];
FormatMeanOrDash[x_List, digits_:2] := Which[
   noPickQ[x], $markNoPick,
   degenerateArrQ[x] || zeroCollapsedQ[x], $markColl,
   True, FormatNum[Mean[x], digits]];
FormatAsymPMOrDash[center_, ci_, digits_:2] :=
  If[MissingQ[center] || MissingQ[ci], $markNA,
   Module[{lower = center - ci[[1]], upper = ci[[2]] - center},
    "$" <> FormatNum[center, digits] <> "^{+" <> FormatNum[upper, digits] <>
     "}_{-" <> FormatNum[lower, digits] <> "}$"]];
FormatP[p_] := Which[MissingQ[p], $markNA, p < 0.001, "<0.001", True,
   ToString[NumberForm[p, {Infinity, 3}, NumberPadding -> {"", "0"},
     NumberPoint -> "."]]];
fmtEffPct[x_] := Which[!NumericQ[x], $markNA,
   degenerateEffQ[x], $markColl, True, fmtNum[100*x, 2]];
fmtDelta[x_, ref_] := Which[!NumericQ[x] || !NumericQ[ref], $markNA,
   degenerateEffQ[x] || degenerateEffQ[ref], $markColl,
   True, fmtNum[100*(x - ref), 2]];

tableLegendText[] := StringJoin[
   $markNoPick, " = rule could not select from this front (knee needs >= 3 ",
   "Pareto points, best-Q' needs >= 1); ", $markColl,
   " = method collapsed (all-zero) or excluded by the saturation guard; ",
   $markNA, " = comparison not computable (a required side is missing)."];

(* Long Print output is wrapped by the front end, which inserts a literal "\"
   line-continuation mid-row. That splits every markdown row across two physical
   lines and corrupts the table on copy-paste. Emit tables in a cell with
   PageWidth -> Infinity so each row stays on one line. Falls back to plain
   Print when there is no front end (headless wolframscript validation). *)
printTable[s_String] := If[$Notebooks =!= True, Print[s],
   CellPrint[Cell[s, "Print", PageWidth -> Infinity,
     ShowStringCharacters -> False]]];
printTable[x_] := Print[x];
