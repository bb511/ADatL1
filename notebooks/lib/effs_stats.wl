(* ::Package:: *)

(* effs_stats.wl -- part of the notebooks/lib shared library for the
   effs_*.nb model-picking notebooks. Loaded with Get[]; all
   symbols live in Global`.

   Statistics for the comparison tables: bootstrap CIs on paired
   differences, exact Wilcoxon signed-rank p-values, and the
   Holm-Bonferroni step-down correction over a table's family. *)

ClearAll[BootstrapPairedDiffCI, WilcoxonSignedRankPExact, SafeWilcoxonP, HolmAdjustRows];

BootstrapPairedDiffCI[a_List, b_List, nBoot_:10000, alpha_:0.05] := Module[{n, diffs, boots}, If[Length[a] =!= Length[b], Return[Missing["LengthMismatch"]]]; n = Length[a]; diffs = a - b; boots = Table[Mean[RandomChoice[diffs, n]], {nBoot}]; Quantile[boots, {alpha/2, 1 - alpha/2}]]; 

WilcoxonSignedRankPExact[a_List, b_List] := Module[{d, absd, sgn, n, order, sortedAbs, sortedIdx,
    ranks, groups, start, wPlus, allSums, pLeft, pRight}, If[Length[a] =!= Length[b], Return[Missing["LengthMismatch"]]];
  d = DeleteCases[a - b, 0.]; If[d === {}, Return[1.]]; absd = Abs[d]; sgn = Sign[d]; n = Length[d];
  order = Ordering[absd]; sortedAbs = absd[[order]]; sortedIdx = order; ranks = ConstantArray[0.,
    n]; groups = Split[Range[n], sortedAbs[[#1]] == sortedAbs[[#2]] & ]; start = 1; Do[Module[{len = Length[g],
        mid}, mid = Mean[Range[start, start + len - 1]]; ranks[[sortedIdx[[g]]]] = mid; start += len;
      ], {g, groups}]; wPlus = Total[Pick[ranks, sgn, 1]]; allSums = Total /@ (Pick[ranks, #1,
      1] & ) /@ Tuples[{0, 1}, n]; pLeft = N[Count[allSums, x_ /; x <= wPlus]/Length[allSums]];
  pRight = N[Count[allSums, x_ /; x >= wPlus]/Length[allSums]]; Min[1., 2.*Min[pLeft, pRight]]];


SafeWilcoxonP[a_List, b_List] := WilcoxonSignedRankPExact[a, b]; 

(* Holm-Bonferroni step-down correction applied jointly to all Wilcoxon
   p-values of one table -- the benchmark's family of comparisons (3 tests
   per architecture row since consistency joined; 18 for physics, 12 for
   cifar10/robustad). Only NumericQ p-values enter the family, so a collapsed
   or absent strategy does not inflate m. Raw p-values are kept under
   <key>Raw. *)
HolmAdjustRows[rows_List] := Module[
  {keys = {"CAPMinusConsP", "CAPMinusStableP", "CAPMinusW1P"}, idx = {}, ps, m, order,
   run = 0., newRows = rows, i, k, adj},
  Do[If[NumericQ[Lookup[newRows[[a]], b]], AppendTo[idx, {a, b}]],
   {a, Length[newRows]}, {b, keys}];
  m = Length[idx];
  If[m == 0, Return[newRows]];
  ps = Lookup[newRows[[#[[1]]]], #[[2]]] & /@ idx;
  order = Ordering[ps];
  Do[
   {i, k} = idx[[order[[j]]]];
   adj = Min[1., (m - j + 1)*ps[[order[[j]]]]];
   run = Max[run, adj];
   newRows[[i]] = Join[newRows[[i]],
     Association[k <> "Raw" -> ps[[order[[j]]]], k -> run]],
   {j, m}];
  newRows];
