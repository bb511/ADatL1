(* ::Package:: *)

(* effs_rules.wl -- part of the notebooks/lib shared library for the
   effs_*.nb model-picking notebooks. Loaded with Get[]; all
   symbols live in Global`.

   Pareto-point selection rules. Every rule shares the signature
   pickF[front, strat, model] and returns a run name or Missing. *)

ClearAll[argBest, oraclePick, bestQPick, kneePick, frontSpearman, rulePickArrays, originalPickF];

(* argmax/argmin over name-sorted candidates: first best wins (deterministic,
   mirrors the reference implementation). *)
argBest[vals_Association, maximize_] := Module[{names, best},
  names = Sort[Keys[vals]];
  best = If[maximize, Max[Values[vals]], Min[Values[vals]]];
  SelectFirst[names, vals[#] == best &]];

oraclePick[front_Association] :=
  argBest[DeleteMissing[#["eff"] & /@ front], True];

(* Pick functions share the signature pickF[front, strat, model]. *)
bestQPick[front_Association, strat_String, model_:None] := Module[{q},
  q = Select[front, !MissingQ[#["q0"]] && !MissingQ[#["eff"]] &];
  If[Length[q] == 0, Missing["NoData"],
   argBest[#["q0"] & /@ q, strategyMaximizeQ[strat]]]];

(* Knee: goodness-orient both recomputed objectives over ALL retrained front
   points (they constitute the search front), min-max normalise, take the
   point of maximal distance from the chord joining the endpoints. *)
kneePick[front_Association, strat_String, model_:None] := Module[
  {maximize0 = strategyMaximizeQ[strat], usable, g, distinct, lo0, hi0,
   lo1, hi1, norm, order, e1, e2, chord, best = Missing["Degenerate"],
   bestD = -1., d},
  usable = Select[front,
    !MissingQ[#["q0"]] && !MissingQ[#["q1"]] && !MissingQ[#["eff"]] &];
  If[Length[usable] < 3, Return[Missing["Degenerate"]]];
  g = KeyValueMap[{#1, If[maximize0, #2["q0"], -#2["q0"]], -#2["q1"]} &,
    KeySort[usable]];
  distinct = DeleteDuplicates[g[[All, 2 ;; 3]]];
  If[Length[distinct] < 3, Return[Missing["Degenerate"]]];
  lo0 = Min[g[[All, 2]]]; hi0 = Max[g[[All, 2]]];
  lo1 = Min[g[[All, 3]]]; hi1 = Max[g[[All, 3]]];
  If[hi0 == lo0 || hi1 == lo1, Return[Missing["Degenerate"]]];
  norm = {#[[1]], (#[[2]] - lo0)/(hi0 - lo0), (#[[3]] - lo1)/(hi1 - lo1)} & /@
    g;
  order = SortBy[norm, {#[[2]], #[[3]]} &];
  e1 = order[[1, 2 ;; 3]]; e2 = order[[-1, 2 ;; 3]];
  chord = Norm[e2 - e1];
  If[chord == 0, Return[Missing["Degenerate"]]];
  Do[
   d = Abs[(e2[[1]] - e1[[1]])*(e1[[2]] - p[[3]]) -
       (e1[[1]] - p[[2]])*(e2[[2]] - e1[[2]])]/chord;
   If[d > bestD, bestD = d; best = p[[1]]],
   {p, order[[2 ;; -2]]}];
  best];

(* Spearman rho between goodness-oriented Q' and the mean test efficiency. *)
frontSpearman[front_Association, strat_String] := Module[{usable, qs, effs, rho},
  usable = KeySort[Select[front, !MissingQ[#["q0"]] && !MissingQ[#["eff"]] &]];
  If[Length[usable] < 3, Return[Missing["TooFew"]]];
  qs = If[strategyMaximizeQ[strat], #, -#] &[Values[#["q0"] & /@ usable]];
  effs = Values[#["eff"] & /@ usable];
  rho = Quiet[Check[N[SpearmanRho[qs, effs]], Missing["Undefined"]]];
  If[NumericQ[rho], rho, Missing["Undefined"]]];

(* Per-signal arrays (percent) of a rule's picks, for the grouped box plot:
   ordered assoc model -> strategy -> list. *)
rulePickArrays[modelExps_Association, pickF_] := Association @@ Map[
   Function[modelKey, modelKey -> Association @@ DeleteMissing[Map[
       Function[strat, Module[{fronts, pick},
         fronts = frontsOf[rebuttalRuns[modelExps[modelKey]]];
         If[!KeyExistsQ[fronts, strat], Missing[],
          pick = pickF[fronts[strat], strat, modelKey];
          If[!StringQ[pick] || MissingQ[fronts[strat][pick]["eff"]], Missing[],
           strat -> 100.*Values[fronts[strat][pick]["testArr"]]]]]],
       boxStrategyOrder]]],
   Keys[modelExps]];

(* Original-pick selector usable as pickF (closes over domain). *)
originalPickF[domain_String] := Function[{front, strat, model},
   Module[{trial, name},
    trial = Query[ToLowerCase[model], strat][
      Lookup[originalPicks, domain, <||>]];
    name = If[IntegerQ[trial], strat <> "_t" <> ToString[trial],
      Missing["NA"]];
    If[StringQ[name] && KeyExistsQ[front, name], name, Missing["NA"]]]];
