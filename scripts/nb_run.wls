#!/usr/bin/env wolframscript
(* Headless Mathematica notebook runner.

   Usage:
     wolframscript -f scripts/nb_run.wls <notebook.nb> <log.txt> [<symbol> <out.m>]

   Evaluates every evaluatable cell of the notebook top-to-bottom in this
   kernel with the front end attached (so NotebookDirectory[], CellPrint and
   friends behave exactly as in an interactive "Evaluate Notebook"). The
   notebook is never saved. All generated Output/Print/Message cells are
   written to <log.txt> so two runs can be diffed. Optionally, a named
   Global` symbol (e.g. corrs) is Put to <out.m> after evaluation.

   The RNG is seeded so bootstrap confidence intervals are reproducible
   across runs. *)

args = Rest[$ScriptCommandLine];
If[Length[args] < 2,
  Print["usage: nb_run.wls <notebook.nb> <log.txt> [<symbol> <out.m>]"];
  Exit[2]];

nbPath = AbsoluteFileName[args[[1]]];
logPath = args[[2]];
If[nbPath === $Failed,
  Print["nb_run: no such notebook: ", args[[1]]]; Exit[2]];

SeedRandom[20260802];

UsingFrontEnd[Module[{nb, cells},
  nb = NotebookOpen[nbPath, Visible -> False];
  If[Head[nb] =!= NotebookObject,
    Print["nb_run: NotebookOpen failed for ", nbPath]; Exit[1]];
  NotebookEvaluate[nb, InsertResults -> True];
  cells = Cases[NotebookRead /@ Cells[nb],
    Cell[_, "Output" | "Print" | "Message" | "Echo", ___]];
  (* Strip volatile metadata so logs from two runs diff cleanly. *)
  cells = cells /. Cell[c_, style_String, opts___] :>
     Cell[c, style, Sequence @@ DeleteCases[{opts},
       (Rule | RuleDelayed)[
         CellChangeTimes | CellLabel | ExpressionUUID, _]]];
  Export[logPath,
    StringRiffle[ToString[#, InputForm] & /@ cells, "\n(* ---- *)\n"],
    "String"];
  NotebookClose[nb]]];

If[Length[args] >= 4,
  Put[Symbol["Global`" <> args[[3]]], args[[4]]]];

Print["nb_run done: ", nbPath];
