#!/usr/bin/env wolframscript
(* Compare two trees of exported notebook artifacts (pdf/svg/jpeg/...).

   Usage:
     wolframscript -f scripts/nb_compare.wls <dirA> <dirB> <failDir>

   Three tiers:
     1. inventory equality (same relative file lists, else exit 1);
     2. per-file byte hash (identical bytes -> OK);
     3. for hash mismatches, rasterize both sides at 72 dpi and require
        RootMeanSquare ImageDistance < 1*^-3 at identical dimensions.
   Failing pairs get an ImageDifference PNG written to <failDir>.
   Exits nonzero if anything differs. *)

args = Rest[$ScriptCommandLine];
If[Length[args] < 3,
  Print["usage: nb_compare.wls <dirA> <dirB> <failDir>"]; Exit[2]];
dirA = AbsoluteFileName[args[[1]]];
dirB = AbsoluteFileName[args[[2]]];
failDir = args[[3]];
If[dirA === $Failed || dirB === $Failed,
  Print["nb_compare: missing input dir"]; Exit[2]];

rel[d_] := Sort[FileNameDrop[#, FileNameDepth[d]] & /@
    Select[FileNames["*", d, Infinity], ! DirectoryQ[#] &]];

inv = {rel[dirA], rel[dirB]};
If[inv[[1]] =!= inv[[2]],
  Print["INVENTORY MISMATCH"];
  Print["only in A: ", Complement[inv[[1]], inv[[2]]]];
  Print["only in B: ", Complement[inv[[2]], inv[[1]]]];
  Exit[1]];

imp[f_] := Switch[ToLowerCase[FileExtension[f]],
  "pdf", First[Import[f, "PageGraphics"]],
  _, Import[f]];

bad = 0;
Do[Module[{a = FileNameJoin[{dirA, f}], b = FileNameJoin[{dirB, f}], ia, ib, d},
   If[FileHash[a] === FileHash[b],
    Print["OK  (bytes)  ", f],
    UsingFrontEnd[
     ia = Rasterize[imp[a], ImageResolution -> 72];
     ib = Rasterize[imp[b], ImageResolution -> 72]];
    Which[
     ImageDimensions[ia] =!= ImageDimensions[ib],
     bad++; Print["FAIL (dims)  ", f, " ",
       ImageDimensions[ia], " vs ", ImageDimensions[ib]],
     (d = ImageDistance[ia, ib, DistanceFunction -> "RootMeanSquare"]) < 1*^-3,
     Print["OK  (raster ", d, ")  ", f],
     True,
     bad++; Print["FAIL (raster ", d, ")  ", f];
     Quiet[CreateDirectory[failDir]];
     UsingFrontEnd[Export[
       FileNameJoin[{failDir, StringReplace[f, "/" -> "__"] <> ".diff.png"}],
       ImageAdjust@ImageDifference[ia, ib]]]]]],
  {f, inv[[1]]}];

If[bad > 0,
  Print[bad, " file(s) differ"]; Exit[1],
  Print["ALL MATCH (", Length[inv[[1]]], " files)"]];
