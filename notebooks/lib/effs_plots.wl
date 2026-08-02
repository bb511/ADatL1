(* ::Package:: *)

(* effs_plots.wl -- part of the notebooks/lib shared library for the
   effs_*.nb model-picking notebooks. Loaded with Get[]; all
   symbols live in Global`.

   Publication plots: the CMS-style box-whisker panel (with its
   pixel-fitted 1300 pt layout and Magnify-based grid) and the
   Pareto-front scatter plots. *)

ClearAll[makeBoxCMS, makeRuleBoxGrid, secObjLabel, studyFile, makeFrontPlot, modelFrontRow];

(* --- rule box-plot grid ------------------------------------------------ *)

(* Legacy per-model box plot (verbatim from the Plotting Method section of
   effs_physics.nb) + a grid assembling one panel per model for a given
   picking rule. Strategies absent under a rule (degenerate fronts) are
   drawn as collapsed (all-zero) boxes, the legacy convention. *)
hexColors = {"#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
   "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"};
makeBoxCMS[data_, labels_, yLabel_, cmsPos_, tevPos_, precY_:1, xLabelY_:0.06, modelLabel_:"AE",
  modelLabelPos_:{0.639, 0.899}] := Module[{majorLen, minorLen, makeTicks01, yVals, yTicks,
    yTicksRight, plot, cmsText, tevText, plotLeft = 0.01, plotBottom = 0.12, plotWidth = 0.98,
    plotHeight = 0.84, xCenters}, majorLen = Scaled[0.015]; minorLen = Scaled[0.015/2]; makeTicks01[{(vmin_)?NumericQ,
      (vmax_)?NumericQ}, prec_] := Module[{maj, min}, maj = N[FindDivisions[{vmin, vmax}, 6]];
    min = N[Flatten[Table[Most[Rest[Subdivide[maj[[k]], maj[[k + 1]], 6]]], {k, 1, Length[maj] - 1}]]];
    Join[({#1, NumberForm[#1, {6, prec}], {majorLen, 0}} & ) /@ maj, ({#1, "", {minorLen, 0}} & ) /@ min]];
  yVals = Flatten[data]; yTicks = makeTicks01[{0, 1.05*Max[yVals]}, precY]; yTicksRight = yTicks /. {v_,
    _, len_} :> {v, "", len}; nBoxes = Length[data]; q10 = (Quantile[#1, 0.1] & ) /@ data; q90 = (Quantile[#1,
      0.9] & ) /@ data; xPos = Range[nBoxes]; p10p90Graphics = Table[{EdgeForm[None], FaceForm[Directive[RGBColor["#92dadd"],
        Opacity[0.22]]], Rectangle[{xPos[[i]] - 0.39, q10[[i]]}, {xPos[[i]] + 0.39, q90[[i]]}]},
    {i, nBoxes}]; plot = BoxWhiskerChart[data, {"Outliers", {"MedianMarker", 1., Directive[RGBColor["#94a4a2"],
        AbsoluteThickness[3], Opacity[2]]}, {"Whiskers", Directive[Black, AbsoluteThickness[3]]},
      {"Fences", 0.5, Directive[Black, AbsoluteThickness[2]]}, {"MeanMarker", 0.78, Directive[Black,
        Opacity[1], AbsoluteThickness[3]]}, {"MeanDiamond", 0.8, Directive[RGBColor["#92dadd"],
        Opacity[0.4]]}, {"Outliers", Graphics[{RGBColor["#3f90da"], Rectangle[{-1, -1}, {1,
        1}]}]}}, Prolog -> p10p90Graphics, BarSpacing -> 0.3, ChartStyle -> Table[Directive[RGBColor["#3f90da"],
        Opacity[0.8]], {Length[data]}], Frame -> True, Axes -> False, FrameTicks -> {{yTicks,
        yTicksRight}, {Automatic, Automatic}}, FrameStyle -> Directive[Black, AbsoluteThickness[2]],
    FrameTicksStyle -> Directive[Black, FontSize -> 22, FontFamily -> "TeX Gyre Heros"], LabelStyle -> Directive[Black,
      FontSize -> 24, FontFamily -> "TeX Gyre Heros"], FrameLabel -> {None, Style[yLabel, Black,
        FontSize -> 30, FontFamily -> "TeX Gyre Heros"]}, GridLines -> {None, Automatic}, GridLinesStyle -> Directive[GrayLevel[0.85],
      AbsoluteThickness[1.5]], ImageSize -> 1300, AspectRatio -> 0.65, ImagePadding -> {{100,
        40}, {40, 40}}, PlotRange -> {0, 1.05*Max[yVals]}, PlotRangePadding -> {{Scaled[-0.1],
        Scaled[-0.03]}, {Scaled[0.03], Scaled[0.01]}}]; cmsText = Row[{Style["CMS", Bold, FontFamily -> "TeX Gyre Heros",
        24], Style[" Simulation Preliminary", Italic, FontFamily -> "TeX Gyre Heros", 24]}];
  tevText = Style["14 TeV", FontFamily -> "TeX Gyre Heros", 24]; xCenters = If[nBoxes == 4,
    {0.165, 0.3, 0.43, 0.57}, Table[0.0975 + (i - 0.5)/nBoxes*0.54, {i, nBoxes}]]; Graphics[Join[{Inset[plot,
        Scaled[{plotLeft, plotBottom}], {Left, Bottom}, Scaled[{plotWidth, plotHeight}]]}, Table[Inset[Style[labels[[i]],
        FontFamily -> "TeX Gyre Heros", 22], Scaled[{xCenters[[i]], xLabelY}], {Center, Top}],
        {i, Length[labels]}], {Inset[cmsText, Scaled[cmsPos], {Left, Bottom}], Inset[tevText,
        Scaled[tevPos], {Right, Bottom}], Inset[Framed[Style[modelLabel, FontFamily -> "TeX Gyre Heros",
        Italic, 32, Black], Background -> White, FrameStyle -> Directive[Black, AbsoluteThickness[1.5]],
        RoundingRadius -> 0, FrameMargins -> {{8, 8}, {4, 4}}], Scaled[modelLabelPos], {Right,
        Top}]}], PlotRange -> {{0, 1}, {0, 1}}, AspectRatio -> 0.5, ImageSize -> 1300]]

makeRuleBoxGrid[groups_Association, ncols_Integer, mag_:0.55] := Module[{panels},
  panels = KeyValueMap[Function[{modelLabel, sub}, Module[{len, data},
      If[Length[sub] == 0, Nothing,
       len = Length[First[Values[sub]]];
       data = Table[If[KeyExistsQ[sub, s], sub[s],
          ConstantArray[0., len]], {s, boxStrategyOrder}];
       (* Labels are derived from strategyPretty in boxStrategyOrder rather
          than hardcoded, so box order and label order cannot drift apart
          when a strategy is added. *)
       makeBoxCMS[data,
        strategyPretty[#] & /@ boxStrategyOrder,
        "Efficiency (%)", {0.085, 0.9}, {0.64, 0.9}, 1, 0.17,
        modelLabel]]]], groups];
  (* Panels are built at full legacy scale (1300 pt each) so fonts and label
     positions keep the exact original proportions; Magnify then shrinks the
     assembled grid uniformly. *)
  Magnify[GraphicsGrid[Partition[panels, UpTo[ncols]],
    ImageSize -> {1300*ncols, Automatic},
    Spacings -> {Scaled[0.01], Scaled[0.01]}], mag]];

(* --- optimisation-front plot --------------------------------------------- *)

secObjLabel = <|"mse" -> "MSE", "mseq99" -> "MSE", "kl" -> "KL",
   "dist" -> "dist", "logp" -> "-log p", "ascore" -> "anomaly score",
   "ascoreq99" -> "anomaly score"|>;

studyFile[frontsDir_String, model_String, strat_String] := Module[{cands},
  cands = Select[FileNames[
     model <> "_" <> strategyStudyObj[strat] <> "_vs_*.csv", frontsDir],
    !StringContainsQ[#, "q99"] && !StringContainsQ[#, "exploration"] &];
  If[cands === {}, Missing["NoStudy"], First[Sort[cands]]]];

makeFrontPlot[frontsDir_String, model_String, strat_String,
   modelLabel_String] := Module[
  {file, raw, hdr, idx, rows, pts, front, dom, secName, xlab, ylab},
  file = studyFile[frontsDir, model, strat];
  If[MissingQ[file], Return[Missing["NoStudy"]]];
  raw = Import[file, "CSV"];
  hdr = First[raw]; rows = Rest[raw];
  idx = AssociationThread[hdr -> Range[Length[hdr]]];
  pts = Select[rows, NumericQ[#[[idx["values_0"]]]] &&
      NumericQ[#[[idx["values_1"]]]] &];
  front = Select[pts, ToString[#[[idx["is_pareto"]]]] === "True" &];
  dom = Select[pts, ToString[#[[idx["is_pareto"]]]] =!= "True" &];
  secName = StringReplace[FileBaseName[file],
    {model <> "_" <> strategyStudyObj[strat] <> "_vs_" -> "", "_b16k" -> ""}];
  xlab = strategyPretty[strat] <> " objective";
  ylab = Lookup[secObjLabel, secName, secName];
  ListPlot[
   {dom[[All, {idx["values_0"], idx["values_1"]}]],
    front[[All, {idx["values_0"], idx["values_1"]}]]},
   PlotStyle -> {Directive[GrayLevel[0.75], PointSize[0.008]],
     Directive[RGBColor["#bd1f01"], PointSize[0.014]]},
   Frame -> True, Axes -> False,
   FrameStyle -> Directive[Black, AbsoluteThickness[1.5]],
   FrameTicksStyle -> Directive[Black, FontSize -> 16,
     FontFamily -> "TeX Gyre Heros"],
   FrameLabel -> {Style[xlab, 18, FontFamily -> "TeX Gyre Heros"],
     Style[ylab, 18, FontFamily -> "TeX Gyre Heros"]},
   PlotLabel -> Style[modelLabel <> ": " <> strategyPretty[strat], 18,
     FontFamily -> "TeX Gyre Heros"],
   ImageSize -> 420, AspectRatio -> 0.8,
   PlotRangePadding -> Scaled[0.05]]];

modelFrontRow[frontsDir_String, model_String, modelLabel_String] :=
  Module[{plots},
   plots = DeleteMissing[
     makeFrontPlot[frontsDir, model, #, modelLabel] & /@
      {"cvar25", "stability", "cap", "consistency", "wasserstein"}];
   If[plots === {}, Missing["NoStudies"], GraphicsRow[plots,
     ImageSize -> 420*Length[plots], Spacings -> 0]]];
