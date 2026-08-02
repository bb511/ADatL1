(* ::Package:: *)

(* pareto_cap_lib.wl -- part of the notebooks/lib shared library, for
   paretos_cap.nb (Pareto fronts of the CAP optimisation sweeps).
   Loaded with Get[]; all symbols live in Global`.

   loadParetoData ingests the fetch_optuna_pareto.py CSVs (relative
   dir default: the notebook SetDirectory's to its own folder);
   makeParetoPlot builds the CMS-styled Pareto plots; exportParetoPlot
   writes the image files. Bodies are verbatim from the notebook.

   makeParetoPlot's 16 trailing positional parameters became string
   options; each default is the modal value over the notebook's 24
   call sites. The old parameter names (xLabel, zoomRange,
   zoomedPlotSize, ...) are reused unchanged as Module locals
   initialised from OptionValue, so the body needed zero edits. This
   is safe: the notebook defines no top-level helper functions named
   zoomRange or zoomedPlotSize (they exist only as per-section global
   variables passed positionally), and inside the body both are only
   ever used as values (zoomRange[[1]], PlotRange -> zoomRange,
   zoomedPlotSize[[2]], ...), never as function calls -- so same-name
   locals reproduce the old pattern substitution exactly.
   makeTicks01/makeLogTicks/paddedZoom/mainPlotFn stay Module-local as
   before. paretoAssoc, dominatedAssoc, trialToPoint and markedPts,
   formerly leaked globals of the body, are now Module locals; nothing
   outside makeParetoPlot reads them. The body's other leaked globals
   (xmin, xmax, ymin, ymax, xPadLeft, xPadRight, yPadBottom, yPadTop,
   imgPadActual, frameThick, linePrimitives) are left as-is to keep
   the body byte-identical. padRange is kept although nothing calls
   it. *)

ClearAll[loadParetoData, padRange, makeParetoPlot, exportParetoPlot];

(*Loads data from specified path. Needs to be .csv files, produced \
with fetch_optuna_pareto.py from the optuna .db file.*)
loadParetoData[dataName_, dir_ : "paretos/cap_optimisation"] :=
 Module[
  {data, headers, rows, col, toNum, toBool, isFiniteReal, 
   completeRows, trialNumsRaw, trialNums, rawPts, ptsAll, paretoRaw, 
   paretoMask, goodMask, pts, paretoPts, dominatedPts},
  data = Import[FileNameJoin[{dir, dataName <> ".csv"}]];
  headers = First[data];
  rows = Rest[data];
  col[name_] := First@FirstPosition[headers, name];
  toNum[x_?NumericQ] := N@x;
  toNum[s_String] := 
   Module[{t = ToLowerCase@StringTrim[s]}, 
    Which[t == "" || t == "nan" || t == "none" || t == "null", 
     Missing["NotANumber"], 
     t == "inf" || t == "+inf" || t == "infinity", Infinity, 
     t == "-inf" || t == "-infinity", -Infinity, True, 
     Quiet@Check[N@ToExpression[StringTrim[s]], 
       Missing["NotANumber"]]]];
  toNum[_] := Missing["NotANumber"];
  toBool[x_?BooleanQ] := x;
  toBool[s_String] := ToLowerCase[StringTrim[s]] === "true";
  toBool[_] := False;
  isFiniteReal[x_] := 
   NumericQ[x] && x =!= Indeterminate && x =!= ComplexInfinity && 
    x =!= Infinity && x =!= -Infinity;
  completeRows = Select[rows, #[[col["state"]]] == "COMPLETE" &];
  trialNumsRaw = completeRows[[All, col["number"]]];
  trialNums = 
   trialNumsRaw /. 
    s_String :> Quiet@Check[ToExpression[StringTrim[s]], s];
  rawPts = completeRows[[All, {col["values_0"], col["values_1"]}]];
  ptsAll = Map[toNum, rawPts, {2}];
  paretoRaw = completeRows[[All, col["is_pareto"]]];
  paretoMask = toBool /@ paretoRaw;
  goodMask = 
   MapThread[
    MatchQ[#1, _Integer | _Real] && 
      MatchQ[#2, {a_, b_} /; 
        isFiniteReal[a] && isFiniteReal[b]] &, {trialNums, ptsAll}];
  trialNums = Pick[trialNums, goodMask];
  pts = Pick[ptsAll, goodMask];
  paretoMask = Pick[paretoMask, goodMask];
  paretoPts = Pick[pts, paretoMask, True];
  dominatedPts = Pick[pts, paretoMask, False];
  <|"trialNums" -> trialNums, "pts" -> pts, 
   "paretoMask" -> paretoMask, "paretoPts" -> paretoPts, 
   "dominatedPts" -> dominatedPts, 
   "paretoTrialNums" -> Pick[trialNums, paretoMask, True], 
   "dominatedTrialNums" -> Pick[trialNums, paretoMask, False]|>
  ]

(*Helper method to pad the range of the plot.*)
padRange[{{xmin_, xmax_}, {ymin_, ymax_}}, frac_ : 0.05] :=
 Module[
  {dx, dy}, dx = (xmax - xmin)*frac;
  dy = (ymax - ymin)*frac;
  {{xmin - dx, xmax + dx}, {ymin - dy, ymax + dy}}
  ]

(*Pareto plot method that takes the data produced with the above \
method and makes the plot.*)

Options[makeParetoPlot] = {
   "XLabel" -> "CAP(\!\(\*StyleBox[\"X\", FontWeight->\"Bold\", FontSlant->\"Plain\"]\), \!\(\*SubscriptBox[StyleBox[\"X\", FontWeight->\"Bold\", FontSlant->\"Plain\"], \(sim\)]\))",
   "YLabel" -> "\!\(\*SubscriptBox[\(MSE\), \(250  Hz\)]\)",
   "ZoomRange" -> {{-0.197, -0.165}, {0.15, 0.5}},
   "ZoomedPlotPos" -> {0.35, 0.33},
   "ZoomedPlotSize" -> {0.5, 0.63},
   "RangePadding" -> {{Scaled[0.055], Scaled[0.08]}, {Scaled[0.055],
       Scaled[0.05]}},
   "LegendPos" -> {0.05, 0.1},
   "CMSPos" -> {0.09, 0.93},
   "TeVPos" -> {0.815, 0.925},
   "PrecX" -> 2,
   "PrecY" -> 2,
   "ShowZoom" -> True,
   "ShowZoomBox" -> True,
   "LogLog" -> False,
   "InvertLines" -> False,
   "HighlightTrials" -> {}};

makeParetoPlot[data_Association, opts : OptionsPattern[]] :=
 Module[
  {xLabel = OptionValue["XLabel"], yLabel = OptionValue["YLabel"],
   zoomRange = OptionValue["ZoomRange"],
   zoomedPlotPos = OptionValue["ZoomedPlotPos"],
   zoomedPlotSize = OptionValue["ZoomedPlotSize"],
   rangePadding = OptionValue["RangePadding"],
   legendPos = OptionValue["LegendPos"],
   cmsPos = OptionValue["CMSPos"],
   tevPos = OptionValue["TeVPos"],
   precX = OptionValue["PrecX"], precY = OptionValue["PrecY"],
   showZoom = OptionValue["ShowZoom"],
   showZoomBox = OptionValue["ShowZoomBox"],
   logLog = OptionValue["LogLog"],
   invertLines = OptionValue["InvertLines"],
   highlightTrials = OptionValue["HighlightTrials"],
   pts, paretoPts, dominatedPts, xVals, yVals, majorLen, minorLen,
   makeTicks01, makeLogTicks, xTicks, yTicks, frameTicksSpec, plt,
   zoomPlt, zoomPad, zoomImageSize, paddedZoom, leftFrac, rightFrac,
   bottomFrac, topFrac, insetBL, insetTR, insetTL, insetBR, cmsText,
   tevText, outer, mainPlotFn, effectiveShowZoom,
   effectiveShowZoomBox, legend, epilogParts,
   paretoAssoc, dominatedAssoc, trialToPoint, markedPts},
  pts = data["pts"];
  
  paretoAssoc = 
   AssociationThread[data["paretoTrialNums"] -> data["paretoPts"]];
  dominatedAssoc = 
   AssociationThread[
    data["dominatedTrialNums"] -> data["dominatedPts"]];
  trialToPoint = Join[paretoAssoc, dominatedAssoc];
  markedPts = Lookup[trialToPoint, highlightTrials, {}]; 
  paretoPts = data["paretoPts"];
  dominatedPts = data["dominatedPts"];
  
  xVals = pts[[All, 1]];
  yVals = pts[[All, 2]];
  majorLen = Scaled[0.015];
  minorLen = Scaled[0.015/2];
  effectiveShowZoom = showZoom && ! logLog;
  effectiveShowZoomBox = showZoomBox && ! logLog;
  
  makeTicks01[{vmin_?NumericQ, vmax_?NumericQ}, prec_] := 
   Module[{maj, min}, maj = N@FindDivisions[{vmin, vmax}, 6];
    min = 
     N@Flatten@
       Table[Most@Rest@Subdivide[maj[[k]], maj[[k + 1]], 6], {k, 1, 
         Length[maj] - 1}];
    Join[({#, NumberForm[#, {Infinity, prec}], {majorLen, 0}} & /@ 
       maj), ({#, "", {minorLen, 0}} & /@ min)]];
  
  makeLogTicks[{vmin_?NumericQ, vmax_?NumericQ}] := 
   Module[{emin, emax, majors, minors}, 
    If[vmin <= 0 || vmax <= 0, Return[{}]];
    emin = Floor[Log10[vmin]];
    emax = Ceiling[Log10[vmax]];
    majors = Table[10.^e, {e, emin, emax}];
    minors = Flatten@Table[Table[m*10.^e, {m, 2, 9}], {e, emin, emax}];
    majors = Select[majors, vmin <= # <= vmax &];
    minors = Select[minors, vmin <= # <= vmax &];
    Join[({#, Superscript[10, Round[Log10[#]]], {majorLen, 0}} & /@ 
       majors), ({#, "", {minorLen, 0}} & /@ minors)]];
  
  xTicks = 
   If[logLog, makeLogTicks[{Min[xVals], Max[xVals]}], 
    makeTicks01[{Min[xVals], Max[xVals]}, precX]];
  yTicks = 
   If[logLog, makeLogTicks[{Min[yVals], Max[yVals]}], 
    makeTicks01[{Min[yVals], Max[yVals]}, precY]];
  
  frameTicksSpec = {{yTicks, (yTicks /. {v_, lab_, len_} :> {v, "", 
         len})}, {xTicks, (xTicks /. {v_, lab_, len_} :> {v, "", 
         len})}};
  
  {xmin, xmax} = zoomRange[[1]];
  {ymin, ymax} = zoomRange[[2]];
  
  xPadLeft = 0.10 (xmax - xmin);
  xPadRight = -0.01 (xmax - xmin);   (*smaller padding on the right*)
  yPadBottom = 0.10 (ymax - ymin);
  yPadTop = 0.10 (ymax - ymin);
  
  paddedZoom = {{xmin - xPadLeft, 
     xmax + xPadRight}, {ymin - yPadBottom, ymax + yPadTop}};
  zoomPad = {{55, 8}, {35, 8}};
  zoomImageSize = {500, 320};
  
  mainPlotFn = If[logLog, ListLogLogPlot, ListPlot];
  
  zoomPlt = 
   Show[mainPlotFn[dominatedPts, 
     PlotStyle -> 
      Directive[RGBColor["#3f90da"], AbsolutePointSize[6]]], 
    mainPlotFn[paretoPts, 
     PlotStyle -> 
      Directive[RGBColor["#bd1f01"], AbsolutePointSize[6]]], 
    Axes -> False, Frame -> True, PlotRange -> zoomRange, 
    PlotRangePadding -> None, ImagePadding -> zoomPad, 
    FrameTicksStyle -> Directive[Black, FontSize -> 14], 
    PlotRangeClipping -> True, ImageSize -> zoomImageSize];
  
  imgPadActual = 
   N[ImagePadding /. AbsoluteOptions[zoomPlt, ImagePadding]];
  frameThick = 
   FirstCase[FrameStyle /. AbsoluteOptions[zoomPlt, FrameStyle], 
    AbsoluteThickness[t_] :> t, 1.25, Infinity];
  insetBL = 
   Offset[{imgPadActual[[1, 1]] - frameThick/2, 
     imgPadActual[[2, 1]] - frameThick/2}, Scaled[zoomedPlotPos]];
  insetTR = 
   Offset[{-imgPadActual[[1, 2]] + 
      frameThick/2, -imgPadActual[[2, 2]] + frameThick/2}, 
    Scaled[zoomedPlotPos + zoomedPlotSize]];
  insetTL = 
   Offset[{imgPadActual[[1, 1]] - 
      frameThick/2, -imgPadActual[[2, 2]] + frameThick/2}, 
    Scaled[zoomedPlotPos + {0, zoomedPlotSize[[2]]}]];
  insetBR = 
   Offset[{-imgPadActual[[1, 2]] + frameThick/2, 
     imgPadActual[[2, 1]] - frameThick/2}, 
    Scaled[zoomedPlotPos + {zoomedPlotSize[[1]], 0}]];
  
  linePrimitives = 
   If[invertLines, {Line[{{paddedZoom[[1, 1]], paddedZoom[[2, 2]]}, 
       insetTL}], 
     Line[{{paddedZoom[[1, 2]], paddedZoom[[2, 1]]}, 
       insetBR}]}, {Line[{{paddedZoom[[1, 1]], paddedZoom[[2, 1]]}, 
       insetBL}], 
     Line[{{paddedZoom[[1, 2]], paddedZoom[[2, 2]]}, insetTR}]}];
  
  legend = Framed[
    Grid[{
      {Graphics[{RGBColor["#3f90da"], Disk[{0, 0}, 0.35]}, 
        ImageSize -> 12, PlotRange -> {{-0.6, 0.6}, {-0.6, 0.6}}], 
       Style["Normal Trial", FontFamily -> "TeX Gyre Heros", 
        14]}, {Graphics[{RGBColor["#ffa90e"], Disk[{0, 0}, 0.35]}, 
        ImageSize -> 12, PlotRange -> {{-0.6, 0.6}, {-0.6, 0.6}}], 
       Style["Pareto Trial", FontFamily -> "TeX Gyre Heros", 14]},
      {Graphics[{RGBColor["#bd1f01"], Disk[{0, 0}, 0.35]}, 
        ImageSize -> 12, PlotRange -> {{-0.6, 0.6}, {-0.6, 0.6}}], 
       Style["Chosen Trial", FontFamily -> "TeX Gyre Heros", 14]}
      },
     Alignment -> {Left, Center}, Spacings -> {0.6, 0.4}], 
    Background -> White, FrameStyle -> LightGray, RoundingRadius -> 5,
     FrameMargins -> 5];
  
  epilogParts = Join[
    If[effectiveShowZoomBox, {{FaceForm[None], 
       EdgeForm[{Black, Dashed, AbsoluteThickness[2]}], 
       Rectangle[{paddedZoom[[1, 1]], 
         paddedZoom[[2, 1]]}, {paddedZoom[[1, 2]], 
         paddedZoom[[2, 2]]}]}}, {}],
    If[effectiveShowZoom, {Inset[zoomPlt, 
       Scaled[zoomedPlotPos], {Left, Bottom}, 
       Scaled[zoomedPlotSize]]}, {}],
    If[effectiveShowZoom, {{Opacity[0.4], Black, 
       AbsoluteThickness[1.5], linePrimitives}}, {}]
    ];
  
  plt = Show[
    mainPlotFn[dominatedPts, 
     PlotStyle -> 
      Directive[RGBColor["#3f90da"], AbsolutePointSize[6]], 
     PlotRange -> All],
    mainPlotFn[paretoPts, 
     PlotStyle -> 
      Directive[RGBColor["#ffa90e"], AbsolutePointSize[8]], 
     PlotRange -> All],
    mainPlotFn[markedPts, 
     PlotStyle -> 
      Directive[RGBColor["#bd1f01"], AbsolutePointSize[9]], 
     PlotRange -> All],
    Axes -> False,
    Frame -> True,
    FrameTicks -> frameTicksSpec,
    FrameLabel -> {Style[xLabel, Black, FontSize -> 30, 
       FontFamily -> "TeX Gyre Heros"], 
      Style[yLabel, Black, FontSize -> 30, 
       FontFamily -> "TeX Gyre Heros"]},
    LabelStyle -> 
     Directive[Black, FontSize -> 28, 
      FontFamily -> "TeX Gyre Heros"],
    FrameTicksStyle -> Directive[AbsoluteThickness[2]],
    PlotRangePadding -> rangePadding,
    ImageSize -> 1300,
    AspectRatio -> 0.7,
    Epilog -> 
     Join[epilogParts, {Inset[legend, 
        Scaled[legendPos], {Left, Bottom}]}]
    ];
  
  (*Make the plot with the CMS guidelines.*)
  cmsText = 
   Row[{Style["CMS", Bold, FontFamily -> "TeX Gyre Heros", 24], 
     Style[" Simulation Preliminary", Italic, 
      FontFamily -> "TeX Gyre Heros", 24]}];
  tevText = Row[{Style["14 TeV", FontFamily -> "TeX Gyre Heros", 24]}];
  
  outer = Graphics[{
     Inset[plt, Scaled[{0.01, 0.01}], {Left, Bottom}, Scaled[0.98]],
     Inset[cmsText, Scaled[cmsPos], {Left, Bottom}],
     Inset[tevText, Scaled[tevPos], {Left, Bottom}]
     },
    PlotRange -> {{0, 1}, {0, 1}},
    AspectRatio -> 0.5, ImageSize -> 1300
    ];
  
  outer
  ]

(*Export a plot next to its data as dir/<dataName>.<ext>, one file per \
format, e.g. formats {"SVG", "PDF"} -> .svg and .pdf files.*)
exportParetoPlot[plot_, dataName_String, dir_String, formats_List] :=
 (Export[FileNameJoin[{dir, dataName <> "." <> ToLowerCase[#]}],
     plot, #] & /@ formats);
