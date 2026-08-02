(* ::Package:: *)

(* corrs_lib.wl -- part of the notebooks/lib shared library for the
   effs_*.nb model-picking notebooks. Loaded with Get[]; all
   symbols live in Global`.

   Spearman correlation between each run's validation metric and
   its median efficiency, from the CSVs exported by
   scripts/analysis/get_mlflow_corrs.py into exported_metrics/. *)

ClearAll[ZeroVarianceQ, RunMethodFromName, MetricFilePatternForMethod, EffMedFile, MetricFileForRun, LoadMetricCSV, MergeOnStep, FlipSignQ, SpearmanFromCSVs, ComputeRunCorrelation, ComputeExperimentCorrelations, ComputeAllCorrelations];

ZeroVarianceQ[x_List] := Length[DeleteDuplicates[N[x]]] <= 1;

(*Infer validation method from run folder name*)
RunMethodFromName[runName_String] := 
  Which[StringStartsQ[runName, "cap"], "CAP", 
   StringStartsQ[runName, "stability"], "Stable", 
   StringStartsQ[runName, "wasserstein"], "W1", 
   StringStartsQ[runName, "cvar25"], "Semi", True, "Unknown"];

(*Which metric file should be used for each method*)
MetricFilePatternForMethod[method_String] := 
  Switch[method, "CAP", {"val_summary_cap_ema*.csv"}, 
   "Stable", {"val_summary_*drift_ema*.csv", 
    "val_summary_operational_drift_ema.csv"}, 
   "W1", {"val_summary_w1dist_ema*.csv"}, 
   "Semi", {"val_summary_eff_cvar25_ema*.csv", 
    "val_summary_eff_cvar25_ema_operational.csv"}, _, None];

(*Find the median efficiency CSV*)
EffMedFile[runDir_String] := 
  Module[{files}, 
   files = FileNames["val_summary_eff_med*.csv", runDir];
   files = SortBy[files, StringLength];
   If[files === {}, Missing["NoEffMed"], First[files]]];

(*Find the main metric CSV for a run*)
MetricFileForRun[runDir_String, method_String] := 
  Module[{pats, files}, pats = MetricFilePatternForMethod[method];
   If[pats === None, Return[Missing["UnknownMethod"]]];
   files = DeleteDuplicates@Flatten[FileNames[#, runDir] & /@ pats];
   If[files === {}, Missing["NoMetricFile"], First[files]]];

(*Read a CSV exported by the Python script*)
LoadMetricCSV[file_String] := 
  Module[{ds, rows}, ds = Import[file, "CSV"];
   If[Length[ds] < 2, Return[<||>]];
   rows = Rest[ds];
   Association["step" -> ToExpression[rows[[All, 3]]], 
    "value" -> ToExpression[rows[[All, 2]]]]];

(*Inner join on step*)
MergeOnStep[effAssoc_Association, metricAssoc_Association] := 
  Module[{effSteps, metSteps, common, effMap, metMap}, 
   effSteps = effAssoc["step"];
   metSteps = metricAssoc["step"];
   common = Intersection[effSteps, metSteps];
   effMap = AssociationThread[effSteps -> effAssoc["value"]];
   metMap = AssociationThread[metSteps -> metricAssoc["value"]];
   <|"step" -> common, "eff" -> Lookup[effMap, common], 
    "metric" -> Lookup[metMap, common]|>];

(*Whether to flip sign so "better metric" means larger is better*)
FlipSignQ[method_String] := MemberQ[{"Stable", "W1"}, method];

(*Compute Spearman from two CSV files*)
SpearmanFromCSVs[effFile_String, metricFile_String, method_String, 
  flipSign_ : True] := 
 Module[{eff, met, merged, x, y}, eff = LoadMetricCSV[effFile];
  met = LoadMetricCSV[metricFile];
  If[eff === <||> || met === <||>, Return[Missing["EmptyCSV"]]];
  merged = MergeOnStep[eff, met];
  x = merged["metric"];
  y = merged["eff"];
  If[Length[x] < 3, Return[Missing["TooFewPoints"]]];
  If[flipSign && FlipSignQ[method], x = -x];
  If[ZeroVarianceQ[x], Return[Missing["ZeroVarianceMetric"]]];
  If[ZeroVarianceQ[y], Return[Missing["ZeroVarianceEfficiency"]]];
  Quiet[Check[SpearmanRho[x, y], Missing["SpearmanFailed"]]]]

(*Compute one run*)
ComputeRunCorrelation[runDir_String, flipSign_ : True] := 
  Module[{runName, method, effFile, metricFile, rho}, 
   runName = FileNameTake[runDir];
   method = RunMethodFromName[runName];
   effFile = EffMedFile[runDir];
   metricFile = MetricFileForRun[runDir, method];
   If[MissingQ[effFile] || MissingQ[metricFile], 
    Return[<|"Run" -> runName, "Method" -> method, 
      "EffMedFile" -> effFile, "MetricFile" -> metricFile, 
      "SpearmanRho" -> Missing["Unavailable"]|>]];
   rho = SpearmanFromCSVs[effFile, metricFile, method, flipSign];
   <|"Run" -> runName, "Method" -> method, "EffMedFile" -> effFile, 
    "MetricFile" -> metricFile, "SpearmanRho" -> rho|>];

(*Compute all runs inside one experiment folder*)
ComputeExperimentCorrelations[expDir_String, flipSign_ : True] := 
  Module[{runDirs, results}, 
   runDirs = Select[FileNames["*", expDir], DirectoryQ];
   results = ComputeRunCorrelation[#, flipSign] & /@ runDirs;
   AssociationThread[FileNameTake /@ runDirs -> results]];

(*Compute all experiment folders under a root*)
ComputeAllCorrelations[rootDir_String, flipSign_ : True] := 
  Module[{expDirs}, 
   expDirs = Select[FileNames["*", rootDir], DirectoryQ];
   AssociationThread[FileNameTake /@ expDirs, 
    ComputeExperimentCorrelations[#, flipSign] & /@ expDirs]];
