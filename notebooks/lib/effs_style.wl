(* ::Package:: *)

(* effs_style.wl -- part of the notebooks/lib shared library for the
   effs_*.nb model-picking notebooks. Loaded with Get[]; all
   symbols live in Global`.

   Shared plot palette: the CMS-recommended Petroff 10-color set.
   The plotting functions style everything explicitly (font
   "TeX Gyre Heros", sizes 16-32, ImageSize 1300 with pixel-
   fitted inset positions); those option blocks stay verbatim
   inside the function bodies in effs_plots.wl so the published
   figure style cannot drift. *)

ClearAll[hexColors];

hexColors = {"#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6",
   "#a96b59", "#e76300", "#b9ac70", "#717581", "#92dadd"};
