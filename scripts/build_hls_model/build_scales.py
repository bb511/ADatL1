import pickle
import json
from pathlib import Path
import numpy as np

base_path = Path("data/data_2025E+G/mlready/eminimal_pdefault_default/robust")

with open(base_path / "object_feature_map.json", "r") as f:
    fmap = json.load(f)


def load_pkl(name: str):
    with open(base_path / name, "rb") as f:
        return pickle.load(f)


norm = {
    "MET": load_pkl("MET_norm_params.pkl"),
    "egammas": load_pkl("egammas_norm_params.pkl"),
    "jets": load_pkl("jets_norm_params.pkl"),
    "muons": load_pkl("muons_norm_params.pkl"),
}

N = 57

shift = np.zeros(N, dtype=np.float64)  # default shift = 0
scale = np.ones(N, dtype=np.float64)  # default scale = 1

# ---------- Fill where params exist ----------
for obj_name, feat_dict in fmap.items():
    if obj_name not in norm:
        continue  # keep defaults

    for feat_name, indices in feat_dict.items():
        if feat_name not in norm[obj_name]:
            continue  # keep defaults

        params = norm[obj_name][feat_name]
        sft = float(params["shift"])
        scl = float(params["scale"])

        for idx in indices:
            if 0 <= idx < N:
                shift[idx] = sft
                scale[idx] = scl

shift_int = shift.astype(int)
scale_int = scale.astype(int)

# ---------- Sanity checks ----------
print("shift = [")
print(", ".join(str(x) for x in shift_int))
print("]")

print("\nscale = [")
print(", ".join(str(x) for x in scale_int))
print("]")
