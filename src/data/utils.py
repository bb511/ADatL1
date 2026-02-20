from typing import Dict, List
from pathlib import Path
import json


def load_object_feature_map(path: str) -> Dict[str, Dict[str, List[int]]]:
    """Return {object: {feature: [col_indices...]}} as plain Python lists."""
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    return {
        obj: {feat: list(indices) for feat, indices in feats.items()}
        for obj, feats in data.items()
    }
