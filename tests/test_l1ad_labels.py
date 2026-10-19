from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.data.L1AD_datamodule import L1ADDataModule, _aux_labels

PATHS_2025 = Path(__file__).resolve().parents[1] / "configs/paths/data_2025E+G.yaml"


def _labels_by_name(names):
    labels, n_background, n_signal = {}, 0, 0
    for name in names:
        if "SingleNeutrino" in name:
            n_background += 1
            labels[name] = -n_background
        else:
            n_signal += 1
            labels[name] = n_signal
    return labels


def _paths_2025():
    return OmegaConf.load(PATHS_2025)


def test_default_labels_match_the_name_rule():
    paths = _paths_2025()
    names = sorted([*paths.signal, *paths.background])

    labels = _aux_labels(names, dict.fromkeys(paths.background))

    assert labels == _labels_by_name(names)
    assert labels["SingleNeutrino_E-10-gun"] == -1
    assert sorted(v for v in labels.values() if v > 0) == list(range(1, 21))


def test_zero_bias_run_under_background_is_background():
    paths = _paths_2025()
    background = {**dict.fromkeys(paths.background), "ZB_run398183": None}
    names = sorted([*paths.signal, *background])

    labels = _aux_labels(names, background)

    assert labels["ZB_run398183"] == -2
    assert labels["SingleNeutrino_E-10-gun"] == -1
    default = _labels_by_name(sorted([*paths.signal, *paths.background]))
    assert {k: labels[k] for k in paths.signal} == {k: default[k] for k in paths.signal}


class _StandInLoader:
    def load_folder(self, folder):
        return torch.zeros(3, 2), torch.ones(3, 2, dtype=torch.bool), torch.zeros(3)


def test_load_aux_split_labels_by_category(tmp_path):
    for name in ("GluGluHto2G_Par-MH-90", "SingleNeutrino_E-10-gun", "ZB_run398183"):
        (tmp_path / "aux" / name / "valid").mkdir(parents=True)
    background = {"SingleNeutrino_E-10-gun": "", "ZB_run398183": ""}
    stand_in = SimpleNamespace(
        loader=_StandInLoader(), hparams=SimpleNamespace(background=background)
    )

    out = L1ADDataModule._load_aux_split(stand_in, tmp_path, "valid")

    labels = {name: split.y.unique().tolist() for name, split in out.items()}
    assert labels == {
        "GluGluHto2G_Par-MH-90": [1],
        "SingleNeutrino_E-10-gun": [-1],
        "ZB_run398183": [-2],
    }
