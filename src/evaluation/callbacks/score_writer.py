"""The scores of named validation sets at every epoch of a replay, written in the layout
of a score dump (``src/callbacks/score_dump.py``), so that
``src.evaluation.components.tables.load_scores``, the lattice step and
``scripts/analysis/cap_beta_fix.py`` read the output as a scores-only lattice.

A run's files sit in ``<dirpath>/<run>/``, ``<run>`` the name of the run: ``initial.pt``
and ``epoch_<nnnn>.pt``, each ``{"epoch": e, "val": {set: {key: (n,)}}}``, and
``meta.pt``, with ``stored`` (``{"val": sets, "test": []}``), ``keys`` and ``rows``:
for each set, the indices of the rows kept among the rows its loader serves uncapped
(``row_caps`` of the evaluation), or None for every row.
"""

from pathlib import Path

from pytorch_lightning.callbacks import Callback

from src.callbacks.score_dump import _Collector, _save_atomic


def _epoch(ckpt_name: str) -> int:
    return -1 if ckpt_name == "initial" else int(ckpt_name.removeprefix("epoch_"))


class ScoreWriter(Callback):
    """Write the scores of ``sets`` at every checkpoint of the validation pass.

    The scores are those the pass serves, stored in the dump or rescored from its
    weights, in the row order of the sets' loaders; the rows of a capped set are those
    that the trainer's ``rows`` names (``src.evaluation.dump.RunEvaluation``). The run
    and the epoch are read from the stem and the parent of ``pl_module._ckpt_path``.

    Args:
        dirpath: Directory of the runs' outputs, kept apart from the evaluation's
            ``out_dir``, since it is far larger than the tables.
        sets: Validation sets whose scores are written.
        keys: Per-event output keys written; a key the module does not emit is left
            out.
        name: Callback name.
    """

    def __init__(
        self,
        dirpath: str,
        sets: list[str],
        keys: list[str] = ("ascore/full", "fit/full"),
        name: str = "score_writer",
    ):
        super().__init__()
        self.dirpath, self.sets, self.keys = Path(dirpath), list(sets), list(keys)
        self.name = name

    def on_test_epoch_start(self, trainer, pl_module):
        """Check that every set is served and reset the scores."""
        missing = [name for name in self.sets if name not in trainer.test_dataloaders]
        if missing:
            raise ValueError(f"{self.__class__.__name__} finds no loader of {missing}.")
        self.collector = _Collector(self.keys, self.sets)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect the scores of the batches of the written sets."""
        name = list(trainer.test_dataloaders)[dataloader_idx]
        if name in self.sets:
            self.collector.add(name, outputs, batch)

    def on_test_epoch_end(self, trainer, pl_module):
        """Write the epoch's file and, at the initial state, the run's meta.pt."""
        ckpt = Path(pl_module._ckpt_path)
        run_dir, epoch = self.dirpath / ckpt.parent.name, _epoch(ckpt.stem)
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {"epoch": epoch, "val": self.collector.scores()}
        _save_atomic(record, run_dir / f"{ckpt.stem}.pt")
        if epoch < 0:
            _save_atomic(self._meta(trainer), run_dir / "meta.pt")

    def _meta(self, trainer) -> dict:
        rows = getattr(trainer, "rows", {}).get("val", {})
        stored = {"val": self.sets, "test": []}
        rows = {name: rows.get(name) for name in self.sets}
        return {"stored": stored, "keys": self.collector.present_keys(), "rows": rows}
