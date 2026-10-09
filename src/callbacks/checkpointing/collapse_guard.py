"""ModelCheckpoint that never selects an epoch whose latent code has collapsed.

The validation objective ``val/loss_total = loss_reco + mi_gamma * loss_mi`` rewards
a collapsing latent: once the Bernoulli codes stop varying, H(L) and with it the MI
term go towards zero, so such an epoch can become the "best" one although the
bottleneck no longer carries information about the event.

This callback measures, after every fit-time validation epoch, the joint entropy of
the hard evaluation codes on the normal validation split, logs it, and lets only the
epochs with ``H(L) >= min_joint_code_entropy_bits`` compete for the monitored
checkpoint. The statistic is the one stage 3 applies to ``loss_total.ckpt``
(``src/evaluation/callbacks/latent_collapse.py``):

    H(L) = - sum_c p(c) log2 p(c),   c = ([p_j >= threshold])_j,   in bits,

over the codes observed in the epoch. It needs the hard code of every validation
event, which ``AE.model_step`` returns as ``latent_code`` outside training. Reading
it draws no random numbers, so the guard never changes the training trajectory;
it only changes WHICH epoch ends up in the checkpoint.

When no epoch qualifies, the best collapsed epoch (by the monitored metric) is kept
as a fallback and moved to the monitored filename at the end of training, and the
report says so (status ``no_non_collapsed_epoch``). Stages 1-3 then still run, and
stage 3 rejects the run through its own collapse rule.

Files next to the checkpoint (``<name>`` = ``filename``, e.g. ``loss_total``):

    <name>.ckpt             best epoch with H(L) >= minimum (or the fallback)
    <name>_collapsed.ckpt   fallback; exists only while no epoch has qualified
    <name>_guard.json       per-epoch history and the final decision

Resuming: the guard state travels in the checkpoint like ModelCheckpoint's own
state. A ``last.ckpt`` written by another ModelCheckpoint that runs before this one
in the same epoch carries that epoch's entropy, but not its monitored value or this
callback's decision for it, exactly as it misses ModelCheckpoint's own best-model
update for that epoch (standard Lightning behaviour).
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer.states import TrainerFn

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

#: Key under which validation_step returns the hard (0/1) latent codes.
LATENT_CODE_KEY = "latent_code"
#: Metrics this callback logs every validation epoch.
ENTROPY_METRIC = "val/latent_joint_code_entropy_bits"
CODE_COUNT_METRIC = "val/latent_observed_code_count"

STATUS_OK = "ok"
STATUS_FALLBACK = "no_non_collapsed_epoch"
STATUS_NO_CHECKPOINT = "no_checkpoint"

REPORT_SCHEMA_VERSION = 1


def joint_code_entropy_bits(code_counts: Mapping[Any, int]) -> float:
    """Shannon entropy in bits of the empirical code distribution."""
    total = int(sum(int(count) for count in code_counts.values()))
    if total <= 0:
        raise ValueError("Cannot compute the entropy of an empty code histogram.")
    entropy = 0.0
    for count in code_counts.values():
        if count > 0:
            probability = int(count) / total
            entropy -= probability * math.log2(probability)
    # A single observed code gives -0.0; report it as 0.0.
    return max(entropy, 0.0)


def count_codes(latent_code: torch.Tensor) -> Counter:
    """Histogram of the hard codes in one batch, keyed by the packed bit pattern."""
    if not isinstance(latent_code, torch.Tensor):
        raise TypeError(f"{LATENT_CODE_KEY!r} must be a tensor, got {type(latent_code)!r}.")
    if latent_code.ndim == 0:
        raise ValueError(f"{LATENT_CODE_KEY!r} must have a batch dimension.")
    codes = latent_code.detach().to("cpu")
    codes = codes.unsqueeze(1) if codes.ndim == 1 else torch.flatten(codes, start_dim=1)
    if codes.shape[0] == 0:
        return Counter()
    if not bool(((codes == 0) | (codes == 1)).all()):
        raise RuntimeError(
            f"{LATENT_CODE_KEY!r} must contain hard 0/1 codes; got values in "
            f"[{float(codes.min())}, {float(codes.max())}]. Was the module left in "
            "training mode during validation?"
        )
    values = codes.to(torch.uint8).numpy()
    packed = np.packbits(values, axis=1, bitorder="little")
    unique_codes, counts = np.unique(packed, axis=0, return_counts=True)
    return Counter(
        {code.tobytes(): int(count) for code, count in zip(unique_codes, counts)}
    )


class CollapseGuardedModelCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` whose monitored checkpoint skips collapsed-latent epochs.

    :param min_joint_code_entropy_bits: An epoch is eligible only when the joint code
        entropy of the normal validation split is at least this many bits. The
        default 0.05 bits blocks full collapse only (one code for ~99.5% of events);
        stage 3 still applies its own, stricter feasibility rule afterwards.
    :param dataset: Name of the validation dataloader the codes are read from.
    :param latent_code_key: Key of the hard codes in the validation_step output.
    :param log_entropy: Log ``val/latent_joint_code_entropy_bits`` and
        ``val/latent_observed_code_count`` every validation epoch.
    :param kwargs: Passed to ``ModelCheckpoint``. ``monitor`` and a literal
        ``filename`` are required; the guard decides once per validation epoch, so
        ``every_n_train_steps``, ``train_time_interval`` and
        ``save_on_train_epoch_end=True`` are rejected.
    """

    def __init__(
        self,
        *args: Any,
        min_joint_code_entropy_bits: float = 0.05,
        dataset: str = "normal",
        latent_code_key: str = LATENT_CODE_KEY,
        log_entropy: bool = True,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("save_on_train_epoch_end"):
            raise ValueError(
                "CollapseGuardedModelCheckpoint decides after validation; "
                "save_on_train_epoch_end must be False."
            )
        kwargs["save_on_train_epoch_end"] = False
        if kwargs.get("every_n_train_steps") or kwargs.get("train_time_interval"):
            raise ValueError(
                "CollapseGuardedModelCheckpoint decides once per validation epoch; "
                "every_n_train_steps and train_time_interval are not supported."
            )
        super().__init__(*args, **kwargs)

        if not self.monitor:
            raise ValueError("CollapseGuardedModelCheckpoint needs a `monitor`.")
        if not self.filename or "{" in str(self.filename):
            raise ValueError(
                "CollapseGuardedModelCheckpoint needs a literal `filename` (e.g. "
                "'loss_total'); the pipeline reads the checkpoint by that name."
            )
        if self.save_top_k != 1:
            raise ValueError("CollapseGuardedModelCheckpoint supports save_top_k=1 only.")
        minimum = float(min_joint_code_entropy_bits)
        if not math.isfinite(minimum) or minimum < 0.0:
            raise ValueError(
                "min_joint_code_entropy_bits must be a finite, non-negative number."
            )

        self.min_joint_code_entropy_bits = minimum
        self.dataset = str(dataset)
        self.latent_code_key = str(latent_code_key)
        self.log_entropy = bool(log_entropy)

        # Per-epoch accumulators.
        self._code_counts: Counter = Counter()
        self._n_events = 0
        self._epoch_entropy: Optional[float] = None
        self._epoch_collapsed = False
        self._pending_record: Optional[dict[str, Any]] = None

        # Persistent guard state (checkpointed, see state_dict).
        self.history: list[dict[str, Any]] = []
        self.selected: Optional[dict[str, Any]] = None
        self.fallback: Optional[dict[str, Any]] = None
        self.status: Optional[str] = None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    @property
    def checkpoint_path(self) -> Path:
        return Path(self.dirpath) / f"{self.filename}{self.FILE_EXTENSION}"

    @property
    def fallback_path(self) -> Path:
        return Path(self.dirpath) / f"{self.filename}_collapsed{self.FILE_EXTENSION}"

    @property
    def report_path(self) -> Path:
        return Path(self.dirpath) / f"{self.filename}_guard.json"

    # ------------------------------------------------------------------
    # Code statistics during fit-time validation
    # ------------------------------------------------------------------
    @staticmethod
    def _is_fit_validation(trainer) -> bool:
        return trainer.state.fn == TrainerFn.FITTING and not trainer.sanity_checking

    @staticmethod
    def _dataloader_name(trainer, dataloader_idx: int) -> Optional[str]:
        loaders = trainer.val_dataloaders
        if isinstance(loaders, Mapping):
            names = list(loaders.keys())
            return str(names[dataloader_idx]) if dataloader_idx < len(names) else None
        # A single unnamed loader is the normal split.
        return "normal" if dataloader_idx == 0 else None

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._code_counts = Counter()
        self._n_events = 0
        self._epoch_entropy = None
        self._epoch_collapsed = False
        self._pending_record = None

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._is_fit_validation(trainer):
            return
        if self._dataloader_name(trainer, dataloader_idx) != self.dataset:
            return
        if not isinstance(outputs, Mapping) or self.latent_code_key not in outputs:
            raise RuntimeError(
                f"{type(self).__name__} needs the hard latent codes of every "
                f"'{self.dataset}' validation batch, but validation_step returned no "
                f"{self.latent_code_key!r}. AE.model_step provides it outside training."
            )
        batch_counts = count_codes(outputs[self.latent_code_key])
        self._code_counts.update(batch_counts)
        self._n_events += int(sum(batch_counts.values()))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._is_fit_validation(trainer) or self._n_events == 0:
            return
        self._epoch_entropy = joint_code_entropy_bits(self._code_counts)
        self._epoch_collapsed = self._epoch_entropy < self.min_joint_code_entropy_bits
        # Recorded now, before any ModelCheckpoint's on_validation_end, so a
        # last.ckpt written later in this epoch already carries the entry. The
        # monitored value is aggregated only after the epoch-end hooks and is
        # filled in by on_validation_end.
        self._pending_record = {
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "monitor_value": None,
            "joint_code_entropy_bits": float(self._epoch_entropy),
            "observed_code_count": int(len(self._code_counts)),
            "n_events": int(self._n_events),
            "collapsed": bool(self._epoch_collapsed),
            "saved": False,
        }
        self.history.append(self._pending_record)
        if self.log_entropy:
            for name, value in (
                (ENTROPY_METRIC, self._epoch_entropy),
                (CODE_COUNT_METRIC, float(len(self._code_counts))),
            ):
                pl_module.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    prog_bar=False,
                    add_dataloader_idx=False,
                )

    # ------------------------------------------------------------------
    # Checkpoint decision
    # ------------------------------------------------------------------
    def check_monitor_top_k(self, trainer, current: Optional[torch.Tensor] = None) -> bool:
        """A collapsed epoch never enters the top-k, whatever its monitored value."""
        if self._epoch_collapsed:
            return False
        return super().check_monitor_top_k(trainer, current)

    def on_validation_end(self, trainer, pl_module) -> None:
        if self._should_skip_saving_checkpoint(trainer):
            return super().on_validation_end(trainer, pl_module)
        if self._pending_record is None:
            raise RuntimeError(
                f"{type(self).__name__}: no '{self.dataset}' validation codes were "
                f"seen in epoch {trainer.current_epoch}, so the collapse guard cannot "
                "decide. Is the normal split among the fit-time validation loaders?"
            )

        record = self._pending_record
        current = trainer.callback_metrics.get(self.monitor)
        record["monitor_value"] = None if current is None else float(current)

        # check_monitor_top_k refuses a collapsed epoch; _update_best_and_save
        # marks the record as selected before the file is written.
        super().on_validation_end(trainer, pl_module)

        if not record["saved"] and record["collapsed"]:
            log.info(
                f"[collapse guard] epoch {record['epoch']}: H(L) = "
                f"{record['joint_code_entropy_bits']:.4f} bits < "
                f"{self.min_joint_code_entropy_bits} bits, {self.monitor} = "
                f"{record['monitor_value']}; not eligible for {self.checkpoint_path.name}."
            )
            if not self.best_model_path:
                self._update_fallback(trainer, record)

        # The decision for this epoch is made; never reuse it.
        self._epoch_entropy = None
        self._epoch_collapsed = False
        self._pending_record = None

    def _update_best_and_save(self, current, trainer, monitor_candidates) -> None:
        # Update the guard state BEFORE the checkpoint is written, so the saved
        # file (a possible resume point) already knows it is the selection.
        record = self._pending_record
        if record is not None:
            record["saved"] = True
            self.selected = {**record, "path": str(self.checkpoint_path), "fallback": False}
            self._drop_fallback(trainer)
        super()._update_best_and_save(current, trainer, monitor_candidates)

    def _is_better(self, value: float, reference: float) -> bool:
        return value < reference if self.mode == "min" else value > reference

    def _update_fallback(self, trainer, record: dict[str, Any]) -> None:
        value = record["monitor_value"]
        if value is None or not math.isfinite(value):
            return
        if self.fallback is not None and not self._is_better(
            value, float(self.fallback["monitor_value"])
        ):
            return
        trainer.save_checkpoint(str(self.fallback_path), self.save_weights_only)
        self.fallback = {**record, "path": str(self.fallback_path)}

    def _drop_fallback(self, trainer) -> None:
        if self.fallback is None:
            return
        if trainer.is_global_zero and self.fallback_path.exists():
            self.fallback_path.unlink()
        self.fallback = None

    # ------------------------------------------------------------------
    # End of training: fallback and report
    # ------------------------------------------------------------------
    def on_train_end(self, trainer, pl_module) -> None:
        super().on_train_end(trainer, pl_module)
        if trainer.fast_dev_run or not self.history:
            return

        if self.best_model_path:
            self.status = STATUS_OK
        elif self.fallback is not None:
            self.status = STATUS_FALLBACK
            if trainer.is_global_zero:
                os.replace(self.fallback_path, self.checkpoint_path)
            self.best_model_path = str(self.checkpoint_path)
            self.best_model_score = torch.tensor(float(self.fallback["monitor_value"]))
            self.selected = {**self.fallback, "path": str(self.checkpoint_path), "fallback": True}
            self.fallback = None
            log.warning(
                f"[collapse guard] every validation epoch had H(L) < "
                f"{self.min_joint_code_entropy_bits} bits. {self.checkpoint_path.name} "
                f"holds the best collapsed epoch {self.selected['epoch']} "
                f"({self.monitor} = {self.selected['monitor_value']}); status "
                f"'{STATUS_FALLBACK}' in {self.report_path.name}."
            )
        else:
            self.status = STATUS_NO_CHECKPOINT
            log.warning(
                f"[collapse guard] no epoch produced {self.checkpoint_path.name} "
                f"(no finite {self.monitor})."
            )

        if self.selected is not None and self.status == STATUS_OK:
            log.info(
                f"[collapse guard] {self.checkpoint_path.name}: epoch "
                f"{self.selected['epoch']}, {self.monitor} = "
                f"{self.selected['monitor_value']}, H(L) = "
                f"{self.selected['joint_code_entropy_bits']:.4f} bits; "
                f"{self.n_collapsed_epochs} of {len(self.history)} epochs were collapsed."
            )
        if trainer.is_global_zero:
            self.write_report()

    @property
    def n_collapsed_epochs(self) -> int:
        return int(sum(1 for record in self.history if record["collapsed"]))

    def report(self) -> dict[str, Any]:
        collapsed = [record["epoch"] for record in self.history if record["collapsed"]]
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "checkpoint": self.checkpoint_path.name,
            "monitor": self.monitor,
            "mode": self.mode,
            "guard": {
                "statistic": "joint_code_entropy_bits",
                "representation": "hard evaluation latent codes",
                "dataset": self.dataset,
                "min_joint_code_entropy_bits": self.min_joint_code_entropy_bits,
                "rule": "an epoch is eligible iff joint_code_entropy_bits >= minimum",
            },
            "status": self.status,
            "selected": self.selected,
            "n_validation_epochs": len(self.history),
            "n_collapsed_epochs": len(collapsed),
            "first_collapsed_epoch": collapsed[0] if collapsed else None,
            "history": self.history,
        }

    def write_report(self) -> Path:
        path = self.report_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.report(), indent=2) + "\n")
        os.replace(tmp, path)
        return path

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["collapse_guard"] = {
            "min_joint_code_entropy_bits": self.min_joint_code_entropy_bits,
            "history": list(self.history),
            "selected": self.selected,
            "fallback": self.fallback,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        guard = state_dict.get("collapse_guard") or {}
        self.history = list(guard.get("history", []))
        self.selected = guard.get("selected")
        self.fallback = guard.get("fallback")
