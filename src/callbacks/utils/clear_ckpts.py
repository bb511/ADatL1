# Clear the directory where the checkpoints were stored if it exists.
import os
from pathlib import Path
import shutil

from pytorch_lightning.callbacks import Callback


class ClearRunCheckpointDir(Callback):
    """Clears the checkpoint directory of a run."""

    def __init__(self, run_ckpts_dir: str):
        self.run_ckpts_dir = Path(run_ckpts_dir)

    def on_fit_start(self, trainer, pl_module):
        if not self.run_ckpts_dir.is_dir():
            return

        for item in self.run_ckpts_dir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
            except FileNotFoundError:
                pass
