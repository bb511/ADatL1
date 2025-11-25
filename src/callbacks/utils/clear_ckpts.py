# Clear the directory where the checkpoints were stored if it exists.
from pathlib import Path
import shutil

from pytorch_lightning.callbacks import Callback


class ClearRunCheckpointDir(Callback):
    """Clears the checkpoint directory of a run."""
    def __init__(self, run_ckpts_dir: str):
        self.run_ckpts_dir = Path(run_ckpts_dir)

    def on_fit_start(self, trainer, pl_module):
        shutil.rmtree(self.run_ckpts_dir)
