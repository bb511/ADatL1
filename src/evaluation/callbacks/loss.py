from typing import Dict
import re
from pytorch_lightning.callbacks import Callback


_FILENAME_RE = re.compile(
    r'^ds=(?P<dataset>.+?)__metric=(?P<metric_name>.+?)__value=(?P<metric_value>.+?)__epoch=(?P<epoch>\d+)\.ckpt$'
)

def _parse_filename(filename: str) -> Dict[str, str]:
    m = _FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    return {
        key: m.group(key)
        for key in ["dataset", "metric_name", "metric_value", "epoch"]
    }


class TestLossCallback(Callback):
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        """Log loss values for every checkpoint."""
        try:
            dset_ref = _parse_filename(trainer.ckpt_path_name).get("dataset")
            if dset_ref is None:
                return
        except ValueError:
            return

        outdict = pl_module.outlog(outputs)
        dset_key = list(getattr(trainer, f"test_dataloaders").keys())[dataloader_idx]        

        self.log_dict(
            {f"test/ckpt.{dset_ref}/ds.{dset_key}/{k}": v for k, v in outdict.items()},
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,  # !!
            add_dataloader_idx=False,
            batch_size=len(batch)
        )
