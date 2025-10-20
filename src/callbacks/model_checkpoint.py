from typing import List, Optional
import os
from collections import defaultdict
import itertools

import torch
from pytorch_lightning.callbacks import ModelCheckpoint


class DatasetAwareModelCheckpoint(ModelCheckpoint):
    """
    ModelCheckpoint class to store dataset-specific checkpoints.

    Args:
        metric_name: Name of the loss key in the validation step outputs
        dirpath: Directory to save the checkpoints
        filename: Checkpoint filename format
        save_top_k: Number of best checkpoints to keep per strategy
        mode: 'min' or 'max' for loss minimization or maximization
        **kwargs: Additional arguments passed to ModelCheckpoint
    """

    prefix = ""

    def __init__(
        self,
        metric_name: str = "loss",
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        save_top_k: int = 1,
        mode: str = "min",
        ds_keys_to_skip: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            save_top_k=save_top_k,
            mode=mode,
            **kwargs,
        )
        self.metric_name = metric_name
        self.metric_cache = defaultdict(list)
        self.ds_keys_to_skip = ds_keys_to_skip or []

        # For tracking and saving per-dataset checkpoints
        self.custom_checkpoints = defaultdict(list)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Store loss values for each dataset during validation."""
        dset_key = list(getattr(trainer, "val_dataloaders").keys())[dataloader_idx]
        if dset_key in self.ds_keys_to_skip:
            return

        if self.metric_name in outputs.keys():
            self.metric_cache[dset_key].append(
                outputs[self.metric_name].detach().cpu().unsqueeze(0)
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Compute dataset losses and delegate to strategy-specific implementations.

        Collects and processes loss data, then calls strategy-specific methods
        for checkpoint selection.
        """
        list_dset_key = [
            ds_key for ds_key in list(getattr(trainer, "val_dataloaders").keys())
            if ds_key not in self.ds_keys_to_skip
        ]
        
        # Fill metric cache:
        if len(self.metric_cache) == 0:
            self.metric_cache = defaultdict(list)
            for key, value in trainer.callback_metrics.items():
                if key.endswith(self.metric_name):
                    dset_key = key.split("/")[1]
                    self.metric_cache[dset_key].append(
                        value.detach().cpu().unsqueeze(0)
                    )

        # Compute total loss and length for each dataset in this epoch
        epoch_loss_totals, epoch_loss_lengths = {}, {}

        for dset_key in list_dset_key:
            if dset_key in self.metric_cache.keys():
                losses = self.metric_cache[dset_key]
                if len(losses) == 0:
                    continue

                loss = torch.cat(losses)
                local_sum = loss.sum()
                local_count = torch.tensor(len(loss), device=local_sum.device)
            else:
                # If no losses were recorded, set to zero
                local_sum = torch.tensor(0.0, device=pl_module.device)
                local_count = torch.tensor(0, device=pl_module.device)

            if trainer.world_size > 1:  # ddp
                local_sum = local_sum.to(pl_module.device)
                local_count = local_count.to(pl_module.device)

                # Sum values across all ranks
                torch.distributed.all_reduce(
                    local_sum, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    local_count, op=torch.distributed.ReduceOp.SUM
                )

            # Store synchronized values (if there was data)
            if local_count.item() > 0:
                epoch_loss_totals[dset_key] = local_sum.item()
                epoch_loss_lengths[dset_key] = local_count.item()

            # Clear cache for next epoch
            if dset_key in self.metric_cache:
                self.metric_cache[dset_key] = []

        # Strategy-specific implementation in subclasses
        self._process_dataset_losses(
            trainer, pl_module, epoch_loss_totals, epoch_loss_lengths
        )

    def _process_dataset_losses(
        self, trainer, pl_module, epoch_loss_totals, epoch_loss_lengths
    ):
        """Process dataset losses according to the checkpoint strategy."""
        raise NotImplementedError("Subclasses must implement _process_dataset_losses")

    def _save_top_k_custom_checkpoints(
        self, trainer, pl_module, key: str, metric_value: float
    ):
        """Save a checkpoint if it's among the top_k best for the given key."""
        new_checkpoint = {
            "epoch": trainer.current_epoch,
            "value": metric_value,
            "filepath": None,  # will be set if we save it
        }

        # Combine with existing checkpoints
        all_checkpoints = self.custom_checkpoints.get(key, []) + [new_checkpoint]

        # Sort by value (ascending for min mode, descending for max mode)
        sorted_checkpoints = sorted(
            all_checkpoints, key=lambda x: x["value"], reverse=(self.mode == "max")
        )
        should_save = new_checkpoint in sorted_checkpoints[: self.save_top_k]

        if should_save:
            filepath = self._save_custom_checkpoint(
                trainer, pl_module, key, metric_value
            )
            new_checkpoint["filepath"] = filepath

            # Update the list with only the top_k checkpoints
            top_k_checkpoints = sorted_checkpoints[: self.save_top_k]

            # Remove any old checkpoints that didn't make the cut
            for checkpoint in sorted_checkpoints[self.save_top_k :]:
                if checkpoint.get("filepath") and os.path.exists(
                    checkpoint["filepath"]
                ):
                    try:
                        os.remove(checkpoint["filepath"])
                    except OSError:
                        pass

            # Update the list
            self.custom_checkpoints[key] = top_k_checkpoints

    def _save_custom_checkpoint(
        self, trainer, pl_module, key: str, metric_value: float
    ):
        """Save a checkpoint with a custom key and metric value."""
        # Format filename based on the template
        filename = self.filename or "epoch={epoch:02d}__step={step}"
        filename = filename.format(
            epoch=trainer.current_epoch,
            step=trainer.global_step,
        )

        # Make the key safe for filenames
        safe_string = lambda string: string.replace("/", "_").replace(":", "_").replace(" ", "_")
        idname = f"ds={safe_string(key)}__metric={safe_string(self.metric_name)}__value={metric_value:.6f}"

        # Modify dirpath to include key-specific subdirectory
        dirpath = self.dirpath or trainer.default_root_dir
        custom_dirpath = os.path.join(dirpath, safe_string(key))
        os.makedirs(custom_dirpath, exist_ok=True)

        # Save the checkpoint
        prefix = (
            self.prefix
            if hasattr(self, "prefix") and getattr(self, "prefix") != None
            else ""
        )
        filepath = os.path.join(custom_dirpath, f"prefix={prefix}__{idname}__{filename}.ckpt")
        self._save_checkpoint(trainer=trainer, filepath=filepath)
        return filepath


class SingleDatasetModelCheckpoint(DatasetAwareModelCheckpoint):
    """ModelCheckpoint that saves checkpoints based on individual dataset performance."""

    prefix = "single"

    def _process_dataset_losses(
        self, trainer, pl_module, epoch_loss_totals, epoch_loss_lengths
    ):
        """Process dataset losses using the per-dataset strategy."""
        for dset_key in epoch_loss_totals.keys():
            # Compute mean loss for this specific dataset only
            dataset_loss = epoch_loss_totals[dset_key]
            num_examples = epoch_loss_lengths[dset_key]
            mean_loss = (
                dataset_loss / num_examples if num_examples > 0 else float("inf")
            )

            # Use the prefix 'single_' to identify single-dataset checkpoints
            key = f"{dset_key}"
            self._save_top_k_custom_checkpoints(trainer, pl_module, key, mean_loss)


class LeaveOneOutModelCheckpoint(DatasetAwareModelCheckpoint):
    """
    ModelCheckpoint that saves checkpoints based on leave-one-out dataset performance.

    For each dataset, this callback computes the mean loss on all OTHER datasets
    (leave-one-out) and saves the checkpoint that performs best according to this metric.
    This helps avoid overfitting to specific anomalies in any single dataset.
    """

    prefix = "loo"

    def _process_dataset_losses(
        self, trainer, pl_module, epoch_loss_totals, epoch_loss_lengths
    ):
        """Process dataset losses using the leave-one-out strategy."""

        for left_out_key in epoch_loss_totals.keys():
            # Compute mean loss on all datasets EXCEPT the left-out one
            other_dsets = [k for k in epoch_loss_totals.keys() if k != left_out_key]
                        
            if not other_dsets:  # Skip if there's only one dataset
                continue

            # Compute weighted mean loss across all other datasets
            total_loss = sum([epoch_loss_totals[k] for k in other_dsets])
            total_examples = sum([epoch_loss_lengths[k] for k in other_dsets])
            loo_loss = (
                total_loss / total_examples if total_examples > 0 else float("inf")
            )

            # Use the prefix 'loo_' to identify leave-one-out checkpoints
            key = f"{left_out_key}"
            self._save_top_k_custom_checkpoints(trainer, pl_module, key, loo_loss)


class LeaveKOutModelCheckpoint(DatasetAwareModelCheckpoint):
    """
    ModelCheckpoint that saves checkpoints based on leave-k-out dataset performance.

    For each dataset group, this callback computes the mean loss on the remaining datasets
    (leave-k-out) and saves the checkpoint that performs best according to this metric.
    """

    prefix = "lko"

    def __init__(
        self,
        ds_keys_selected: List[str],
        metric_name: str = "loss",
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        save_top_k: int = 1,
        mode: str = "min",
        ds_keys_to_skip: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            metric_name=metric_name,
            dirpath=dirpath,
            save_top_k=save_top_k,
            filename=filename,
            mode=mode,
            ds_keys_to_skip=ds_keys_to_skip,
            **kwargs,
        )
        self.ds_keys_selected = ds_keys_selected
    
    def _process_dataset_losses(
        self, trainer, pl_module, epoch_loss_totals, epoch_loss_lengths
    ):
        """Process dataset losses using the leave-k-out strategy."""

        available_keys = [k for k in epoch_loss_totals.keys() if k not in self.ds_keys_to_skip]
        selected_k = [k for k in available_keys if k in self.ds_keys_selected]
        selected_k_c = [k for k in available_keys if k not in self.ds_keys_selected]

        # Compute mean loss for the two groups:
        total_loss_k = sum([epoch_loss_totals[k] for k in selected_k])
        total_examples_k = sum([epoch_loss_lengths[k] for k in selected_k])
        total_loss_k_c = sum([epoch_loss_totals[k] for k in selected_k_c])
        total_examples_k_c = sum([epoch_loss_lengths[k] for k in selected_k_c])
        if total_examples_k == 0 or total_examples_k_c == 0:
            return
        
        loss_k = (
            total_loss_k / total_examples_k if total_examples_k > 0 else float("inf")
        )
        self._save_top_k_custom_checkpoints(trainer, pl_module, "leave-k-in", loss_k)
        
        loss_k_c = (
            total_loss_k_c / total_examples_k_c if total_examples_k_c > 0 else float("inf")
        )
        self._save_top_k_custom_checkpoints(trainer, pl_module, "leave-k-out", loss_k_c)
