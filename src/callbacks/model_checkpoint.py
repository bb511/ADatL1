from typing import Dict, List, Optional, Any
from collections import defaultdict
import os

import torch
from pytorch_lightning.callbacks import ModelCheckpoint


class DatasetAwareModelCheckpoint(ModelCheckpoint):
    """
    ModelCheckpoint class to store dataset-specific checkpoints.
    
    Args:
        loss_name: Name of the loss key in the validation step outputs
        dirpath: Directory to save the checkpoints
        filename: Checkpoint filename format
        save_top_k: Number of best checkpoints to keep per strategy
        mode: 'min' or 'max' for loss minimization or maximization
        **kwargs: Additional arguments passed to ModelCheckpoint
    """
    
    def __init__(
        self,
        loss_name: str = "loss",
        dirpath: Optional[str] = None,
        filename: Optional[str] = "{epoch:02d}-{step}",
        save_top_k: int = 1,
        mode: str = "min",
        **kwargs
    ):
        super().__init__(
            dirpath=dirpath, 
            filename=filename, 
            save_top_k=save_top_k, 
            mode=mode,
            **kwargs
        )
        self.loss_name = loss_name
        self.loss_cache = defaultdict(list)
        
        # For tracking and saving per-dataset checkpoints
        self.custom_checkpoints = defaultdict(list)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Store loss values for each dataset during validation."""
        dset_key = list(getattr(trainer, "val_dataloaders").keys())[dataloader_idx]
        self.loss_cache[dset_key].append(
            outputs[self.loss_name].detach().cpu()
        )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Compute dataset losses and delegate to strategy-specific implementations.
        
        Collects and processes loss data, then calls strategy-specific methods
        for checkpoint selection.
        """
        if not self.loss_cache:
            return
            
        # Compute total loss and length for each dataset in this epoch
        epoch_loss_totals, epoch_loss_lengths = {}, {}
        for dset_key, losses in self.loss_cache.items():
            if len(losses) == 0:
                continue
            loss = torch.cat(losses)
            epoch_loss_totals[dset_key] = loss.sum().item()
            epoch_loss_lengths[dset_key] = len(loss)

            # Restart for the next epoch:
            self.loss_cache[dset_key] = []
        
        # Strategy-specific implementation in subclasses
        self._process_dataset_losses(trainer, pl_module, epoch_loss_totals, epoch_loss_lengths)
    
    def _process_dataset_losses(self, trainer, pl_module, epoch_loss_totals, epoch_loss_lengths):
        """Process dataset losses according to the checkpoint strategy."""
        raise NotImplementedError("Subclasses must implement _process_dataset_losses")
    
    def _save_top_k_custom_checkpoints(self, trainer, pl_module, key: str, metric_value: float):
        """Save a checkpoint if it's among the top_k best for the given key."""
        new_checkpoint = {
            'epoch': trainer.current_epoch,
            'value': metric_value,
            'filepath': None  # will be set if we save it
        }
        
        # Combine with existing checkpoints
        all_checkpoints = self.custom_checkpoints.get(key, []) + [new_checkpoint]
        
        # Sort by value (ascending for min mode, descending for max mode)
        sorted_checkpoints = sorted(
            all_checkpoints,
            key=lambda x: x['value'],
            reverse=(self.mode == "max")
        )
        should_save = new_checkpoint in sorted_checkpoints[:self.save_top_k]
        
        if should_save:
            filepath = self._save_custom_checkpoint(trainer, pl_module, key, metric_value)
            new_checkpoint['filepath'] = filepath
            
            # Update the list with only the top_k checkpoints
            top_k_checkpoints = sorted_checkpoints[:self.save_top_k]
            
            # Remove any old checkpoints that didn't make the cut
            for checkpoint in sorted_checkpoints[self.save_top_k:]:
                if checkpoint.get('filepath') and os.path.exists(checkpoint['filepath']):
                    try:
                        os.remove(checkpoint['filepath'])
                    except OSError:
                        pass
            
            # Update the list
            self.custom_checkpoints[key] = top_k_checkpoints
    
    def _save_custom_checkpoint(self, trainer, pl_module, key: str, metric_value: float):
        """Save a checkpoint with a custom key and metric value."""
        # Format filename based on the template
        filename = self.filename or "{epoch:02d}-{step}"
        filename = filename.format(
            epoch=trainer.current_epoch,
            step=trainer.global_step,
        )
        
        # Make the key safe for filenames
        safe_key = key.replace('/', '_').replace(':', '_').replace(' ', '_')
        filename = f"{filename}-{safe_key}-value_{metric_value:.6f}"
        
        # Modify dirpath to include key-specific subdirectory
        dirpath = self.dirpath or trainer.default_root_dir
        custom_dirpath = os.path.join(dirpath, safe_key)
        os.makedirs(custom_dirpath, exist_ok=True)
        
        # Save the checkpoint
        filepath = os.path.join(custom_dirpath, f"{filename}.ckpt")
        self._save_model(filepath, trainer, pl_module)
        return filepath


class SingleDatasetModelCheckpoint(DatasetAwareModelCheckpoint):
    """ModelCheckpoint that saves checkpoints based on individual dataset performance."""
    
    def _process_dataset_losses(self, trainer, pl_module, epoch_loss_totals, epoch_loss_lengths):
        """Process dataset losses using the per-dataset strategy."""
        for dset_key in epoch_loss_totals.keys():
            # Compute mean loss for this specific dataset only
            dataset_loss = epoch_loss_totals[dset_key]
            num_examples = epoch_loss_lengths[dset_key]
            mean_loss = dataset_loss / num_examples if num_examples > 0 else float('inf')
            
            # Use the prefix 'single_' to identify single-dataset checkpoints
            key = f"single_{dset_key}"
            self._save_top_k_custom_checkpoints(trainer, pl_module, key, mean_loss)


class LeaveOneOutModelCheckpoint(DatasetAwareModelCheckpoint):
    """
    ModelCheckpoint that saves checkpoints based on leave-one-out dataset performance.
    
    For each dataset, this callback computes the mean loss on all OTHER datasets
    (leave-one-out) and saves the checkpoint that performs best according to this metric.
    This helps avoid overfitting to specific anomalies in any single dataset.
    """
    
    def _process_dataset_losses(self, trainer, pl_module, epoch_loss_totals, epoch_loss_lengths):
        """Process dataset losses using the leave-one-out strategy."""

        for left_out_key in epoch_loss_totals.keys():         
            # Compute mean loss on all datasets EXCEPT the left-out one
            other_dsets = [k for k in epoch_loss_totals.keys() if k != left_out_key]
            
            if not other_dsets:  # Skip if there's only one dataset
                continue
                
            # Compute weighted mean loss across all other datasets
            total_loss = sum([epoch_loss_totals[k] for k in other_dsets])
            total_examples = sum([epoch_loss_lengths[k] for k in other_dsets])
            loo_loss = total_loss / total_examples if total_examples > 0 else float('inf')
            
            # Use the prefix 'loo_' to identify leave-one-out checkpoints
            key = f"loo_{left_out_key}"
            self._save_top_k_custom_checkpoints(trainer, pl_module, key, loo_loss)


