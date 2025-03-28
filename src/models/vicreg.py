from typing import Optional
import numpy as np
import h5py
import copy
from tqdm import tqdm

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import L1ADLightningModule
from src.models.quantization import Quantizer

class VICReg(L1ADLightningModule):
    """Contrastive VAE."""
    def __init__(
            self,
            projector: nn.Module,
            feature_blur: nn.Module,
            object_mask: nn.Module,
            lorentz_rotation: nn.Module,
            qdata: Optional[Quantizer] = None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model", "loss", "qdata", "projector", "feature_blur", "object_mask", "lorentz_rotation"])
        self.projector = projector

        # Data quantization:
        self.qdata = qdata or Quantizer(None, None)
        
        # Instantiate augmentation modules
        self.fb1, self.fb2 = copy.deepcopy(feature_blur), copy.deepcopy(feature_blur)
        self.om1, self.om2 = copy.deepcopy(object_mask), copy.deepcopy(object_mask)
        self.lor1, self.lor2 = copy.deepcopy(lorentz_rotation), copy.deepcopy(lorentz_rotation)

    def _extract_batch(self, batch):
        # TODO: Remove this after debugging:
        batch = batch[0]
        return torch.flatten(batch, start_dim=1).to(dtype=torch.float32)
    
    def model_step(self, batch: tuple, batch_idx: int):
        x = self._extract_batch(batch)

        # Quantize data
        x = self.qdata(x)
        
        # Apply augmentations
        x1, x2 = x.clone(), x.clone()
        x1 = self.lor1(self.om1(self.fb1(x1)))
        x2 = self.lor2(self.om2(self.fb2(x2)))
        
        # Get projections
        z1 = self.projector(self.model(x1))
        z2 = self.projector(self.model(x2))

        # Compute loss and return
        loss_inv, loss_var, loss_cov, loss_total = self.loss(z1, z2)
        return {
            "loss": loss_total,
            "loss_inv": loss_inv,
            "loss_var": loss_var,
            "loss_cov": loss_cov,
        }

    
    

# class CombinedAnomalyModel(nn.Module):
#     def __init__(self, backbone, vae_encoder, vae_decoder, total_features):
#         super().__init__()
#         self.backbone = backbone
#         self.vae_encoder = vae_encoder
#         self.vae_decoder = vae_decoder
#         self.total_features = total_features
    
#     def forward(self, x):
#         # Get embedding from backbone
#         embedding = self.backbone(x)
        
#         # Get latent representation from VAE encoder
#         mu, _ = self.vae_encoder(embedding)
        
#         # Reconstruct the embedding
#         recon = self.vae_decoder(mu)
        
#         # Calculate reconstruction error (anomaly score)
#         diff = recon - embedding
#         score = torch.sum(diff * diff, dim=1, keepdim=True)
        
#         return score

# def fast_score(model, data_bkg, bkg_l1_bits, distance_func, data_signal, signal_l1_bits, evaluation_threshold=1):
#     """
#     Calculate performance metrics for anomaly detection
#     """
#     # Calculate background scores
#     bkg_scores = []
#     for i in range(0, len(data_bkg), 1000):
#         batch = torch.tensor(data_bkg[i:i+1000], dtype=torch.float32)
#         with torch.no_grad():
#             scores = model.get_anomaly_score(batch).detach().cpu().numpy()
#         bkg_scores.append(scores)
#     bkg_scores = np.concatenate(bkg_scores)
    
#     # Calculate signal scores for each signal type
#     results = {}
#     for signal_name, signal_data in data_signal.items():
#         sig_scores = []
#         for i in range(0, len(signal_data), 1000):
#             batch = torch.tensor(signal_data[i:i+1000], dtype=torch.float32)
#             with torch.no_grad():
#                 scores = model.get_anomaly_score(batch).detach().cpu().numpy()
#             sig_scores.append(scores)
#         sig_scores = np.concatenate(sig_scores)
        
#         # Calculate ROC AUC
#         from sklearn.metrics import roc_auc_score
        
#         # Create labels (1 for signal, 0 for background)
#         y_true = np.concatenate([np.ones(len(sig_scores)), np.zeros(len(bkg_scores))])
#         y_score = np.concatenate([sig_scores, bkg_scores])
        
#         # Calculate AUC
#         auc = roc_auc_score(y_true, y_score)
#         results[f"AUC_{signal_name}"] = auc
    
#     return results

# def run(config):
#     # Set determinism
#     set_determinism(config)
    
#     # Load data
#     f = h5py.File(config["data_config"]["Processed_data_path"], "r")
    
#     x_train = f['Background_data']['Train']['DATA'][:]
#     x_test = f['Background_data']['Test']['DATA'][:]
    
#     x_train_background = np.reshape(x_train, (x_train.shape[0], -1))
#     x_test_background = np.reshape(x_test, (x_test.shape[0], -1))
    
#     total_num_features = x_test_background.shape[-1]
    
#     scale = f['Normalisation']['norm_scale'][:]
#     bias = f['Normalisation']['norm_bias'][:]
    
#     l1_bits_bkg_test = f['Background_data']['Test']['L1bits'][:]
    
#     # Convert to PyTorch tensors
#     train_tensor = torch.tensor(x_train_background, dtype=torch.float32)
#     test_tensor = torch.tensor(x_test_background, dtype=torch.float32)
    
#     # Load signal data
#     signal_names = list(f['Signal_data'].keys())
#     signal_data_dict = {}
#     signal_l1_dict = {}
    
#     for signal_name in signal_names:
#         x_signal = f['Signal_data'][signal_name]['DATA'][:]
#         x_signal = np.reshape(x_signal, (x_signal.shape[0], -1))
#         l1_bits = f['Signal_data'][signal_name]['L1bits'][:]
        
#         signal_data_dict[signal_name] = x_signal
#         signal_l1_dict[signal_name] = l1_bits
    
#     f.close()
    
#     # Create dataloaders
#     train_dataset = TensorDataset(train_tensor)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config["train"]["Contrastive_VAE"]["batch_size"],
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     test_dataset = TensorDataset(test_tensor)
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=config["train"]["Contrastive_VAE"]["batch_size"],
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # Model parameters
#     vic_encoder_nodes = config["model"]["train_mode"]["Contrastive_VAE"]["encoder_nodes"]
#     projector_features = vic_encoder_nodes[-1] * 4
    
#     vae_encoder_nodes = config["model"]["train_mode"]["Contrastive_VAE"]["vae_nodes"]
#     vae_latent_dim = config["model"]["train_mode"]["Contrastive_VAE"]["vae_latent"][0]
    
#     # Making it symmetric
#     vae_decoder_nodes = [vic_encoder_nodes[-1]] + vae_encoder_nodes.copy()
#     vae_decoder_nodes.reverse()
    
#     # Get ap_fixed parameters
#     ap_fixed_kernel = config["model"]["ap_fixed_kernel"]
#     ap_fixed_bias = config["model"]["ap_fixed_bias"]
#     ap_fixed_act = config["model"]["ap_fixed_activation"]
    
#     # Initialize models
#     backbone = ModelBackbone(
#         nodes=vic_encoder_nodes,
#         ap_fixed_kernel=ap_fixed_kernel,
#         ap_fixed_bias=ap_fixed_bias,
#         ap_fixed_activation=ap_fixed_act
#     )
    
#     projector = ModelProjector(projector_features)
    
#     # Initialize VICReg Lightning module
#     vicreg_model = VICRegModule(
#         backbone=backbone,
#         projector=projector,
#         config=config,
#         num_features=projector_features,
#         batch_size=config["train"]["Contrastive_VAE"]["batch_size"]
#     )
    
#     # Setup MLflow logger
#     mlflow_logger = MLFlowLogger(
#         experiment_name="vicreg_vae",
#         tracking_uri=mlflow.get_tracking_uri()
#     )
    
#     # Train VICReg model
#     lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
#     trainer = pl.Trainer(
#         max_epochs=config["train"]["Contrastive_VAE"]["epochs_contrastive"],
#         logger=mlflow_logger,
#         callbacks=[lr_monitor],
#         deterministic=True
#     )
    
#     trainer.fit(vicreg_model, train_loader)
    
#     # After VICReg training, prepare data for VAE
#     backbone = vicreg_model.backbone
#     backbone.eval()
    
#     # Process data through the backbone for VAE training
#     train_embeddings = []
#     with torch.no_grad():
#         for batch in tqdm(train_loader, desc="Processing train data for VAE"):
#             embeddings = backbone(batch[0])
#             train_embeddings.append(embeddings)
#     train_embeddings = torch.cat(train_embeddings, dim=0)
    
#     test_embeddings = []
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="Processing test data for VAE"):
#             embeddings = backbone(batch[0])
#             test_embeddings.append(embeddings)
#     test_embeddings = torch.cat(test_embeddings, dim=0)
    
#     # Process signal data through backbone
#     signal_embeddings = {}
#     for signal_name, signal_data in signal_data_dict.items():
#         signal_tensor = torch.tensor(signal_data, dtype=torch.float32)
#         signal_loader = DataLoader(
#             TensorDataset(signal_tensor),
#             batch_size=1000,
#             shuffle=False
#         )
        
#         signal_emb = []
#         with torch.no_grad():
#             for batch in signal_loader:
#                 embeddings = backbone(batch[0])
#                 signal_emb.append(embeddings)
#         signal_embeddings[signal_name] = torch.cat(signal_emb, dim=0).detach().cpu().numpy()
    
#     # Create VAE dataloaders
#     vae_train_dataset = TensorDataset(train_embeddings)
#     vae_train_loader = DataLoader(
#         vae_train_dataset,
#         batch_size=config["train"]["Contrastive_VAE"]["batch_size"],
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     vae_test_dataset = TensorDataset(test_embeddings)
#     vae_test_loader = DataLoader(
#         vae_test_dataset,
#         batch_size=config["train"]["Contrastive_VAE"]["batch_size"],
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # Initialize VAE models
#     vae_encoder = VAE_Encoder(
#         nodes=vae_encoder_nodes,
#         feature_size=vae_latent_dim,
#         ap_fixed_kernel=ap_fixed_kernel,
#         ap_fixed_bias=ap_fixed_bias,
#         ap_fixed_activation=ap_fixed_act
#     )
    
#     vae_decoder = VAE_Decoder(
#         nodes=vae_decoder_nodes,
#         ap_fixed_kernel=ap_fixed_kernel,
#         ap_fixed_bias=ap_fixed_bias,
#         ap_fixed_activation=ap_fixed_act
#     )
    
#     # Initialize VAE Lightning module
#     vae_model = VAEModule(
#         encoder=vae_encoder,
#         decoder=vae_decoder,
#         config=config,
#         total_features=train_embeddings.shape[1]
#     )
    
#     # Train VAE model
#     vae_trainer = pl.Trainer(
#         max_epochs=config["train"]["Contrastive_VAE"]["epochs_vae"],
#         logger=mlflow_logger,
#         callbacks=[lr_monitor],
#         deterministic=True
#     )
    
#     vae_trainer.fit(vae_model, vae_train_loader