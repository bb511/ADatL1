# Repository Inventory

This repository supports anomaly-detection experiments for AD@L1 trigger studies.
The main research focus is:

- validating CAP as a label-free model evaluation and model-selection metric;
- exploring algorithms and architectures for the L1AD trigger anomaly pipeline.

Stack: `uv`, Hydra, PyTorch Lightning, PyTorch, TorchMetrics, Optuna, MLflow/WandB,
POT, optional HGQ/Keras quantization, and private `capmetric`.

## Shared Contract

| Item | Role | Key Parameters / Outputs |
|---|---|---|
| `ADLightningModule` | Base Lightning module for all algorithms. Subclasses implement `model_step(batch)`. | `optimizer`, `scheduler`, `target_rate`, `base_rate`; returns scalar `loss` plus per-event tensors such as `ascore/full`, `loss/full`, or `loss/total/full`. |
| Operating threshold | Converts score distributions into L1-style fixed-rate selections. | If `base_rate` is set: `fpr = target_rate / base_rate`; otherwise `fpr = target_rate`. Threshold is `quantile(score_normal, 1 - fpr)`. |
| Label convention | Shared by physics and image datamodules. | `y = 0`: normal; `y > 0`: signal/anomaly; `y < 0`: background or shifted-normal simulation. |
| Loader convention | Validation/test dataloaders are dictionaries. | Current normal loader name is `normal`; some evaluation callbacks still accept legacy `main_test`. |

## Data

### Datamodules

| Datamodule | Purpose | Key Parameters | Scientific Notes |
|---|---|---|---|
| `L1ADDataModule` | Physics/L1 trigger data from raw inputs to ML-ready tensors. | `train_features`, `data_extractor`, `data_processor`, `data_normalizer`, `data_mlready`, `data_awkward2torch.nconst`, `batch_size`, `max_val_batches`. | Training uses zero-bias normal data. Auxiliary validation/test loaders contain signals and simulated backgrounds. |
| `CIFAR10DataModule` | Small image anomaly benchmark. | `normal_classes`, `signal_classes`, `background_classes`, `val_fraction`, `reference_fraction`, `normalize`, `stats_file`. | Normalization statistics are computed on normal-class training data only. Default: class `0` normal, classes `1..9` anomalies. |
| `RobustADDataModule` | Shifted image anomaly benchmark from `AmazonScience/RobustAD`. | `subset`, `image_size`, `val_fraction`, `test_fraction`, `normalize`. | Supports `pcb`, `metal_parts`, `piled_bags`; exposes shifted normal and shifted anomaly loaders. Useful for CAP under distribution shift. |
| `SyntheticL1ADDataModule` | In-memory L1-like smoke-test data. | `n_features`, `n_train`, `n_val`, `n_test`, `batch_size`, `seed`, `paper_aliases`. | `paper_aliases=true` exposes paper L1 dataset names over synthetic tensors. Not a physics benchmark. |
| `SyntheticImageADDataModule` | In-memory image smoke-test data. | `image_size`, `n_train`, `n_val`, `n_test`, `shifted_domains`, `n_cifar_signals`. | Exposes CIFAR-style `reference_normal`/`1..9` and RobustAD-style `shifted_normal_all`/`shifted_anomaly_*` loaders. |

### L1 Data Cards

| Config | Feature Contract | Key Parameters / Differences |
|---|---|---|
| `data=default` | Broad 2024E L1 object set. | Default extractor/processor, unnormalized normalizer; useful as a raw-scale template. |
| `data=axov4` | 57-feature AXO-v4 style data. | Minimal extractor, AXO robust normalizer, minimal object multiplicities; muons/jets/egammas/MET. |
| `data=axov5` | Same data contract as AXO-v4. | Separate card so AXO-v5 models can vary independently of data preparation. |
| `data=axov6` | 2025 E+G data with AXO-style objects. | Robust normalizer on newer eras; used for era-transfer checks. |
| `data=basis` | 117-feature current physics basis. | Adds taus and replaces MET with FET; uses `minimal_tau_fet`, `default_tau_FET`, `extended_taus`. |

## Algorithms

### Vector / L1 Algorithms

| Algorithm | Objective / Score | Key Parameters | Configs |
|---|---|---|---|
| `AE` | Trains masked Huber reconstruction; scores with masked MSE `s(x)=mean_j m_j(x_j-xhat_j)^2/sum_j m_j`. | `encoder.in_dim`, `encoder.nodes`, `decoder.nodes`, `delta`, `input_noise_std`, `target_rate`, `base_rate`. | `algorithm=ae` |
| `VAE` | Trains `L = L_rec + lambda_KL KL(q(z\|x) || N(0,I))`; operational score is raw per-event KL. | latent dim `encoder.nodes[-1]`, `clamp_zlogvar_range`, `kl_scale`, `kl_warmup_frac`, `mask`, `features`, `ckpt`. | `algorithm=vae` |
| AXO VAE | VAE with cylindrical reconstruction loss on selected `pT`, derived `pz=pT sinh(eta)`, and MET `phi`. | 57-feature object map, `loss.scale`, `loss.kl_scale`, Lion LR, CDRW restart schedule. | `axov4`, `axov5_vae` |
| `RVAE` | VAE backed by composable `MultiLoss`; default active loss is reconstruction + KL. | `loss.list_losses`, component `scale`, component `reduction`, latent dim, label availability for MI/alignment. | `rvae` |
| `DeepSVDD` | Scores by latent-center distance `s(x)=||f(x)-c||_2^2`; optional soft boundary. | `center_init_method`, `nu`, `soft_boundary`, `weight_decay`, `encoder.nodes`. | `svdd` |
| `RealNVP` | Normalizing flow; score is negative log-likelihood `s(x)=-log p_theta(x)`. | `flow.input_dim`, `n_flows`, `hidden_dim`, `n_hidden_layers`, `noise_scale`, `scale_clamp`. | `realnvp` |
| `DeepSetsAE` | Per-object DeepSets encoder plus dense decoder; AE-style reconstruction score. | `object_dims`, `object_phi_nodes`, `rho_nodes`, `pooling`, `add_counts`, `delta`, `input_noise_std`. | `dsae` |
| `DeepSetsVAE` | DeepSets event encoder plus VAE latent head; score is raw KL. | DeepSets params plus `rho_nodes[-1]`, `clamp_zlogvar_range`, `kl_scale`, `kl_warmup_frac`. | `dsvae` |
| `VICReg` | Self-supervised representation learning with two augmented views. Loss: invariance + variance + covariance. | `feature_blur.*`, `object_mask.prob`, `lorentz_rotation.prob`, `model.out_dim`, `projector.out_dim`, `inv_coef`, `rvar_coef`, `rcov_coef`. | `vicreg`, `qvicreg`, `axov5_vicreg` |
| VICReg-feature VAE | VAE trained on frozen VICReg representation. | `algorithm.features.ckpt_path`, `features.attr=model`, representation dimension, optional quantized encoder/decoder. | `vicreg_vae`, `vicreg_qvae` |
| `WNAE` | Reconstruction energy model with Langevin negative samples. `L = E_pos + W(x_pos,x_neg) + mean(E_pos)-mean(E_neg)`. | `sampling`, `replay`, `replay_ratio`, `buffer_size`, `initial_dist`, `clip`, `spherical`, sampler `n_steps/step_size/noise_scale/temperature`. | `wnae` |

### Image Algorithms

| Algorithm | Purpose | Key Parameters | Configs |
|---|---|---|---|
| `ImageAE` | Image reconstruction baseline for CIFAR-10/RobustAD. | `in_channels`, `input_size`, conv `nodes`, `strides`, `batchnorm`, `delta`, `input_noise_std`. | `image_ae` |
| `ImageVAE` | Image VAE; score is raw KL. | conv `nodes`, latent dim, `strides`, `clamp_zlogvar_range`, `kl_scale`, `kl_warmup_frac`. | `image_vae` |
| `ImageDeepSVDD` | Image encoder with latent-center distance score. | conv encoder params, `center_init_method`, `nu`, `soft_boundary`, `weight_decay`. | `image_svdd` |
| Image `RealNVP` | Flow over flattened images. | `input_dim=3*H*W`, `n_flows`, `hidden_dim`, `scale_clamp`, `noise_scale`. | `image_realnvp` |

### Quantized Models

| Model | Purpose | Key Parameters |
|---|---|---|
| `qvae` | HGQ/Keras quantized VAE. | `nodes`, `q_type`, `place`, `i0`, `f0`, `trainable`, `ic`, `fc`, `ebops`. |
| `qvicreg` | HGQ/Keras quantized VICReg encoder. | Same quantization params plus VICReg augmentation and loss coefficients. |
| `vicreg_qvae` | Quantized VAE on frozen VICReg features. | Requires `algorithm.features.ckpt_path`; also requires `quant` extra and `KERAS_BACKEND=torch`. |

## Components

| Component | Purpose | Key Parameters |
|---|---|---|
| `MLP`, `Encoder`, `Decoder` | Dense vector networks. | `in_dim`, `nodes`, `out_dim`, `activation`, `batchnorm`, `final_activation`, initializers. |
| `VariationalEncoder` | Dense encoder producing `z_mean`, `z_log_var`, and sampled `z`. | `nodes[-2]` hidden dim, `nodes[-1]` latent dim, `clamp_zlogvar_range`. |
| `DeepSetsEncoder` | Permutation-invariant object encoder: `z = rho(concat_t pool_j phi_t(x_tj))`. | `object_dims`, `object_phi_nodes`, `rho_nodes`, `pooling`, `add_counts`. |
| `ImageEncoder` / `ImageDecoder` | Convolutional image networks. | `nodes`, `strides`, `input_size`, `in_channels/out_channels`, `batchnorm`, `final_activation`. |
| RealNVP flow layers | Alternating affine coupling layers. | `n_flows`, `hidden_dim`, `n_hidden_layers`, `activation`, `scale_clamp`. |
| VICReg augmentations | Build two stochastic physics-aware views. | `FastFeatureBlur(prob,magnitude,strength)`, `FastObjectMask(prob)`, `FastLorentzRotation(prob)`. |
| Input masking | Random or multiplicity-based object masking. | `object_probs` or `percentile`, `mask_value`, `training_only`. |
| MCMC sampler | Langevin sampler for WNAE negatives. | `n_steps`, `step_size`, `noise_scale`, `temperature`, `clip`, `clip_grad`, `mh`. |
| `FeaturesFromCkpt` | Frozen checkpoint feature extractor. | `litmodule_cls`, `ckpt_path`, `attr`. |

## Losses

| Loss | Formula / Role | Key Parameters |
|---|---|---|
| `MSEReconstructionLoss` | Per-event masked MSE reconstruction energy. | `scale`, `reduction`, optional `mask`. |
| `HuberReconstructionLoss` | SmoothL1 reconstruction, robust to tails. | `delta`, `scale`, `reduction`, optional `mask`. |
| `CylPtPzReconstructionLoss` | AXO-specific reconstruction in `(pT, pz, MET phi)` space. | Injected `object_feature_map`, `scale`, `reduction`. |
| `KLDivergenceLoss` | `-0.5 sum(1 + logvar - mu^2 - exp(logvar))`. | `scale`, `reduction`; can be annealed by VAE warmup. |
| `ClassicVAELoss` | `scale * (reco_scale*L_rec + kl_scale*KL)`. | `scale`, `reco_scale`, `kl_scale`, `reduction`. |
| `MultiLoss` | Composes named losses and returns `loss/total`. | `list_losses`, component configs, outer `scale`. |
| `SVDDLoss` | One-class or soft-boundary latent distance loss. | `nu`, `soft_boundary`, `weight_decay`, `scale`. |
| `VICRegLoss` | Invariance, variance floor, covariance decorrelation. | `inv_coef`; effective `var_coef=inv_coef*rvar_coef`, `cov_coef=inv_coef*rcov_coef`. |
| `BetaVAELoss` | Capacity-controlled beta-VAE regularization. | `beta`, `gamma`, `max_capacity`, `capacity_leadin`, `distance=kl/mmd`. |
| `DIPVAELoss` | Penalizes latent covariance diagonal/off-diagonal structure. | `lambda_diag`, `lambda_offdiag`. |
| `MMDLoss` | Matches latent distribution to prior with kernel MMD. | `kernel`, `kernel_bandwidth`, `prior`, `block_size`, `max_samples`, `use_cpu`. |
| `MILoss` | Minimizes `-I(Z;S)` using labels. | `eps`, `scale`; requires meaningful `y`. |
| `AlignmentLoss` | Pushes or clusters latent codes by zero-bias/background labels. | `strategy`, `distance`, `margin`. |
| `GradientPenaltyLoss` | Penalizes Jacobian norm away from `target_norm`. | `target_norm`, `scale`. |
| `WassersteinLoss` | OT cost between positive and negative batches using POT. | ground metric order `p`, `scale`. |
| `NAE` | Energy gap `mean(E_pos)-mean(E_neg)`. | `scale`. |

## Optimization

| Item | Role | Key Parameters |
|---|---|---|
| AdamW | Default optimizer in `configs/train.yaml`. | `lr`, `weight_decay`, `betas`, `eps`. |
| Lion | Sign-momentum optimizer used by AXO/WNAE configs. | `lr`, `betas`, `weight_decay`. |
| `LinearWarmup` | Scalar warmup, mainly for KL scale. | `final_value`, `warmup_frac`, `total_steps`. |
| `CosineWithWarmup` | Step warmup then cosine decay, used by VICReg. | `warmup_ratio`, `min_lr_ratio`, `warmup_start_ratio`, `total_steps`. |
| `CDRW` | Cosine decay with restarts and warmup. | `lr0`, `s0`, `t_mul`, `m_mul`, `alpha`, `warmup_epochs`. |
| Optuna sweeps | Multi-objective HPO through Hydra. | Search spaces cover architecture widths, learning rate, KL scale, SVDD `nu`, RealNVP depth, VICReg augmentations/loss weights. |

## Evaluation And Metrics

| Metric / Callback | Purpose | Key Parameters | Selection Use |
|---|---|---|---|
| Training anomaly efficiency | Compute validation threshold on normal data and apply it to aux datasets. | `output_name`, `target_rates`, `base_rate`, `beta`. | Signal-aware checkpointing. |
| Evaluation anomaly efficiency | Same logic on saved checkpoints. | `output_name`, `target_rates`, `pure_thres`, `cvar_summary`, `ds`. | Signal-aware Optuna/test reporting. |
| CAP | Label-free score-distribution capacity: approximately `max_beta log Z12 - log Z1 - log Z2`. | `dataset_1`, `dataset_2`, `pairing_type`, `normalization_type`, `energy_type`, `regularization_type`, `beta0`, `lr`, `n_epochs`, `batch_size`. | Agnostic model selection. |
| Rank correlation | Stored alongside CAP for paired score order agreement. | Determined by CAP pairing. | Diagnostic for CAP behavior. |
| Threshold drift | Split normal scores into calibration/evaluation and measure fixed-rate transfer. | `output_name`, `target_rates`, `base_rate`, `calibration_fraction`, `split_seed`, `beta`. | Agnostic stability metric. |
| Wasserstein score distance | Compare score distributions between two datasets. | `output_name`, `dataset_1`, `dataset_2`, `apply_log1p`, `beta`. | Agnostic shift/separation metric. |
| Output summary | Mean/scalar output summaries by dataset/checkpoint. | `output_name`, `ds`. | Secondary metrics and diagnostics. |
| Output histogram | Histograms of selected output tensors. | `output_name`, `bins`, `warmup_batches`, `ckpts`, `ds`. | Distribution inspection. |
| Reconstruction plots | Compare input and reconstruction. | `output_name=reconstructed_data`, `datasets`, `warmup_batches`, `datamodule`. | AE/VAE diagnostics. |
| KNN AUPRC | Representation evaluation, mainly for VICReg. | `output_name=vicreg_rep_data`, `k`, `reference_sample_size`, `skip_ds`. | Representation model selection. |
| ROC/AUROC | Normal-vs-auxiliary ROC curves. | `metric_name`, reference normal loader. | Diagnostic, less central than fixed-rate metrics. |
| Loss callback | Mean loss by dataset and checkpoint. | `loss_name`, `skip_ds`, `name`. | Secondary Optuna metric. |
| Pileup rate | Background rate versus pileup after fixed normal threshold. | `target_rates`, `base_rate`, `output_name`, pileup fields. | Physics robustness diagnostic. |

### CAP Options

| Option Type | Supported Values | Notes |
|---|---|---|
| Pairing | `none`, `random`, `absolute`, `cdf`, label-based pairing | `cdf` pairs by empirical percentile; useful for distributional comparisons. |
| Normalization | `none`, `minmax`, `sigmoid`, `softmax`, `rank`, `rank_mid`, `log_sigmoid` | `rank`/`rank_mid` are robust to heavy-tailed scores. |
| Energy | `baseline`, `focal`, `exponential`, `margin`, `contrastive`, `adaptive` | Controls how normalized scores define binary assignment energies. |
| Regularization | `none`, threshold variants, smooth variants, `percentile` | Constrains or reshapes feasible anomaly assignments. |

## Checkpointing

| Strategy / Criterion | Purpose | Key Parameters |
|---|---|---|
| `SingleDatasetModelCheckpoint` | Save checkpoints per monitored dataset. | `monitor`, `ds`, `criterion`, `dirpath`. |
| `LeaveOneOutModelCheckpoint` | Save by aggregate score excluding each dataset in turn. | `monitor`, `criterion`. |
| `LeaveKOutModelCheckpoint` | Save by aggregate over selected datasets. | `selected_datasets`, `monitor`, `criterion`. |
| `Min` / `Max` | Standard top-k metric selection. | `top_k`. |
| `Stable` | Select when a metric changes by less than a threshold for a patience window. | `top_k`, `threshold`, `patience`. |
| `Last` | Save final checkpoint. | `top_k`. |

## Experiments

| Group | Scope | Key Selection Logic |
|---|---|---|
| `configs/experiment/demo` | Short CPU smoke runs for CIFAR and synthetic L1. | No checkpointing/evaluator; verifies config and training loops. |
| `configs/experiment/physics` | Main L1 trigger experiments for AE/VAE/SVDD/RealNVP/DeepSets. | Signal-aware variants optimize efficiency; agnostic variants optimize CAP/drift/Wasserstein. |
| `configs/experiment/cifar10` | Image benchmark experiments. | Checks CAP/model-selection behavior outside L1 data. |
| `configs/experiment/robustad` | Shifted-domain image anomaly benchmark. | Useful for CAP under source-normal vs shifted-normal comparison. |

### Verified Demo Commands

```bash
uv run python src/train.py experiment=demo/cifar10_ae
uv run python src/train.py experiment=demo/l1_vae
uv run python src/train.py experiment=demo/l1_vicreg
uv run python src/train.py experiment=demo/l1_wnae
uv run python src/train.py experiment=demo/l1_rvae
KERAS_BACKEND=torch uv run python src/train.py experiment=demo/l1_vae algorithm=qvae
KERAS_BACKEND=torch uv run python src/train.py experiment=demo/l1_vicreg algorithm=qvicreg
```

## Runtime And Dependencies

| Area | Support |
|---|---|
| Environment | Managed by `uv` with `pyproject.toml` and `uv.lock`. |
| Core ML | PyTorch, PyTorch Lightning, TorchMetrics, torchvision. |
| Config/HPO | Hydra, Hydra Optuna sweeper, Hydra submitit launcher. |
| Physics/data | awkward, uproot, h5py, fastparquet, pyarrow, `adl1t-datamaker`. |
| Metrics | POT for optimal transport, internal CAP, private `capmetric`. |
| Logging | `none`, CSV, MLflow, WandB, MLflow+WandB. |
| Optional extras | `quant` for Keras/HGQ/hls4ml; `wandb` for WandB logging. |

## Conditional Support

| Feature | Requirement |
|---|---|
| `vicreg_vae` | Set `algorithm.features.ckpt_path` to a trained VICReg checkpoint. |
| `vicreg_qvae` | Same as `vicreg_vae`, plus quantization dependencies. |
| `qvae`, `qvicreg` | Install/use `quant` extra and run with `KERAS_BACKEND=torch`. |
| CAP templates | Supply `dataset_1`, `dataset_2`, and `pairing_type`. |
| Synthetic L1 demos | Smoke tests only; not physics performance evidence. |
