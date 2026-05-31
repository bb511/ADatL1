import torch

class BernoulliMILoss(torch.nn.Module):

    def __init__(self, temperature: float = 6.0, use_quantized_sigmoid: bool = False,
        bits_bernoulli_sigmoid: int = 8, eps: float = 1e-20) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_quantized_sigmoid = use_quantized_sigmoid
        self.bits_bernoulli_sigmoid = bits_bernoulli_sigmoid
        self.eps = eps

        if self.use_quantized_sigmoid:
            raise NotImplementedError(
                "Quantized sigmoid is not implemented yet in this PyTorch version."
            )

    def forward(self, latent: torch.Tensor, sensitive: torch.Tensor) -> torch.Tensor:
        latent = latent.to(dtype=torch.float64)

        if sensitive.dim() > 1:
            sensitive = sensitive[:, 0]

        sensitive = sensitive.to(device=latent.device, dtype=torch.long)

        H_L_n = self.get_h_bernoulli(latent)

        conditional_entropy = torch.zeros(
            (), device=latent.device, dtype=torch.float64
        )

        unique_sensitive = torch.unique(sensitive)

        for value in unique_sensitive:
            H_L_n_si, norm_si = self.compute_for_value(value, latent, sensitive)
            conditional_entropy = conditional_entropy + norm_si * H_L_n_si

        MI = H_L_n - conditional_entropy
        MI = torch.nan_to_num(MI, nan=0.0, posinf=0.0, neginf=0.0)

        return MI.to(dtype=torch.float32)


 # ----------------------------------------
 # Helpers
    def get_theta(self, x: torch.Tensor) -> torch.Tensor:
        std = 1.0
        return torch.sigmoid(self.temperature * x / std)

    def log2(self, x: torch.Tensor) -> torch.Tensor:
        numerator = torch.log(x + self.eps)
        denominator = torch.log(
            torch.tensor(2.0, device=x.device, dtype=x.dtype)
        )
        return numerator / denominator
    
    def get_h_bernoulli(self, tensor: torch.Tensor) -> torch.Tensor:
        theta = torch.mean(self.get_theta(tensor), dim=0)

        entropy_per_unit = (-(1.0 - theta) * self.log2(1.0 - theta) - theta * self.log2(theta))

        return torch.sum(entropy_per_unit)
    
    def compute_for_value(self, value: torch.Tensor, latent: torch.Tensor, sensitive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = sensitive == value
        latent_i = latent[mask]

        if latent_i.numel() == 0:
            H_L_n_si = torch.zeros(
                (), device=latent.device, dtype=torch.float64
            )
        else:
            H_L_n_si = self.get_h_bernoulli(latent_i)

        count_i = torch.tensor(
            latent_i.shape[0], device=latent.device, dtype=torch.float64
        )

        batch_size = torch.tensor(
            latent.shape[0], device=latent.device, dtype=torch.float64
        )

        norm_si = count_i / batch_size

        return H_L_n_si, norm_si
