import torch
import torch.nn as nn
import torch.nn.functional as F


class NeFLIM(nn.Module):
    """Shared INR backbone with separate fluorescence-yield and lifetime heads."""

    def __init__(
        self,
        input_dim: int = 60,
        layer_dim: int = 256,
        activation_fn: str = "relu",
        yield_min: float = 0.0,
        yield_max: float = 1.0,
        tau_min: float = 0.05,
        tau_max: float = 3.0,
    ) -> None:
        super().__init__()
        if yield_max <= yield_min:
            raise ValueError("yield_max must be greater than yield_min")
        if tau_max <= tau_min or tau_min <= 0:
            raise ValueError("Require 0 < tau_min < tau_max")

        self.yield_min = float(yield_min)
        self.yield_max = float(yield_max)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        if activation_fn == "relu":
            self.activation = F.relu
        elif activation_fn == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation_fn == "silu":
            self.activation = F.silu
        else:
            raise NotImplementedError(f"Unsupported activation: {activation_fn}")

        self.layer_0 = nn.Linear(input_dim, layer_dim)
        self.layer_1 = nn.Linear(layer_dim, layer_dim)
        self.layer_2 = nn.Linear(layer_dim, layer_dim)
        self.layer_3 = nn.Linear(layer_dim, layer_dim)
        self.layer_4 = nn.Linear(layer_dim, layer_dim)
        self.layer_5 = nn.Linear(input_dim + layer_dim, layer_dim)
        self.layer_6 = nn.Linear(layer_dim, layer_dim)
        self.layer_7 = nn.Linear(layer_dim, layer_dim // 2)
        self.yield_head = nn.Linear(layer_dim // 2, 1)
        self.tau_head = nn.Linear(layer_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.activation(self.layer_0(x))
        feat = self.activation(self.layer_1(feat))
        feat = self.activation(self.layer_2(feat))
        feat = self.activation(self.layer_3(feat))
        feat = self.activation(self.layer_4(feat))
        feat = self.activation(self.layer_5(torch.cat((x, feat), dim=-1)))
        feat = self.activation(self.layer_6(feat))
        feat = self.activation(self.layer_7(feat))

        yield_pred = self.yield_min + (self.yield_max - self.yield_min) * torch.sigmoid(self.yield_head(feat))
        tau_pred = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.tau_head(feat))
        return yield_pred, tau_pred
