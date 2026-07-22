import torch
import torch.nn as nn
import torch.nn.functional as F


class NeFLIM(nn.Module):
    """Shared INR trunk with separate fluorescence-yield and lifetime heads.

    Both outputs are constrained to user-specified physical ranges with a
    sigmoid transform. Coordinates are expected to be encoded before being
    passed to this module, matching the existing NeuFMT pipeline.
    """

    def __init__(
        self,
        input_dim: int = 60,
        layer_dim: int = 256,
        activation_fn: str = "relu",
        yield_min: float = 0.0,
        yield_max: float = 1.5,
        lifetime_min: float = 0.01,
        lifetime_max: float = 3.0,
    ) -> None:
        super().__init__()

        if yield_max <= yield_min:
            raise ValueError("yield_max must be larger than yield_min")
        if lifetime_max <= lifetime_min:
            raise ValueError("lifetime_max must be larger than lifetime_min")

        self.yield_min = float(yield_min)
        self.yield_max = float(yield_max)
        self.lifetime_min = float(lifetime_min)
        self.lifetime_max = float(lifetime_max)

        if activation_fn == "relu":
            self.activation = F.relu
        elif activation_fn == "leaky_relu":
            self.activation = F.leaky_relu
        elif activation_fn == "silu":
            self.activation = F.silu
        elif activation_fn == "tanh":
            self.activation = torch.tanh
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

        # Separate heads prevent the very different yield/lifetime scales from
        # competing in the last linear layer.
        self.yield_head = nn.Linear(layer_dim // 2, 1)
        self.lifetime_head = nn.Linear(layer_dim // 2, 1)

    @staticmethod
    def _bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        return lower + (upper - lower) * torch.sigmoid(raw)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.activation(self.layer_0(x))
        feat = self.activation(self.layer_1(feat))
        feat = self.activation(self.layer_2(feat))
        feat = self.activation(self.layer_3(feat))
        feat = self.activation(self.layer_4(feat))
        feat = self.activation(self.layer_5(torch.cat((x, feat), dim=-1)))
        feat = self.activation(self.layer_6(feat))
        feat = self.activation(self.layer_7(feat))

        yield_pred = self._bounded(
            self.yield_head(feat), self.yield_min, self.yield_max
        )
        lifetime_pred = self._bounded(
            self.lifetime_head(feat), self.lifetime_min, self.lifetime_max
        )
        return yield_pred, lifetime_pred
