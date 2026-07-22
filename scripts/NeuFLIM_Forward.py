import torch
import torch.nn as nn


class TimeDomainFLIMForward(nn.Module):
    """Differentiable lifetime-FMT forward model with precomputed operators.

    Shapes
    ------
    excitation : [T, Nh, Ns]
    A_em       : [Nh, Nh], emission state propagation operator
    B_em       : [Nh, Nh], nodal fluorescence source -> emission field operator
    P          : [Nd, Nh], detector projection operator
    yield_map  : [Nh] or [Nh, 1]
    lifetime   : [Nh] or [Nh, 1]
    output     : [T, Nd, Ns]

    The recurrence is
        R_k   = decay * R_{k-1} + gain * Phi_x,k
        Phi_m = A_em * Phi_m + B_em * (yield * R_k)
        y_k   = P * Phi_m
    """

    def __init__(
        self,
        excitation: torch.Tensor,
        A_em: torch.Tensor,
        B_em: torch.Tensor,
        P: torch.Tensor,
        dt: float,
        lifetime_scheme: str = "exact",
        tau_epsilon: float = 1e-8,
    ) -> None:
        super().__init__()

        if excitation.ndim != 3:
            raise ValueError("excitation must have shape [T, Nh, Ns]")
        if A_em.ndim != 2 or A_em.shape[0] != A_em.shape[1]:
            raise ValueError("A_em must be square [Nh, Nh]")
        if B_em.shape != A_em.shape:
            raise ValueError("B_em must have the same shape as A_em")
        if P.ndim != 2 or P.shape[1] != A_em.shape[0]:
            raise ValueError("P must have shape [Nd, Nh]")
        if excitation.shape[1] != A_em.shape[0]:
            raise ValueError("excitation node dimension does not match A_em")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if lifetime_scheme not in {"exact", "euler"}:
            raise ValueError("lifetime_scheme must be 'exact' or 'euler'")

        self.register_buffer("excitation", excitation)
        self.register_buffer("A_em", A_em)
        self.register_buffer("B_em", B_em)
        self.register_buffer("P", P)

        self.dt = float(dt)
        self.lifetime_scheme = lifetime_scheme
        self.tau_epsilon = float(tau_epsilon)

    def forward(
        self, yield_map: torch.Tensor, lifetime: torch.Tensor
    ) -> torch.Tensor:
        yield_map = yield_map.reshape(-1)
        lifetime = lifetime.reshape(-1)

        nh = self.excitation.shape[1]
        ns = self.excitation.shape[2]
        if yield_map.numel() != nh or lifetime.numel() != nh:
            raise ValueError("yield_map and lifetime must contain Nh values")

        tau = torch.clamp(lifetime, min=self.tau_epsilon)
        decay = torch.exp(-self.dt / tau)
        if self.lifetime_scheme == "exact":
            gain = 1.0 - decay
        else:
            gain = self.dt / tau

        state_dtype = self.excitation.dtype
        state_device = self.excitation.device
        R = torch.zeros((nh, ns), dtype=state_dtype, device=state_device)
        phi_em = torch.zeros_like(R)
        measurements = []

        decay = decay[:, None]
        gain = gain[:, None]
        yield_map = yield_map[:, None]

        for phi_ex in self.excitation:
            R = decay * R + gain * phi_ex
            fluorescence_source = yield_map * R
            phi_em = self.A_em @ phi_em + self.B_em @ fluorescence_source
            measurements.append(self.P @ phi_em)

        return torch.stack(measurements, dim=0)
