"""Symmetric pair-scoring head for bond classification (T3).

Scores an unordered atom pair from the two endpoint embeddings and the
pair distance. Invariant to swapping (i, j) by construction: the embeddings
enter as h_i + h_j and |h_i - h_j|. Emits one logit per pair; apply the
sigmoid in the loss (BCE-with-logits) or at prediction time, never here.
"""

from typing import List, Optional

import torch
import torch.nn as nn



class SymmetricPairHead(nn.Module):
    """MLP([h_i + h_j, |h_i - h_j|, rbf(d_ij), d_ij / r_ref]) -> logit.

    Args:
        in_dim: width of the per-atom embedding h.
        hidden: MLP hidden widths.
        rbf: "bessel" (sinusoidal Bessel with p=5 envelope) or "gaussian".
        rbf_n: number of radial basis functions.
        rbf_cutoff: radial cutoff in Angstrom; set at or above the largest
            candidate distance so the envelope never zeroes a real pair.
        dropout: dropout after each hidden layer.
        use_ratio: include the scalar d_ij / r_ref (r_ref = rcov_i + rcov_j),
            which lets the head express the distance rule exactly.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: List[int] = (256, 128),
        rbf: str = "bessel",
        rbf_n: int = 50,
        rbf_cutoff: float = 6.0,
        dropout: float = 0.1,
        activation: str = "SiLU",
        use_ratio: bool = True,
    ):
        super().__init__()
        assert rbf in ("bessel", "gaussian"), f"rbf must be bessel or gaussian, got {rbf}"
        self.rbf = rbf
        self.rbf_n = int(rbf_n)
        self.rbf_cutoff = float(rbf_cutoff)
        self.use_ratio = bool(use_ratio)

        # Radial-basis constants as buffers: the functional forms in
        # utils/descriptors rebuild them on every call, which costs an
        # allocation plus a host-to-device scalar copy per forward.
        if rbf == "bessel":
            self.register_buffer("rbf_n_idx", torch.arange(1, self.rbf_n + 1, dtype=torch.float32))
            self.register_buffer("rbf_norm", torch.tensor((2.0 / self.rbf_cutoff) ** 0.5))
        else:
            centers = torch.linspace(0.0, self.rbf_cutoff, self.rbf_n)
            spacing = self.rbf_cutoff / (self.rbf_n - 1) if self.rbf_n > 1 else self.rbf_cutoff
            self.register_buffer("rbf_centers", centers)
            self.register_buffer("rbf_gamma", torch.tensor(1.0 / (2.0 * spacing**2)))

        feat_dim = 2 * in_dim + self.rbf_n + (1 if self.use_ratio else 0)
        act = getattr(nn, activation)
        layers: List[nn.Module] = []
        d_in = feat_dim
        for h in hidden:
            layers += [nn.Linear(d_in, h), act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

    def expand_distance(self, d: torch.Tensor) -> torch.Tensor:
        """(P,) distances -> (P, rbf_n) radial features.

        Same math as utils.descriptors.sinusoidal_bessel_rbf (p=5 polynomial
        envelope, zero beyond cutoff) and gaussian_rbf, with constants cached.
        """
        d = d.to(torch.float32).unsqueeze(-1)
        if self.rbf == "bessel":
            d_c = d.clamp_min(1e-8)
            rbf = self.rbf_norm * torch.sin(self.rbf_n_idx * torch.pi * d_c / self.rbf_cutoff) / d_c
            x = d / self.rbf_cutoff
            envelope = (1.0 - 21.0 * x**5 + 35.0 * x**6 - 15.0 * x**7).clamp_min(0.0)
            return rbf * envelope * (d < self.rbf_cutoff)
        return torch.exp(-self.rbf_gamma * (d - self.rbf_centers) ** 2)

    def forward(
        self,
        h: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        d: torch.Tensor,
        r_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h: (N, in_dim) atom embeddings.
            i, j: (P,) candidate endpoints.
            d: (P,) pair distances in Angstrom.
            r_ref: (P,) reference lengths (rcov_i + rcov_j); required when
                use_ratio is True.

        Returns:
            (P,) logits.
        """
        hi, hj = h[i], h[j]
        parts = [hi + hj, (hi - hj).abs(), self.expand_distance(d).to(h.dtype)]
        if self.use_ratio:
            if r_ref is None:
                raise ValueError("r_ref is required when use_ratio=True")
            parts.append((d / r_ref.clamp_min(1e-6)).unsqueeze(-1).to(h.dtype))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x).squeeze(-1)
