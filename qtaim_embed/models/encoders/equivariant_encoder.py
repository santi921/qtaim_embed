"""MACE-style equivariant 3D atom encoder.

Minimal equivariant message passing with e3nn: internal features carry
l = 0..lmax irreps, messages are FullyConnectedTensorProduct(h_j, Y(r_ij))
with weights generated from a radial (Bessel) basis, and the readout takes
only the l=0 scalar block - exactly how MACE reads out energy. What MACE
proper adds beyond this (higher body order via message products, lmax=3,
learned radial bases) is deliberately out of scope; call this "MACE-style
equivariant convolution", not MACE.
"""

import torch
import torch.nn as nn
import e3nn.o3 as o3
from torch_geometric.nn.models.dimenet import BesselBasisLayer

from qtaim_embed.models.encoders.neighbors import radius_neighbors


class EquivariantEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 64,
        num_interactions: int = 3,
        num_radial: int = 8,
        lmax: int = 1,
        cutoff: float = 5.0,
        max_z: int = 119,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cutoff = cutoff

        # parity of Y_l is (-1)^l: 0e, 1o, 2e, ...
        self.irreps_h = o3.Irreps(
            "+".join(
                f"{hidden_channels}x{l}{'e' if l % 2 == 0 else 'o'}"
                for l in range(lmax + 1)
            )
        )
        self.irreps_edge = o3.Irreps.spherical_harmonics(lmax)

        self.embedding = nn.Embedding(max_z, hidden_channels, padding_idx=0)
        self.bessel = BesselBasisLayer(num_radial, cutoff)

        self.tensor_products = nn.ModuleList()
        self.radial_nets = nn.ModuleList()
        for _ in range(num_interactions):
            tp = o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_edge, self.irreps_h, shared_weights=False
            )
            self.tensor_products.append(tp)
            self.radial_nets.append(
                nn.Sequential(
                    nn.Linear(num_radial, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, tp.weight_numel),
                )
            )

    def forward_features(self, pos, z, batch=None):
        """Full irreps features (N, irreps_h.dim); used by equivariance tests."""
        pos = pos.to(self.embedding.weight.dtype)
        edge_index, d_ij = radius_neighbors(pos, batch, cutoff=self.cutoff)
        src, dst = edge_index[0], edge_index[1]

        vec = pos[dst] - pos[src]
        sh = o3.spherical_harmonics(
            self.irreps_edge, vec, normalize=True, normalization="component"
        )
        rbf = self.bessel(d_ij)

        h = pos.new_zeros(pos.shape[0], self.irreps_h.dim)
        h[:, : self.hidden_channels] = self.embedding(z)

        for tp, radial in zip(self.tensor_products, self.radial_nets):
            m = tp(h[src], sh, radial(rbf))
            # index_add_ does not promote dtypes; under bf16 autocast the
            # tensor product returns bf16 while h stays float32
            agg = torch.zeros_like(h).index_add_(0, dst, m.to(h.dtype))
            h = h + agg
        return h

    def forward(self, pos, z, batch=None):
        # invariant scalar readout: the 0e block leads the irreps layout
        return self.forward_features(pos, z, batch)[:, : self.hidden_channels]
