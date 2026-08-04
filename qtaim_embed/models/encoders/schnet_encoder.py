"""SchNet-style invariant 3D atom encoder.

Adapter around PyG's reference SchNet blocks (GaussianSmearing,
InteractionBlock). Differs from torch_geometric.nn.models.SchNet in exactly
two ways: the radius graph comes from our torch-only neighbor module (no
torch_cluster), and the output is per-atom embeddings rather than a pooled
per-molecule scalar. The interaction loop mirrors the reference model, which
is what the weight-copy parity test in tests/test_encoder_parity.py checks.
"""

import torch
import torch.nn as nn
from torch_geometric.nn.models.schnet import GaussianSmearing, InteractionBlock

from qtaim_embed.models.encoders.neighbors import radius_neighbors


class SchNetEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 64,
        num_interactions: int = 3,
        num_gaussians: int = 50,
        cutoff: float = 5.0,
        num_filters: int = None,
        max_z: int = 119,
    ):
        super().__init__()
        num_filters = num_filters or hidden_channels
        self.cutoff = cutoff
        self.embedding = nn.Embedding(max_z, hidden_channels, padding_idx=0)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.interactions = nn.ModuleList(
            [
                InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
                for _ in range(num_interactions)
            ]
        )

    def forward(self, pos, z, batch=None):
        pos = pos.to(self.embedding.weight.dtype)
        edge_index, edge_weight = radius_neighbors(pos, batch, cutoff=self.cutoff)
        edge_attr = self.distance_expansion(edge_weight)
        h = self.embedding(z)
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        return h
