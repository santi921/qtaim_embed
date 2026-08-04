"""DimeNet++-style directional 3D atom encoder.

Adapter around PyG's reference DimeNet++ blocks (EmbeddingBlock,
BesselBasisLayer, SphericalBasisLayer, InteractionPPBlock, OutputPPBlock).
PyG's own triplets() needs torch-sparse and its radius_graph needs
torch_cluster, so both come from our neighbor module instead; everything
else - including the exact angle formula - mirrors
torch_geometric.nn.models.DimeNetPlusPlus.forward. Output is the per-atom
OutputPPBlock sum rather than a pooled molecule scalar.

Note PyG's EmbeddingBlock hard-codes Embedding(95, H), so z must be < 95.

Cost warning: triplet count grows as sum over j of deg_j^2, which is the
memory hot spot on dense systems - hence the max_num_neighbors cap.
"""

import torch
import torch.nn as nn
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    SphericalBasisLayer,
    EmbeddingBlock,
    InteractionPPBlock,
    OutputPPBlock,
)
from torch_geometric.nn.models.schnet import ShiftedSoftplus

from qtaim_embed.models.encoders.neighbors import radius_neighbors, build_triplets


def _cap_neighbors(edge_index, d, max_n):
    """Keep only the max_n nearest sources per target node."""
    dst = edge_index[1]
    order_d = torch.argsort(d, stable=True)
    order = order_d[torch.argsort(dst[order_d], stable=True)]
    dst_sorted = dst[order]
    counts = torch.bincount(dst_sorted)
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)])[:-1]
    rank = torch.arange(dst_sorted.numel(), device=d.device) - starts[dst_sorted]
    keep = order[rank < max_n]
    return edge_index[:, keep], d[keep]


class DimeNetPPEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 64,
        num_interactions: int = 3,
        num_radial: int = 6,
        num_spherical: int = 7,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        int_emb_size: int = 32,
        basis_emb_size: int = 8,
        num_output_layers: int = 2,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        act = ShiftedSoftplus()

        self.rbf = BesselBasisLayer(num_radial, cutoff)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff)
        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.interaction_blocks = nn.ModuleList(
            [
                InteractionPPBlock(
                    hidden_channels=hidden_channels,
                    int_emb_size=int_emb_size,
                    basis_emb_size=basis_emb_size,
                    num_spherical=num_spherical,
                    num_radial=num_radial,
                    num_before_skip=1,
                    num_after_skip=2,
                    act=act,
                )
                for _ in range(num_interactions)
            ]
        )
        self.output_blocks = nn.ModuleList(
            [
                OutputPPBlock(
                    num_radial=num_radial,
                    hidden_channels=hidden_channels,
                    out_emb_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_layers=num_output_layers,
                    act=act,
                    output_initializer="glorot_orthogonal",
                )
                for _ in range(num_interactions + 1)
            ]
        )

    def forward(self, pos, z, batch=None):
        pos = pos.to(self.emb.lin_rbf.weight.dtype)
        edge_index, dist = radius_neighbors(pos, batch, cutoff=self.cutoff)
        edge_index, dist = _cap_neighbors(edge_index, dist, self.max_num_neighbors)
        j, i = edge_index[0], edge_index[1]

        idx_i, idx_j, idx_k, idx_kj, idx_ji = build_triplets(
            edge_index, num_nodes=z.size(0)
        )

        # angle formula copied from DimeNetPlusPlus.forward
        pos_jk = pos[idx_j] - pos[idx_k]
        pos_ij = pos[idx_i] - pos[idx_j]
        a = (pos_ij * pos_jk).sum(dim=-1)
        b = torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))
        for interaction, output in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output(x, rbf, i, num_nodes=pos.size(0))
        return P
