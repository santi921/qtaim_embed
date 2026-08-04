import math
from types import SimpleNamespace

import torch

from qtaim_embed.models.encoders import build_encoder
from qtaim_embed.models.encoders.neighbors import radius_neighbors, build_triplets
from qtaim_embed.models.encoders.schnet_encoder import SchNetEncoder
from qtaim_embed.models.encoders.dimenetpp_encoder import DimeNetPPEncoder


class TestSchNetParity:
    def test_matches_pyg_reference_blocks(self):
        """Weight-copied adapter must reproduce PyG SchNet numerics exactly.

        PyG's full SchNet can't be called end-to-end (its radius_graph needs
        torch_cluster), so its embedding/interactions submodules are driven
        directly with our edge_index and compared against our forward.
        """
        from torch_geometric.nn.models import SchNet

        torch.manual_seed(0)
        H, cutoff = 32, 5.0
        ref = SchNet(
            hidden_channels=H,
            num_filters=H,
            num_interactions=3,
            num_gaussians=50,
            cutoff=cutoff,
        ).eval()
        ours = SchNetEncoder(
            hidden_channels=H,
            num_interactions=3,
            num_gaussians=50,
            cutoff=cutoff,
        ).eval()
        ours.embedding.weight.data[: ref.embedding.num_embeddings] = (
            ref.embedding.weight.data
        )
        ours.interactions.load_state_dict(ref.interactions.state_dict())

        pos = torch.rand(25, 3) * 4.0
        z = torch.randint(1, 20, (25,))
        batch = torch.zeros(25, dtype=torch.long)

        with torch.no_grad():
            got = ours(pos, z, batch)

            edge_index, edge_weight = radius_neighbors(pos, batch, cutoff=cutoff)
            edge_attr = ref.distance_expansion(edge_weight)
            want = ref.embedding(z)
            for interaction in ref.interactions:
                want = want + interaction(want, edge_index, edge_weight, edge_attr)

        assert torch.allclose(got, want, atol=1e-6)

    def test_batch_independence(self):
        """Encoding two molecules batched must equal encoding them separately."""
        torch.manual_seed(1)
        enc = SchNetEncoder(hidden_channels=16, num_interactions=2, cutoff=5.0).eval()
        pos_a, z_a = torch.rand(8, 3) * 3.0, torch.randint(1, 10, (8,))
        pos_b, z_b = torch.rand(6, 3) * 3.0, torch.randint(1, 10, (6,))
        batch = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(6, dtype=torch.long)])
        with torch.no_grad():
            joint = enc(torch.cat([pos_a, pos_b]), torch.cat([z_a, z_b]), batch)
            solo_a = enc(pos_a, z_a)
            solo_b = enc(pos_b, z_b)
        assert torch.allclose(joint[:8], solo_a, atol=1e-5)
        assert torch.allclose(joint[8:], solo_b, atol=1e-5)


class TestDimeNetPP:
    def test_forward_shape_and_gradients(self):
        torch.manual_seed(0)
        enc = DimeNetPPEncoder(hidden_channels=16, num_interactions=2)
        pos = torch.rand(15, 3) * 3.0
        z = torch.randint(1, 20, (15,))
        out = enc(pos, z, torch.zeros(15, dtype=torch.long))
        assert out.shape == (15, 16)
        assert torch.isfinite(out).all()
        out.sum().backward()
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_water_interior_angle_via_triplets(self):
        # the triplet indices must recover the known 104.5 degree HOH angle
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.23999, 0.92663, 0.0]]
        )
        edge_index, _ = radius_neighbors(pos, cutoff=5.0)
        idx_i, idx_j, idx_k, _, _ = build_triplets(edge_index, 3)
        v1 = pos[idx_i] - pos[idx_j]
        v2 = pos[idx_k] - pos[idx_j]
        interior = torch.atan2(
            torch.cross(v1, v2, dim=-1).norm(dim=-1), (v1 * v2).sum(dim=-1)
        )
        at_oxygen = interior[idx_j == 0] * 180.0 / math.pi
        assert at_oxygen.numel() == 2  # both orderings of (H1, H2)
        assert torch.allclose(at_oxygen, torch.full_like(at_oxygen, 104.5), atol=0.5)

    def test_batch_independence(self):
        torch.manual_seed(1)
        enc = DimeNetPPEncoder(hidden_channels=16, num_interactions=2).eval()
        pos_a, z_a = torch.rand(8, 3) * 3.0, torch.randint(1, 10, (8,))
        pos_b, z_b = torch.rand(6, 3) * 3.0, torch.randint(1, 10, (6,))
        batch = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(6, dtype=torch.long)])
        with torch.no_grad():
            joint = enc(torch.cat([pos_a, pos_b]), torch.cat([z_a, z_b]), batch)
            solo_a = enc(pos_a, z_a)
            solo_b = enc(pos_b, z_b)
        assert torch.allclose(joint[:8], solo_a, atol=1e-4)
        assert torch.allclose(joint[8:], solo_b, atol=1e-4)

    def test_neighbor_cap(self):
        torch.manual_seed(2)
        enc = DimeNetPPEncoder(hidden_channels=8, num_interactions=1, max_num_neighbors=4)
        pos = torch.rand(30, 3)  # dense cluster, everyone within cutoff
        z = torch.randint(1, 10, (30,))
        out = enc(pos, z)
        assert torch.isfinite(out).all()


class TestBuildEncoderDispatch:
    def _hparams(self, encoder_fn):
        return SimpleNamespace(
            encoder_fn=encoder_fn,
            encoder_hidden=8,
            encoder_cutoff=5.0,
            encoder_n_interactions=1,
            encoder_num_gaussians=10,
            encoder_num_radial=4,
            encoder_lmax=1,
            encoder_max_neighbors=16,
        )

    def test_all_encoder_fns(self):
        assert build_encoder(self._hparams("none")) is None
        pos = torch.rand(6, 3) * 3.0
        z = torch.randint(1, 10, (6,))
        for name in ["schnet", "dimenetpp", "equivariant"]:
            enc = build_encoder(self._hparams(name))
            out = enc(pos, z)
            assert out.shape == (6, 8), name
