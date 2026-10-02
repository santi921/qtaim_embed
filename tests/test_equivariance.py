import pytest
import torch
import e3nn.o3 as o3

from qtaim_embed.models.encoders.schnet_encoder import SchNetEncoder
from qtaim_embed.models.encoders.equivariant_encoder import EquivariantEncoder


def _random_system(n=20, seed=0):
    torch.manual_seed(seed)
    pos = torch.rand(n, 3) * 4.0
    z = torch.randint(1, 20, (n,))
    return pos, z


class TestEquivariantEncoder:
    def test_scalar_readout_is_invariant(self):
        pos, z = _random_system()
        enc = EquivariantEncoder(hidden_channels=16, num_interactions=2, lmax=1).eval()
        R = o3.rand_matrix()
        with torch.no_grad():
            h_rot = enc(pos @ R.T, z)
            h = enc(pos, z)
        assert torch.allclose(h_rot, h, atol=1e-4)

    def test_features_transform_by_wigner_d(self):
        pos, z = _random_system(seed=1)
        enc = EquivariantEncoder(hidden_channels=8, num_interactions=2, lmax=1).eval()
        R = o3.rand_matrix()
        with torch.no_grad():
            h_rot = enc.forward_features(pos @ R.T, z)
            h = enc.forward_features(pos, z)
        D = enc.irreps_h.D_from_matrix(R)
        assert torch.allclose(h_rot, h @ D.T, atol=1e-4)

    @pytest.mark.parametrize("tp_mode", ["channelwise", "fully_connected"])
    def test_both_tensor_products_invariant_and_equivariant(self, tp_mode):
        torch.manual_seed(11)
        enc = EquivariantEncoder(hidden_channels=8, num_interactions=2, lmax=1, tp_mode=tp_mode).eval()
        pos, z = torch.randn(9, 3) * 2.0, torch.randint(1, 20, (9,))
        R = o3.rand_matrix()
        D = enc.irreps_h.D_from_matrix(R)
        with torch.no_grad():
            h = enc.forward_features(pos, z)
            h_rot = enc.forward_features(pos @ R.T, z)
        assert torch.allclose(h_rot, h @ D.T, atol=1e-4)
        assert torch.allclose(enc(pos @ R.T, z), enc(pos, z), atol=1e-4)

    def test_channelwise_weights_per_edge_are_small(self):
        cw = EquivariantEncoder(hidden_channels=64, num_interactions=1, lmax=1)
        fc = EquivariantEncoder(hidden_channels=64, num_interactions=1, lmax=1, tp_mode="fully_connected")
        assert cw.tensor_products[0].weight_numel == 256
        assert fc.tensor_products[0].weight_numel == 16384
        with pytest.raises(ValueError):
            EquivariantEncoder(tp_mode="dense")

    def test_l1_block_is_nonzero(self):
        # the vector channels must actually carry signal, otherwise the
        # equivariance assertions above pass vacuously on zeros
        pos, z = _random_system(seed=2)
        enc = EquivariantEncoder(hidden_channels=8, num_interactions=2, lmax=1).eval()
        with torch.no_grad():
            h = enc.forward_features(pos, z)
        assert h[:, 8:].abs().max() > 0

    def test_batch_independence(self):
        pos_a, z_a = _random_system(n=8, seed=3)
        pos_b, z_b = _random_system(n=6, seed=4)
        enc = EquivariantEncoder(hidden_channels=8, num_interactions=2, lmax=1).eval()
        batch = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(6, dtype=torch.long)])
        with torch.no_grad():
            joint = enc(torch.cat([pos_a, pos_b]), torch.cat([z_a, z_b]), batch)
            solo_a = enc(pos_a, z_a)
            solo_b = enc(pos_b, z_b)
        assert torch.allclose(joint[:8], solo_a, atol=1e-5)
        assert torch.allclose(joint[8:], solo_b, atol=1e-5)


class TestInvariantEncoders:
    def test_schnet_rotation_invariance(self):
        # invariant by construction; catches accidental use of raw coordinates
        pos, z = _random_system(seed=5)
        enc = SchNetEncoder(hidden_channels=16, num_interactions=2).eval()
        R = o3.rand_matrix()
        with torch.no_grad():
            h_rot = enc(pos @ R.T, z)
            h = enc(pos, z)
        assert torch.allclose(h_rot, h, atol=1e-4)

    def test_dimenetpp_rotation_invariance(self):
        from qtaim_embed.models.encoders.dimenetpp_encoder import DimeNetPPEncoder

        # float64 makes this decisive: any use of raw coordinates would show
        # up far above the 1e-8 numerics floor
        pos, z = _random_system(seed=6)
        pos = pos.double()
        enc = DimeNetPPEncoder(hidden_channels=16, num_interactions=2).double().eval()
        # rand_matrix() in float32 cast to double is only orthogonal to 1e-7
        R = o3.rand_matrix(dtype=torch.float64)
        with torch.no_grad():
            h_rot = enc(pos @ R.T, z)
            h = enc(pos, z)
        assert torch.allclose(h_rot, h, atol=1e-8)
