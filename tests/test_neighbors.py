import torch
import pytest

from qtaim_embed.models.encoders.neighbors import (
    RCOV,
    radius_neighbors,
    candidate_pairs,
    build_triplets,
)


def _brute_radius(pos, batch, cutoff):
    n = pos.shape[0]
    pairs = set()
    for i in range(n):
        for j in range(n):
            if i == j or batch[i] != batch[j]:
                continue
            if torch.norm(pos[i] - pos[j]) <= cutoff:
                pairs.add((j, i))  # (source, target)
    return pairs


def _brute_candidates(pos, z, batch, rcov, mult):
    n = pos.shape[0]
    pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            if batch[i] != batch[j]:
                continue
            thr = mult * (rcov[z[i]] + rcov[z[j]])
            if torch.norm(pos[i] - pos[j]) <= thr:
                pairs.add((i, j))
    return pairs


def _brute_triplets(edge_index):
    row, col = edge_index
    trips = set()
    for e_ji in range(row.numel()):
        j, i = row[e_ji].item(), col[e_ji].item()
        for e_kj in range(row.numel()):
            k, tgt = row[e_kj].item(), col[e_kj].item()
            if tgt == j and k != i:
                trips.add((i, j, k, e_kj, e_ji))
    return trips


class TestRadiusNeighbors:
    def test_vs_brute_force_across_chunk_boundaries(self):
        torch.manual_seed(0)
        for _ in range(5):
            pos = torch.rand(40, 3) * 8.0
            batch = torch.zeros(40, dtype=torch.long)
            edge_index, d = radius_neighbors(pos, batch, cutoff=3.0, chunk=7)
            got = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
            assert got == _brute_radius(pos, batch, 3.0)
            src, dst = edge_index
            assert torch.equal(d, torch.norm(pos[src] - pos[dst], dim=-1))

    def test_bidirectional(self):
        torch.manual_seed(1)
        pos = torch.rand(20, 3) * 5.0
        edge_index, _ = radius_neighbors(pos, cutoff=3.0)
        pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        assert all((b, a) in pairs for (a, b) in pairs)

    def test_no_cross_molecule_edges(self):
        # two overlapping molecules: every cross pair is within the cutoff,
        # so any batch-mask failure produces cross edges
        torch.manual_seed(2)
        pos_a = torch.rand(15, 3)
        pos_b = pos_a + 0.1
        pos = torch.cat([pos_a, pos_b])
        batch = torch.cat([torch.zeros(15, dtype=torch.long), torch.ones(15, dtype=torch.long)])
        edge_index, _ = radius_neighbors(pos, batch, cutoff=5.0, chunk=7)
        assert (batch[edge_index[0]] == batch[edge_index[1]]).all()
        assert edge_index.shape[1] == 2 * 15 * 14  # both dense blocks, no cross


class TestCandidatePairs:
    def test_vs_brute_force(self):
        torch.manual_seed(3)
        for _ in range(5):
            pos = torch.rand(40, 3) * 6.0
            z = torch.randint(1, 36, (40,))
            batch = torch.zeros(40, dtype=torch.long)
            ii, jj, dd = candidate_pairs(pos, z, batch, pool_multiplier=2.0, chunk=7)
            got = set(zip(ii.tolist(), jj.tolist()))
            assert got == _brute_candidates(pos, z, batch, RCOV, 2.0)
            assert (ii < jj).all()

    def test_batch_mask(self):
        torch.manual_seed(4)
        pos = torch.rand(10, 3)
        pos = torch.cat([pos, pos])
        z = torch.full((20,), 6, dtype=torch.long)
        batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
        ii, jj, _ = candidate_pairs(pos, z, batch, pool_multiplier=2.0, chunk=3)
        assert (batch[ii] == batch[jj]).all()

    def test_rcov_table(self):
        assert RCOV.shape == (119,)
        assert RCOV[1] > 0.2 and RCOV[6] > RCOV[1]  # C > H, both physical


class TestBuildTriplets:
    def test_vs_brute_force(self):
        torch.manual_seed(5)
        for _ in range(5):
            pos = torch.rand(12, 3) * 3.0
            edge_index, _ = radius_neighbors(pos, cutoff=2.0)
            if edge_index.numel() == 0:
                continue
            idx_i, idx_j, idx_k, idx_kj, idx_ji = build_triplets(edge_index, 12)
            got = set(
                zip(idx_i.tolist(), idx_j.tolist(), idx_k.tolist(),
                    idx_kj.tolist(), idx_ji.tolist())
            )
            assert got == _brute_triplets(edge_index)

    def test_round_trip_consistency(self):
        torch.manual_seed(6)
        pos = torch.rand(15, 3) * 3.0
        edge_index, _ = radius_neighbors(pos, cutoff=2.5)
        row, col = edge_index
        idx_i, idx_j, idx_k, idx_kj, idx_ji = build_triplets(edge_index, 15)
        assert (col[idx_ji] == idx_i).all()
        assert (row[idx_ji] == idx_j).all()
        assert (col[idx_kj] == idx_j).all()
        assert (row[idx_kj] == idx_k).all()
        assert (idx_k != idx_i).all()
