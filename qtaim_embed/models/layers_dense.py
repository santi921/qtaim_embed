"""Dense padded heterogeneous convolution (performance plan A3).

Same function as `HeteroConv` of `GraphConvDropoutBatch` (`layers.py`), on a
padded per-molecule layout: every molecule in the batch occupies a fixed
(N_b atoms, B_b bonds, 1 global) block, so

  a2b / b2a  become one `bmm` against a 0/1 incidence matrix per molecule,
  a2g / b2g  become masked sums over the atom / bond axis,
  g2a / g2b  become broadcasts of the global row,
  a2a / b2b / g2g (self loops) become identities,

and no gather / scatter kernel runs. With bucketed batches (`data/bucketing.py`)
the shapes are static, which is what lets `torch.compile(mode="reduce-overhead")`
capture the conv stack as a CUDA graph. E3 in
docs/research/2026-09-track-a-measurements.md measured 1.4-2.2x over the
sparse reference (fwd+bwd, hidden 128-256, bf16) before padding waste is
reduced by bucketing.

Parameter layout (per destination type t, per incoming slot s):
  W_rel[t][s]  (in, out_t)   GraphConv.lin_rel.weight.T
  b_rel[t][s]  (out_t,)      GraphConv.lin_rel.bias
  W_root[t]    (in, 3*out_t) GraphConv.lin_root.weight.T, slot-major columns
Slot order is `SLOTS[t]`; `copy_residual_block_weights` maps a trained
`ResidualBlock` onto it exactly (parity test in tests/test_layers_dense.py).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_batch

from qtaim_embed.models.layers import EDGE_TYPE_MAP, ResidualBlock

NODE_TYPES: Tuple[str, str, str] = ("atom", "bond", "global")
# incoming edge types per destination type; index in the tuple is the slot
SLOTS: Dict[str, Tuple[str, str, str]] = {
    "atom": ("b2a", "g2a", "a2a"),
    "bond": ("a2b", "g2b", "b2b"),
    "global": ("a2g", "b2g", "g2g"),
}


def round_up(n: int, grid: int) -> int:
    return max(grid, ((int(n) + grid - 1) // grid) * grid)


@dataclass
class DenseHeteroBatch:
    """Padded view of a batched heterograph; features live in `x`."""

    x: Dict[str, torch.Tensor]  # atom [G,N_b,F], bond [G,B_b,F], global [G,1,F]
    mask: Dict[str, torch.Tensor]  # atom [G,N_b,1], bond [G,B_b,1] (float 0/1)
    inc_a2b: torch.Tensor  # [G, B_b, N_b], 1 where atom i -> bond j
    inc_b2a: torch.Tensor  # [G, N_b, B_b], 1 where bond j -> atom i
    num_graphs: int
    # flat indices of valid rows per type ([N_valid]); only set on the eager
    # path, where they let batch norm run on valid rows through cuDNN instead
    # of the masked arithmetic that torch.compile fuses but eager autograd
    # would save eight intermediates for; None when built with with_valid=False
    valid: Optional[Dict[str, torch.Tensor]] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.num_graphs, self.x["atom"].shape[1], self.x["bond"].shape[1]

    def to_flat(self, x: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Padded [G,N_b,F] back to the flat [N,F] node order of the PyG batch."""
        x = self.x if x is None else x
        return {
            "atom": x["atom"][self.mask["atom"].squeeze(-1).bool()],
            "bond": x["bond"][self.mask["bond"].squeeze(-1).bool()],
            # clone: under CUDA graphs a view of the graph output would be
            # overwritten by the next replay (gradient accumulation runs several
            # replays before the optimizer step)
            "global": x["global"].reshape(self.num_graphs, -1).clone(),
        }


def _incidence(edge_index, batch_src, ptr_src, ptr_dst, n_graphs, n_dst, n_src, dtype):
    g = batch_src[edge_index[0]]
    inc = torch.zeros(n_graphs, n_dst, n_src, device=edge_index.device, dtype=dtype)
    inc[g, edge_index[1] - ptr_dst[g], edge_index[0] - ptr_src[g]] = 1.0
    return inc


def to_dense_hetero(
    graph: HeteroData,
    feats: Dict[str, torch.Tensor],
    grid: int = 16,
    shape: Optional[Tuple[int, int]] = None,
    with_valid: bool = True,
) -> DenseHeteroBatch:
    """Pad a batched heterograph to (N_b, B_b) blocks.

    N_b / B_b are the per-molecule atom / bond maxima rounded up to `grid`
    unless `shape` is given. Bucketed LMDB loaders stamp the shape class on
    the batch as `graph.dense_shape` (see `BucketBatchSampler.batch_shape`)
    and the models pass it here, so every batch of a class shares one static
    shape for the compiled conv stack. `with_valid=False` skips the valid-row
    index build (a host sync) that only the eager batch-norm path uses.
    """
    ba, bb = graph["atom"].batch, graph["bond"].batch
    n_graphs = int(graph.num_graphs) if hasattr(graph, "num_graphs") else int(ba.max()) + 1
    n_a = torch.bincount(ba, minlength=n_graphs)
    n_b = torch.bincount(bb, minlength=n_graphs)
    if shape is None:
        n_b_max = round_up(int(n_a.max()), grid)
        b_b_max = round_up(int(n_b.max()), grid)
    else:
        n_b_max, b_b_max = int(shape[0]), int(shape[1])
    xa, ma = to_dense_batch(feats["atom"], ba, max_num_nodes=n_b_max, batch_size=n_graphs)
    xb, mb = to_dense_batch(feats["bond"], bb, max_num_nodes=b_b_max, batch_size=n_graphs)
    xg = feats["global"].reshape(n_graphs, 1, -1)
    ptr_a = torch.cat([n_a.new_zeros(1), n_a.cumsum(0)[:-1]])
    ptr_b = torch.cat([n_b.new_zeros(1), n_b.cumsum(0)[:-1]])
    dtype = feats["atom"].dtype
    inc_a2b = _incidence(graph[EDGE_TYPE_MAP["a2b"]].edge_index, ba, ptr_a, ptr_b,
                         n_graphs, b_b_max, n_b_max, dtype)
    inc_b2a = _incidence(graph[EDGE_TYPE_MAP["b2a"]].edge_index, bb, ptr_b, ptr_a,
                         n_graphs, n_b_max, b_b_max, dtype)
    return DenseHeteroBatch(
        x={"atom": xa, "bond": xb, "global": xg},
        mask={"atom": ma.unsqueeze(-1).to(dtype), "bond": mb.unsqueeze(-1).to(dtype)},
        inc_a2b=inc_a2b,
        inc_b2a=inc_b2a,
        num_graphs=n_graphs,
        valid={"atom": ma.reshape(-1).nonzero(as_tuple=True)[0],
               "bond": mb.reshape(-1).nonzero(as_tuple=True)[0]} if with_valid else None,
    )


class MaskedBatchNorm(nn.Module):
    """BatchNorm1d over the valid rows of a padded [S, C] tensor.

    Statistics use only rows with mask 1, so padding does not shift the
    running mean / var relative to the unpadded `nn.BatchNorm1d`. Same
    parameters and buffers as `nn.BatchNorm1d(C)`.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor],
                valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Both paths behave like nn.BatchNorm1d under autocast: statistics and
        # running buffers in fp32, output in the input dtype.
        if valid is not None:
            # eager path: cuDNN batch norm on the valid rows, zeros elsewhere
            rows = torch.nn.functional.batch_norm(
                x.index_select(0, valid), self.running_mean, self.running_var,
                self.weight, self.bias, self.training, self.momentum, self.eps,
            )
            if self.training:
                self.num_batches_tracked += 1
            return x.new_zeros(x.shape).index_copy(0, valid, rows.to(x.dtype))
        # masked path (compiled): statistics in fp32 even when x is bf16, or the
        # running mean / var drift by 1e-3 per step relative to the reference
        xf = x.float()
        if self.training:
            if mask is None:
                n = xf.shape[0]
                mean = xf.mean(0)
                var = xf.var(0, unbiased=False)
            else:
                mf = mask.to(xf.dtype)
                n = mf.sum()
                mean = (xf * mf).sum(0) / n
                var = (((xf - mean) ** 2) * mf).sum(0) / n
            with torch.no_grad():
                n_t = torch.as_tensor(n, dtype=var.dtype, device=var.device)
                unbiased = var * n_t / (n_t - 1).clamp_min(1)  # BatchNorm1d tracks unbiased var
                self.running_mean.lerp_(mean.detach().to(self.running_mean.dtype), self.momentum)
                self.running_var.lerp_(unbiased.detach().to(self.running_var.dtype), self.momentum)
                self.num_batches_tracked += 1
        else:
            mean, var = self.running_mean, self.running_var
        out = (xf - mean) * torch.rsqrt(var + self.eps) * self.weight.float() + self.bias.float()
        return out.to(x.dtype)


class DenseTypedGraphConv(nn.Module):
    """One hetero conv layer (9 edge types, sum aggregation) on the padded layout.

    Per destination type t: z_s = agg_s @ W_rel[t][s] + b_rel[t][s] + x_t @ W_root[t][s]
    for its three incoming slots, then activation, dropout, batch norm per slot
    (or batch norm, activation, dropout when `bn_before_activation`), and the
    sum over slots (exactly HeteroConv(aggr="sum") of GraphConvDropoutBatch).
    `global_mean` divides the atom and bond sums into the global node by the
    molecule's atom / bond count (GraphConv aggr="mean" on a2g / b2g). Padded
    rows are zeroed afterwards so they never leak into the sums that feed the
    global node.
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: Dict[str, int],
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        bias: bool = True,
        bn_before_activation: bool = False,
        global_mean: bool = False,
    ):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = dict(out_feats)
        self.activation = activation
        self.bn_before_activation = bn_before_activation
        self.global_mean = global_mean
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.W_rel = nn.ParameterList()
        self.b_rel = nn.ParameterList()
        self.W_root = nn.ParameterList()
        self.norms = nn.ModuleList()
        for nt in NODE_TYPES:
            out = out_feats[nt]
            w_rel = torch.empty(3, in_feats, out)
            w_root = torch.empty(in_feats, 3 * out)
            for s in range(3):
                nn.init.kaiming_uniform_(w_rel[s].T, a=5 ** 0.5)
                nn.init.kaiming_uniform_(w_root[:, s * out:(s + 1) * out].T, a=5 ** 0.5)
            self.W_rel.append(nn.Parameter(w_rel))
            self.W_root.append(nn.Parameter(w_root))
            self.b_rel.append(nn.Parameter(torch.zeros(3, 1, out), requires_grad=bias))
            self.norms.append(MaskedBatchNorm(3 * out) if batch_norm else nn.Identity())

    def _head(self, t: int, agg: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor],
              valid: Optional[torch.Tensor] = None):
        out = self.out_feats[NODE_TYPES[t]]
        n_graphs, n_rows = x.shape[0], x.shape[1]
        rows = n_graphs * n_rows
        z = torch.baddbmm(self.b_rel[t], agg.reshape(3, rows, self.in_feats), self.W_rel[t])
        root = torch.mm(x.reshape(rows, self.in_feats), self.W_root[t]).view(rows, 3, out).transpose(0, 1)
        z = z + root

        def norm(z):
            if isinstance(self.norms[t], nn.Identity):
                return z
            flat_mask = None if mask is None else mask.reshape(rows, 1)
            z = self.norms[t](z.transpose(0, 1).reshape(rows, 3 * out), flat_mask, valid)
            return z.view(rows, 3, out).transpose(0, 1)

        if self.bn_before_activation:
            z = norm(z)
        if self.activation is not None:
            z = self.activation(z)
        if self.dropout is not None:
            z = self.dropout(z)
        if not self.bn_before_activation:
            z = norm(z)
        z = z.sum(0).view(n_graphs, n_rows, out)
        return z * mask if mask is not None else z

    def forward(self, dense: DenseHeteroBatch, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        xa, xb, xg = x["atom"], x["bond"], x["global"]
        n_graphs, n_b, _ = xa.shape
        b_b = xb.shape[1]
        agg_a = torch.stack([torch.bmm(dense.inc_b2a, xb), xg.expand(n_graphs, n_b, -1), xa])
        agg_b = torch.stack([torch.bmm(dense.inc_a2b, xa), xg.expand(n_graphs, b_b, -1), xb])
        sum_a, sum_b = xa.sum(1, keepdim=True), xb.sum(1, keepdim=True)
        if self.global_mean:
            sum_a = sum_a / dense.mask["atom"].sum(1, keepdim=True).clamp_min(1)
            sum_b = sum_b / dense.mask["bond"].sum(1, keepdim=True).clamp_min(1)
        agg_g = torch.stack([sum_a, sum_b, xg])
        valid = dense.valid or {}
        return {
            "atom": self._head(0, agg_a, xa, dense.mask["atom"], valid.get("atom")),
            "bond": self._head(1, agg_b, xb, dense.mask["bond"], valid.get("bond")),
            "global": self._head(2, agg_g, xg, None),
        }


class DenseResidualBlock(nn.Module):
    """Drop-in for `layers.ResidualBlock` on the padded layout.

    Accepts the same `layer_args` dict (short edge-type keys with in_feats /
    out_feats / activation / dropout / batch_norm_tf, plus the `_inner`
    variants for the output block) so `get_layer_args` and the model
    constructors need no changes beyond the class name.
    """

    def __init__(
        self,
        layer_args: Dict[str, Dict],
        aggregate: str = "sum",
        resid_n_graph_convs: int = 2,
        output_block: bool = False,
    ):
        super().__init__()
        assert aggregate == "sum", f"DenseResidualBlock supports aggregate='sum' only, got {aggregate!r}"
        self.output_block = output_block
        self.layers = nn.ModuleList()
        for i in range(resid_n_graph_convs):
            inner = output_block and i < resid_n_graph_convs - 1
            self.layers.append(self._layer_from_args(layer_args, suffix="_inner" if inner else ""))
        self.out_feats = {e: self.layers[-1].out_feats[EDGE_TYPE_MAP[e][2]] for e in EDGE_TYPE_MAP}

    @staticmethod
    def _layer_from_args(layer_args, suffix):
        ins = {layer_args[e + suffix]["in_feats"] for slots in SLOTS.values() for e in slots}
        assert len(ins) == 1, f"dense layout needs one input width across edge types, got {ins}"
        outs = {}
        for nt, slots in SLOTS.items():
            outs_t = {layer_args[e + suffix]["out_feats"] for e in slots}
            assert len(outs_t) == 1, f"edge types into {nt!r} disagree on out_feats: {outs_t}"
            outs[nt] = outs_t.pop()
        ref = layer_args["a2b" + suffix]
        return DenseTypedGraphConv(
            in_feats=ins.pop(),
            out_feats=outs,
            activation=ref.get("activation"),
            dropout=ref.get("dropout", 0.0),
            batch_norm=ref.get("batch_norm_tf", False),
            bias=True,  # GraphConvDropoutBatch ignores layer_args["bias"]; GraphConv always has lin_rel.bias
            bn_before_activation=ref.get("bn_before_activation", False),
            global_mean=layer_args["a2g" + suffix].get("aggr", "add") == "mean",
        )

    def forward(self, dense: DenseHeteroBatch, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = x
        for layer in self.layers:
            x = layer(dense, x)
        if not self.output_block:
            x = {k: x[k] + inputs[k] for k in x}
        return x


@torch.no_grad()
def copy_residual_block_weights(src: ResidualBlock, dst: DenseResidualBlock) -> None:
    """Copy a trained ResidualBlock (HeteroConv of GraphConvDropoutBatch) onto a
    DenseResidualBlock of the same layer_args. Exact parity at fp32."""
    assert len(src.layers) == len(dst.layers)
    for hetero, dense in zip(src.layers, dst.layers):
        for t, nt in enumerate(NODE_TYPES):
            out = dense.out_feats[nt]
            for s, e in enumerate(SLOTS[nt]):
                mod = hetero.convs[EDGE_TYPE_MAP[e]]
                gc = mod.graph_conv
                dense.W_rel[t][s].copy_(gc.lin_rel.weight.t())
                if gc.lin_rel.bias is not None:
                    dense.b_rel[t][s, 0].copy_(gc.lin_rel.bias)
                dense.W_root[t][:, s * out:(s + 1) * out].copy_(gc.lin_root.weight.t())
                if mod.batch_norm is not None:
                    bn = dense.norms[t]
                    sl = slice(s * out, (s + 1) * out)
                    bn.weight[sl].copy_(mod.batch_norm.weight)
                    bn.bias[sl].copy_(mod.batch_norm.bias)
                    bn.running_mean[sl].copy_(mod.batch_norm.running_mean)
                    bn.running_var[sl].copy_(mod.batch_norm.running_var)
