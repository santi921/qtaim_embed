"""3D geometric atom encoders (SchNet-style, DimeNet++-style, MACE-style).

Each encoder consumes only atom.pos / atom.z and returns per-atom embeddings
of width encoder_hidden. The output is concatenated with atom.feat ahead of
UnifySize; see encode_atom_inputs.
"""

import torch

ENCODER_FNS = ["none", "schnet", "dimenetpp", "equivariant"]


def build_encoder(hparams):
    """encoder_fn string -> encoder module, or None for "none"."""
    encoder_fn = getattr(hparams, "encoder_fn", "none") or "none"
    if encoder_fn == "none":
        return None
    if encoder_fn == "schnet":
        from qtaim_embed.models.encoders.schnet_encoder import SchNetEncoder

        return SchNetEncoder(
            hidden_channels=hparams.encoder_hidden,
            num_interactions=hparams.encoder_n_interactions,
            num_gaussians=hparams.encoder_num_gaussians,
            cutoff=hparams.encoder_cutoff,
        )
    if encoder_fn == "dimenetpp":
        from qtaim_embed.models.encoders.dimenetpp_encoder import DimeNetPPEncoder

        return DimeNetPPEncoder(
            hidden_channels=hparams.encoder_hidden,
            num_interactions=hparams.encoder_n_interactions,
            num_radial=hparams.encoder_num_radial,
            cutoff=hparams.encoder_cutoff,
            max_num_neighbors=hparams.encoder_max_neighbors,
        )
    if encoder_fn == "equivariant":
        from qtaim_embed.models.encoders.equivariant_encoder import EquivariantEncoder

        return EquivariantEncoder(
            hidden_channels=hparams.encoder_hidden,
            num_interactions=hparams.encoder_n_interactions,
            num_radial=hparams.encoder_num_radial,
            lmax=hparams.encoder_lmax,
            cutoff=hparams.encoder_cutoff,
            tp_mode=getattr(hparams, "encoder_tp", "channelwise") or "channelwise",
        )
    raise ValueError(f"encoder_fn must be one of {ENCODER_FNS}, got {encoder_fn}")


def encode_atom_inputs(encoder, graph, inputs):
    """Concatenate 3D encoder output onto the atom features.

    Called at every embedding call site (forward and feature_at_each_layer of
    each model class). No-op when no encoder is configured.
    """
    if encoder is None:
        return inputs
    atom = graph["atom"]
    with torch.profiler.record_function("encoder"):
        h_enc = encoder(atom.pos, atom.z, atom.batch if "batch" in atom else None)
    inputs = dict(inputs)
    inputs["atom"] = torch.cat(
        [inputs["atom"], h_enc.to(inputs["atom"].dtype)], dim=-1
    )
    return inputs
