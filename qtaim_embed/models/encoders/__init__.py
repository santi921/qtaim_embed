"""3D geometric atom encoders (SchNet-style, DimeNet++-style, MACE-style).

Each encoder consumes only atom.pos / atom.z and returns per-atom embeddings
of width encoder_hidden. The output is concatenated with atom.feat ahead of
UnifySize; see encode_atom_inputs.
"""

import torch

ENCODER_FNS = ["none", "schnet", "dimenetpp", "equivariant"]

# Every encoder_* knob and its default, in one place. Loaders read config
# through encoder_kwargs_from_config, default configs splat ENCODER_DEFAULTS,
# and build_encoder falls back to these for hparams saved before a knob
# existed. Adding a knob: add it here, consume it in build_encoder, and add
# the keyword to the four model constructors (Lightning hparams are explicit).
ENCODER_DEFAULTS = {
    "encoder_fn": "none",
    "encoder_hidden": 64,
    "encoder_cutoff": 5.0,
    "encoder_n_interactions": 3,
    "encoder_num_gaussians": 50,
    "encoder_num_radial": 6,
    "encoder_lmax": 1,
    "encoder_max_neighbors": 16,
    "encoder_tp": "channelwise",
    "encoder_max_z": 119,
}


def encoder_kwargs_from_config(config: dict) -> dict:
    """config dict -> the encoder_* constructor kwargs, defaults filled in."""
    return {k: config.get(k, v) for k, v in ENCODER_DEFAULTS.items()}


def _knob(hparams, key):
    value = getattr(hparams, key, None)
    return ENCODER_DEFAULTS[key] if value is None else value


def check_encoder_hparams(encoder_fn: str, compiled: bool = False, conv_fn: str = "") -> None:
    """Constructor-time guards shared by the hetero models."""
    assert encoder_fn in ENCODER_FNS, f"encoder_fn must be one of {ENCODER_FNS} but got {encoder_fn}"
    assert not (compiled and encoder_fn != "none" and conv_fn != "ResidualBlockDense"), (
        "compiled=True is unsupported with a 3D encoder unless conv_fn is "
        "ResidualBlockDense (which compiles only the padded conv stack): the "
        "in-forward neighbor construction is data-dependent and graph-breaks torch.compile"
    )


def attach_encoder(hparams):
    """(encoder or None, extra atom input width) for a model's hparams."""
    encoder = build_encoder(hparams)
    return encoder, (_knob(hparams, "encoder_hidden") if encoder is not None else 0)


def build_encoder(hparams):
    """encoder_fn string -> encoder module, or None for "none"."""
    k = {key: _knob(hparams, key) for key in ENCODER_DEFAULTS}
    encoder_fn = k["encoder_fn"] or "none"
    if encoder_fn == "none":
        return None
    if encoder_fn == "schnet":
        from qtaim_embed.models.encoders.schnet_encoder import SchNetEncoder

        return SchNetEncoder(
            hidden_channels=k["encoder_hidden"],
            num_interactions=k["encoder_n_interactions"],
            num_gaussians=k["encoder_num_gaussians"],
            cutoff=k["encoder_cutoff"],
            max_z=k["encoder_max_z"],
        )
    if encoder_fn == "dimenetpp":
        from qtaim_embed.models.encoders.dimenetpp_encoder import DimeNetPPEncoder

        return DimeNetPPEncoder(
            hidden_channels=k["encoder_hidden"],
            num_interactions=k["encoder_n_interactions"],
            num_radial=k["encoder_num_radial"],
            cutoff=k["encoder_cutoff"],
            max_num_neighbors=k["encoder_max_neighbors"],
            max_z=k["encoder_max_z"],
        )
    if encoder_fn == "equivariant":
        from qtaim_embed.models.encoders.equivariant_encoder import EquivariantEncoder

        return EquivariantEncoder(
            hidden_channels=k["encoder_hidden"],
            num_interactions=k["encoder_n_interactions"],
            num_radial=k["encoder_num_radial"],
            lmax=k["encoder_lmax"],
            cutoff=k["encoder_cutoff"],
            tp_mode=k["encoder_tp"] or "channelwise",
            max_z=k["encoder_max_z"],
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
