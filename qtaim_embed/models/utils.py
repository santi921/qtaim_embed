import torch


def _split_batched_output(graph, value, key):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.

    Returns:
        list of tensor.

    """
    n_nodes = graph.batch_num_nodes(key)
    # convert to tuple
    n_nodes = tuple(n_nodes)
    return torch.split(value, n_nodes)


def get_layer_args(hparams, layer_ind=None):
    """
    Converts hparam dictionary to a dictionary of arguments for a layer.
    """

    assert hparams.conv_fn in [
        "GraphConvDropoutBatch",
        "ResidualBlock",
    ], "conv_fn must be either GraphConvDropoutBatch or ResidualBlock"

    layer_args = {}
    if hparams.conv_fn == "GraphConvDropoutBatch":
        atom_out = hparams.atom_input_size
        bond_out = hparams.bond_input_size
        global_out = hparams.global_input_size

        if layer_ind == hparams.n_conv_layers - 1:
            if "atom" in hparams.target_dict.keys():
                atom_out = len(hparams.target_dict["atom"])
            if "bond" in hparams.target_dict.keys():
                bond_out = len(hparams.target_dict["bond"])
            if "global" in hparams.target_dict.keys():
                global_out = len(hparams.target_dict["global"])

        layer_args["a2b"] = {
            "in_feats": hparams.atom_input_size,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2a"] = {
            "in_feats": hparams.bond_input_size,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2g"] = {
            "in_feats": hparams.atom_input_size,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2g"] = {
            "in_feats": hparams.bond_input_size,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2a"] = {
            "in_feats": hparams.global_input_size,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2b"] = {
            "in_feats": hparams.global_input_size,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2a"] = {
            "in_feats": hparams.atom_input_size,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2b"] = {
            "in_feats": hparams.bond_input_size,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2g"] = {
            "in_feats": hparams.global_input_size,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        return layer_args

    elif hparams.conv_fn == "ResidualBlock":
        atom_out = hparams.atom_input_size
        bond_out = hparams.bond_input_size
        global_out = hparams.global_input_size
        # resid_n_graph_convs = hparams.resid_n_graph_convs

        if layer_ind != -1:  # last residual layer has different args
            layer_args["a2b_inner"] = {
                "in_feats": hparams.atom_input_size,
                "out_feats": bond_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["b2a_inner"] = {
                "in_feats": hparams.bond_input_size,
                "out_feats": atom_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["a2g_inner"] = {
                "in_feats": hparams.atom_input_size,
                "out_feats": global_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["b2g_inner"] = {
                "in_feats": hparams.bond_input_size,
                "out_feats": global_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["g2a_inner"] = {
                "in_feats": hparams.global_input_size,
                "out_feats": atom_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["g2b_inner"] = {
                "in_feats": hparams.global_input_size,
                "out_feats": bond_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["a2a_inner"] = {
                "in_feats": hparams.atom_input_size,
                "out_feats": atom_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["b2b_inner"] = {
                "in_feats": hparams.bond_input_size,
                "out_feats": bond_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["g2g_inner"] = {
                "in_feats": hparams.global_input_size,
                "out_feats": global_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            if "atom" in hparams.target_dict.keys():
                atom_out = len(hparams.target_dict["atom"])
            if "bond" in hparams.target_dict.keys():
                bond_out = len(hparams.target_dict["bond"])
            if "global" in hparams.target_dict.keys():
                global_out = len(hparams.target_dict["global"])

        layer_args["a2b"] = {
            "in_feats": hparams.atom_input_size,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2a"] = {
            "in_feats": hparams.bond_input_size,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2g"] = {
            "in_feats": hparams.atom_input_size,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2g"] = {
            "in_feats": hparams.bond_input_size,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2a"] = {
            "in_feats": hparams.global_input_size,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2b"] = {
            "in_feats": hparams.global_input_size,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2a"] = {
            "in_feats": hparams.atom_input_size,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2b"] = {
            "in_feats": hparams.bond_input_size,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2g"] = {
            "in_feats": hparams.global_input_size,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        return layer_args
