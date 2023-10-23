import torch


def get_layer_args(hparams, layer_ind=None, embedding_in=False):
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
        atom_in = hparams.atom_input_size
        bond_in = hparams.bond_input_size
        global_in = hparams.global_input_size

        if layer_ind == hparams.n_conv_layers - 1:
            if "atom" in hparams.target_dict.keys():
                atom_out = len(hparams.target_dict["atom"])
            if "bond" in hparams.target_dict.keys():
                bond_out = len(hparams.target_dict["bond"])
            if "global" in hparams.target_dict.keys():
                global_out = len(hparams.target_dict["global"])

        if embedding_in:
            atom_in = hparams.embedding_size
            bond_in = hparams.embedding_size
            global_in = hparams.embedding_size

        layer_args["a2b"] = {
            "in_feats": atom_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2a"] = {
            "in_feats": bond_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2g"] = {
            "in_feats": atom_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2g"] = {
            "in_feats": bond_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2a"] = {
            "in_feats": global_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2b"] = {
            "in_feats": global_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2a"] = {
            "in_feats": atom_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2b"] = {
            "in_feats": bond_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2g"] = {
            "in_feats": global_in,
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
        atom_in = hparams.atom_input_size
        bond_in = hparams.bond_input_size
        global_in = hparams.global_input_size

        if embedding_in:
            atom_in = hparams.embedding_size
            bond_in = hparams.embedding_size
            global_in = hparams.embedding_size
        # resid_n_graph_convs = hparams.resid_n_graph_convs

        if layer_ind != -1:  # last residual layer has different args
            # print("triggered early stop condition!!!")
            layer_args["a2b_inner"] = {
                "in_feats": atom_in,
                "out_feats": bond_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["b2a_inner"] = {
                "in_feats": bond_in,
                "out_feats": atom_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["a2g_inner"] = {
                "in_feats": atom_in,
                "out_feats": global_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["b2g_inner"] = {
                "in_feats": bond_in,
                "out_feats": global_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["g2a_inner"] = {
                "in_feats": global_in,
                "out_feats": atom_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["g2b_inner"] = {
                "in_feats": global_in,
                "out_feats": bond_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": hparams.activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["a2a_inner"] = {
                "in_feats": atom_in,
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
                "in_feats": global_in,
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
            # print("target_dict", hparams.target_dict)

        layer_args["a2b"] = {
            "in_feats": atom_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2a"] = {
            "in_feats": bond_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2g"] = {
            "in_feats": atom_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2g"] = {
            "in_feats": bond_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2a"] = {
            "in_feats": global_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2b"] = {
            "in_feats": global_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2a"] = {
            "in_feats": atom_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2b"] = {
            "in_feats": bond_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2g"] = {
            "in_feats": global_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": hparams.activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        return layer_args


def link_fmt_to_node_fmt(dict_feats):
    """
    Converts a dictionary of features from link format to node format.
    """
    ret_dict = {}
    for k, v in dict_feats.items():
        assert k[-1] in ["g", "b", "a"], "key must end with g, b, or a"
        if k[-1] == "g":
            ret_dict["global"] = v
        elif k[-1] == "b":
            ret_dict["bond"] = v
        elif k[-1] == "a":
            ret_dict["atom"] = v

    return ret_dict


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
