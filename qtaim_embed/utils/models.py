import torch
from typing import Optional, Dict, Any
from dgl import batch


def get_layer_args(
    hparams: Any,
    layer_ind: Optional[int] = None,
    embedding_in: bool = False,
    activation: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Converts hparam dictionary to a dictionary of arguments for a layer.
    Args:
        hparams: hparam dictionary
        layer_ind: layer index
        embedding_in: whether to use embedding input
        activation: activation function
    Returns:
        layer_args: dictionary of arguments for a layer
    """

    assert hparams.conv_fn in [
        "GraphConvDropoutBatch",
        "ResidualBlock",
        "GATConv",
    ], "conv_fn must be either GraphConvDropoutBatch, GATConv or ResidualBlock"

    layer_args = {}
    if hparams.conv_fn == "GraphConvDropoutBatch":
        atom_out = hparams.atom_input_size
        bond_out = hparams.bond_input_size
        global_out = hparams.global_input_size

        atom_in = hparams.atom_input_size
        bond_in = hparams.bond_input_size
        global_in = hparams.global_input_size

        if embedding_in and layer_ind == 0:
            atom_in = hparams.embedding_size
            bond_in = hparams.embedding_size
            global_in = hparams.embedding_size
            atom_out = hparams.hidden_size
            bond_out = hparams.hidden_size
            global_out = hparams.hidden_size

        if layer_ind > 0:
            atom_in = hparams.hidden_size
            bond_in = hparams.hidden_size
            global_in = hparams.hidden_size
            atom_out = hparams.hidden_size
            bond_out = hparams.hidden_size
            global_out = hparams.hidden_size

        if layer_ind == hparams.n_conv_layers - 1:
            if "atom" in hparams.target_dict.keys():
                atom_out = len(hparams.target_dict["atom"])
            if "bond" in hparams.target_dict.keys():
                bond_out = len(hparams.target_dict["bond"])
            if "global" in hparams.target_dict.keys():
                global_out = len(hparams.target_dict["global"])

        layer_args["a2b"] = {
            "in_feats": atom_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2a"] = {
            "in_feats": bond_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2g"] = {
            "in_feats": atom_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2g"] = {
            "in_feats": bond_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2a"] = {
            "in_feats": global_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2b"] = {
            "in_feats": global_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2a"] = {
            "in_feats": atom_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2b"] = {
            "in_feats": bond_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2g"] = {
            "in_feats": global_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

    elif hparams.conv_fn == "ResidualBlock":

        atom_out = hparams.atom_input_size
        bond_out = hparams.bond_input_size
        global_out = hparams.global_input_size

        atom_in = hparams.atom_input_size
        bond_in = hparams.bond_input_size
        global_in = hparams.global_input_size

        # print("layer ind: ", layer_ind)
        if embedding_in:
            atom_in = hparams.embedding_size
            bond_in = hparams.embedding_size
            global_in = hparams.embedding_size
            atom_out = hparams.embedding_size
            bond_out = hparams.embedding_size
            global_out = hparams.embedding_size

        if layer_ind == -1:  # last residual layer has different args
            # print("triggered early stop condition!!!")
            layer_args["a2b_inner"] = {
                "in_feats": atom_in,
                "out_feats": bond_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["b2a_inner"] = {
                "in_feats": bond_in,
                "out_feats": atom_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["a2g_inner"] = {
                "in_feats": atom_in,
                "out_feats": global_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["b2g_inner"] = {
                "in_feats": bond_in,
                "out_feats": global_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["g2a_inner"] = {
                "in_feats": global_in,
                "out_feats": atom_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["g2b_inner"] = {
                "in_feats": global_in,
                "out_feats": bond_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["a2a_inner"] = {
                "in_feats": atom_in,
                "out_feats": atom_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["b2b_inner"] = {
                "in_feats": bond_in,
                "out_feats": bond_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": activation,
                "allow_zero_in_degree": True,
                "dropout": hparams.dropout,
                "batch_norm_tf": hparams.batch_norm_tf,
            }

            layer_args["g2g_inner"] = {
                "in_feats": global_in,
                "out_feats": global_out,
                "norm": hparams.norm,
                "bias": hparams.bias,
                "activation": activation,
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
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2a"] = {
            "in_feats": bond_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2g"] = {
            "in_feats": atom_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2g"] = {
            "in_feats": bond_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2a"] = {
            "in_feats": global_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2b"] = {
            "in_feats": global_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["a2a"] = {
            "in_feats": atom_in,
            "out_feats": atom_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["b2b"] = {
            "in_feats": bond_in,
            "out_feats": bond_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

        layer_args["g2g"] = {
            "in_feats": global_in,
            "out_feats": global_out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

    elif hparams.conv_fn == "GATConv":
        atom_in = hparams.atom_input_size
        bond_in = hparams.bond_input_size
        global_in = hparams.global_input_size

        atom_out = hparams.hidden_size
        bond_out = hparams.hidden_size
        global_out = hparams.hidden_size

        num_heads = hparams.num_heads

        if embedding_in and layer_ind == 0:
            atom_in = hparams.embedding_size
            bond_in = hparams.embedding_size
            global_in = hparams.embedding_size

        if layer_ind > 0:
            atom_in = hparams.hidden_size * num_heads
            bond_in = hparams.hidden_size * num_heads
            global_in = hparams.hidden_size * num_heads

        if layer_ind == hparams.n_conv_layers - 1:
            num_heads = 1
            if "atom" in hparams.target_dict.keys():
                atom_out = len(hparams.target_dict["atom"])
            if "bond" in hparams.target_dict.keys():
                bond_out = len(hparams.target_dict["bond"])
            if "global" in hparams.target_dict.keys():
                global_out = len(hparams.target_dict["global"])

        layer_args["a2b"] = {
            "in_feats": atom_in,
            "out_feats": bond_out,
            "num_heads": num_heads,
            "feat_drop": hparams.feat_drop,
            "attn_drop": hparams.attn_drop,
            "residual": hparams.residual,
            "allow_zero_in_degree": True,
            "bias": hparams.bias,
            "activation": activation,
        }

        layer_args["b2a"] = {
            "in_feats": bond_in,
            "out_feats": atom_out,
            "num_heads": num_heads,
            "feat_drop": hparams.feat_drop,
            "attn_drop": hparams.attn_drop,
            "residual": hparams.residual,
            "allow_zero_in_degree": True,
            "bias": hparams.bias,
            "activation": activation,
        }

        layer_args["a2g"] = {
            "in_feats": atom_in,
            "out_feats": global_out,
            "num_heads": num_heads,
            "feat_drop": hparams.feat_drop,
            "attn_drop": hparams.attn_drop,
            "residual": hparams.residual,
            "allow_zero_in_degree": True,
            "bias": hparams.bias,
            "activation": activation,
        }

        layer_args["b2g"] = {
            "in_feats": bond_in,
            "out_feats": global_out,
            "num_heads": num_heads,
            "feat_drop": hparams.feat_drop,
            "attn_drop": hparams.attn_drop,
            "residual": hparams.residual,
            "allow_zero_in_degree": True,
            "bias": hparams.bias,
            "activation": activation,
        }

        layer_args["g2a"] = {
            "in_feats": global_in,
            "out_feats": atom_out,
            "num_heads": num_heads,
            "feat_drop": hparams.feat_drop,
            "attn_drop": hparams.attn_drop,
            "residual": hparams.residual,
            "allow_zero_in_degree": True,
            "bias": hparams.bias,
            "activation": activation,
        }

        layer_args["g2b"] = {
            "in_feats": global_in,
            "out_feats": bond_out,
            "num_heads": num_heads,
            "feat_drop": hparams.feat_drop,
            "attn_drop": hparams.attn_drop,
            "residual": hparams.residual,
            "allow_zero_in_degree": True,
            "bias": hparams.bias,
            "activation": activation,
        }

        layer_args["a2a"] = {
            "in_feats": atom_in,
            "out_feats": atom_out,
            "num_heads": num_heads,
            "feat_drop": hparams.feat_drop,
            "attn_drop": hparams.attn_drop,
            "residual": hparams.residual,
            "allow_zero_in_degree": True,
            "bias": hparams.bias,
            "activation": activation,
        }

        layer_args["b2b"] = {
            "in_feats": bond_in,
            "out_feats": bond_out,
            "num_heads": num_heads,
            "feat_drop": hparams.feat_drop,
            "attn_drop": hparams.attn_drop,
            "residual": hparams.residual,
            "allow_zero_in_degree": True,
            "bias": hparams.bias,
            "activation": activation,
        }

        layer_args["g2g"] = {
            "in_feats": global_in,
            "out_feats": global_out,
            "num_heads": num_heads,
            "feat_drop": hparams.feat_drop,
            "attn_drop": hparams.attn_drop,
            "residual": hparams.residual,
            "allow_zero_in_degree": True,
            "bias": hparams.bias,
            "activation": activation,
        }

        # print("layer args: ", layer_args)

    return layer_args


def get_layer_args_homo(
    hparams: Any,
    layer_ind: Optional[int] = None,
    embedding_in: bool = False,
    activation: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Converts hparam dictionary to a dictionary of arguments for a layer.

    Args:
        hparams (Any): Hyperparameter dictionary.
        layer_ind (Optional[int]): Layer index. Defaults to None.
        embedding_in (bool): Whether to use embedding input. Defaults to False.
        activation (Optional[Any]): Activation function. Defaults to None.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of arguments for a layer.
    """

    assert hparams.conv_fn in [
        "GraphConvDropoutBatch",
        "ResidualBlock",
        "GATConv",
        "GraphSAGE",
    ], "conv_fn must be either GraphConvDropoutBatch, GATConv or ResidualBlock"

    layer_args = {}

    if hparams.conv_fn == "GraphConvDropoutBatch":

        if embedding_in:
            in_feats = hparams.embedding_size
        else:
            in_feats = hparams.input_size

        if layer_ind > 0:
            in_feats = hparams.hidden_size

        out = hparams.hidden_size

        layer_args["conv"] = {
            "in_feats": in_feats,
            "out_feats": out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
        }

    elif hparams.conv_fn == "ResidualBlock":

        if embedding_in:
            in_feats = hparams.embedding_size
        else:
            in_feats = hparams.input_size

        if layer_ind > 0:
            in_feats = hparams.hidden_size
        # print("layer ind: ", layer_ind)
        out = hparams.hidden_size

        layer_args["layer_args"] = {
            "in_feats": in_feats,
            "out_feats": out,
            "norm": hparams.norm,
            "bias": hparams.bias,
            "activation": activation,
            "allow_zero_in_degree": True,
            "dropout": hparams.dropout,
            "batch_norm_tf": hparams.batch_norm_tf,
            "embedding_size": hparams.embedding_size,
        }

    elif hparams.conv_fn == "GATConv":
        in_feats = hparams.input_size
        num_heads = hparams.num_heads

        if layer_ind > 0:
            in_feats = hparams.hidden_size

            if num_heads > 1:
                in_feats = hparams.hidden_size * num_heads

        else:
            if embedding_in:
                in_feats = hparams.embedding_size

        if layer_ind == hparams.n_conv_layers - 1:
            num_heads = 1
            # TESTING
            # out = 1

        out = hparams.hidden_size

        layer_args["conv"] = {
            "in_feats": in_feats,
            "out_feats": out,
            "num_heads": num_heads,
            "feat_drop": hparams.feat_drop,
            "attn_drop": hparams.attn_drop,
            "residual": hparams.residual,
            "allow_zero_in_degree": True,
            "bias": hparams.bias,
            "activation": activation,
        }

    elif hparams.conv_fn == "GraphSAGE":
        in_feats = hparams.input_size

        if embedding_in:
            in_feats = hparams.embedding_size

        if layer_ind > 0:
            in_feats = hparams.hidden_size

        out = hparams.hidden_size

        layer_args["conv"] = {
            "in_feats": in_feats,
            "out_feats": out,
            "aggregator_type": hparams.aggregator_type,
            "bias": hparams.bias,
            "activation": activation,
            "feat_drop": hparams.dropout,
            "norm": None,
        }

    return layer_args


def link_fmt_to_node_fmt(
    dict_feats: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, Any]]:
    """
    Converts a dictionary of features from link format to node format.
    The input dictionary should have keys ending with 'g', 'b', or 'a',
    representing global, bond, and atom features, respectively.
    Args:
        dict_feats (dict): Dictionary of features with keys ending with 'g', 'b', or 'a'.
    Returns:
        dict: Dictionary with keys 'global', 'bond', and 'atom' containing the corresponding features.
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


def _split_batched_output(graph: batch, value: torch.Tensor, key: str = "global"):
    """
    Split a tensor into `num_graphs` chunks, the size of each chunk equals the
    number of bonds in the graph.

    Args:
        graph (dgl.DGLGraph): Batched graph.
        value (torch.Tensor): Tensor to be split.
        key (str): Key for the graph.

    Returns:
        list of tensor.

    """
    n_nodes = graph.batch_num_nodes(key)
    # convert to tuple
    n_nodes = tuple(n_nodes)
    return torch.split(value, n_nodes)
