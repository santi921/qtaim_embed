from qtaim_embed.data.grapher import HeteroCompleteGraphFromMolWrapper
from qtaim_embed.data.featurizer import (
    BondAsNodeGraphFeaturizerGeneral,
    AtomFeaturizerGraphGeneral,
    GlobalFeaturizerGraph,
)


def get_grapher(
    element_set,
    atom_keys=[],
    bond_keys=[],
    global_keys=[],
    allowed_ring_size=[],
    allowed_charges=None,
    allowed_spins=None,
    self_loop=True,
    atom_featurizer_tf=True,
    bond_featurizer_tf=True,
    global_featurizer_tf=True,
):
    if not atom_featurizer_tf:
        atom_featurizer = None
    else:
        atom_featurizer = AtomFeaturizerGraphGeneral(
            selected_keys=atom_keys,
            element_set=element_set,
            allowed_ring_size=allowed_ring_size,
        )

    if not bond_featurizer_tf:
        bond_featurizer = None
    else:
        bond_featurizer = BondAsNodeGraphFeaturizerGeneral(
            selected_keys=bond_keys,
            allowed_ring_size=allowed_ring_size,
        )

    if not global_featurizer_tf:
        global_featurizer = None
    else:
        global_featurizer = GlobalFeaturizerGraph(
            selected_keys=global_keys,
            allowed_charges=allowed_charges,
            allowed_spins=allowed_spins,
        )

    grapher = HeteroCompleteGraphFromMolWrapper(
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer,
        global_featurizer=global_featurizer,
        self_loop=self_loop,
    )
    return grapher
