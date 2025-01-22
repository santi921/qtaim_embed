import dgl


class HeteroCompleteGraphFromMolWrapper:
    """ """

    def __init__(
        self,
        atom_featurizer=None,
        bond_featurizer=None,
        global_featurizer=None,
        self_loop=True,
    ):
        self.self_loop = self_loop
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer
        self.global_featurizer = global_featurizer

    def build_graph(self, mol):
        bonds = list(mol.bonds.keys())
        # print("bonds", bonds)
        num_bonds = len(bonds)
        num_atoms = len(mol.coords)
        a2b = []
        b2a = []
        if num_bonds == 0:
            #print("num bonds 1 examples!!!!!")
            num_bonds = 1
            a2b = [(0, 0)]
            b2a = [(0, 0)]

        else:
            a2b = []
            b2a = []
            for b in range(num_bonds):
                u = bonds[b][0]
                v = bonds[b][1]
                b2a.extend([[b, u], [b, v]])
                a2b.extend([[u, b], [v, b]])

        a2g = [(a, 0) for a in range(num_atoms)]
        g2a = [(0, a) for a in range(num_atoms)]
        b2g = [(b, 0) for b in range(num_bonds)]
        g2b = [(0, b) for b in range(num_bonds)]

        edges_dict = {
            ("atom", "a2b", "bond"): a2b,
            ("bond", "b2a", "atom"): b2a,
            ("atom", "a2g", "global"): a2g,
            ("global", "g2a", "atom"): g2a,
            ("bond", "b2g", "global"): b2g,
            ("global", "g2b", "bond"): g2b,
        }
        if self.self_loop:
            a2a = [(i, i) for i in range(num_atoms)]
            b2b = [(i, i) for i in range(num_bonds)]
            g2g = [(0, 0)]
            edges_dict.update(
                {
                    ("atom", "a2a", "atom"): a2a,
                    ("bond", "b2b", "bond"): b2b,
                    ("global", "g2g", "global"): g2g,
                }
            )

        g = dgl.heterograph(edges_dict)
        # g = dgl.add_self_loop(g)
        # add name
        g.mol_name = mol.id

        return g

    def featurize(self, g, mol, ret_feat_names=False, **kwargs):
        if self.atom_featurizer is not None:
            feat_dict, feat_atom = self.atom_featurizer(mol, **kwargs)
            g.nodes["atom"].data.update(feat_dict)

        if self.bond_featurizer is not None:
            feat_dict, feat_bond = self.bond_featurizer(mol, **kwargs)
            g.nodes["bond"].data.update(feat_dict)

        if self.global_featurizer is not None:
            feat_dict, globe_feat = self.global_featurizer(mol, **kwargs)
            g.nodes["global"].data.update(feat_dict)

        if ret_feat_names:
            feat_names = {}
            if self.atom_featurizer is not None:
                feat_names["atom"] = feat_atom
            if self.bond_featurizer is not None:
                feat_names["bond"] = feat_bond
            if self.global_featurizer is not None:
                feat_names["global"] = globe_feat
            return g, feat_names

        return g
