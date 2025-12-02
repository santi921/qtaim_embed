from rdkit import Chem
import numpy as np
import networkx as nx
import copy
from typing import Optional, Dict, Any, List, Tuple, Union

from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core.structure import Molecule

from qtaim_embed.utils.files import create_directory, to_path


class MoleculeWrapper:
    """
    A wrapper of pymatgen Molecule, MoleculeGraph, rdkit Chem.Mol... to make it
    easier to use molecules.

    Arguments:
        mol_graph (MoleculeGraph): pymatgen molecule graph instance
        free_energy (float): free energy of the molecule
        id (str): (unique) identification of the molecule
        functional_group (str): functional group of the molecule
        bonds (list of tuple): each tuple is a bond (atom indices)
        non_metal_bonds (list of tuple): each tuple is a bond (atom indices) between
            non-metal atoms
        atom_features (dict): features of atoms
        bond_features (dict): features of bonds
        global_features (dict): features of the molecule
        original_atom_ind (list): original atom indices
        original_bond_mapping (list): original bond indices
    """

    def __init__(
        self,
        mol_graph: MoleculeGraph,
        functional_group: Optional[str] = None,
        free_energy: Optional[float] = None,
        id: Optional[str] = None,
        bonds: Optional[List[Tuple[int, int]]] = None,
        non_metal_bonds: Optional[List[Tuple[int, int]]] = None,
        atom_features: Optional[Dict[str, Any]] = {},
        bond_features: Optional[Dict[str, Any]] = {},
        global_features: Optional[Dict[str, Any]] = {},
        original_atom_ind: Optional[List[int]] = None,
        original_bond_mapping: Optional[List[int]] = None,
    ):
        self.mol_graph = mol_graph
        self.pymatgen_mol = mol_graph.molecule
        self.manual_bonds = bonds
        self.nonmetal_bonds = non_metal_bonds
        self.free_energy = free_energy
        self.functional_group = functional_group
        self.id = id
        self.atom_features = atom_features
        self.bond_features = bond_features
        self.global_features = global_features
        self.original_atom_ind = original_atom_ind
        self.original_bond_mapping = original_bond_mapping
        # print("final bond features:")
        # print(self.bond_features)
        # set when corresponding method is called
        self._rdkit_mol = None
        self._fragments = None
        self._isomorphic_bonds = None

    @property
    def charge(self) -> int:
        """
        Returns:
            int: charge of the molecule
        """
        return self.pymatgen_mol.charge

    @property
    def formula(self) -> str:
        """
        Returns:
            str: chemical formula of the molecule, e.g. H2CO3.
        """
        return self.pymatgen_mol.composition.alphabetical_formula.replace(" ", "")

    @property
    def composition_dict(self) -> Dict[str, int]:
        """
        Returns:
            dict: with chemical species as key and number of the species as value.
        """
        d = self.pymatgen_mol.composition.as_dict()
        return {k: int(v) for k, v in d.items()}

    @property
    def weight(self) -> float:
        """
        Returns:
            int: molecule weight
        """
        return self.pymatgen_mol.composition.weight

    @property
    def num_atoms(self) -> int:
        """
        Returns:
            int: number of atoms in molecule
        """
        return len(self.pymatgen_mol)

    @property
    def species(self) -> List[str]:
        """
        Species of atoms. Order is the same as self.atoms.
        Returns:
            list: Species string.
        """
        return [str(s) for s in self.pymatgen_mol.species]

    @property
    def coords(self) -> np.ndarray:
        """
        Returns:
            2D array: of shape (N, 3) where N is the number of atoms.
        """
        return np.asarray(self.pymatgen_mol.cart_coords)

    @property
    def bonds(self) -> Dict[Tuple[int, int], Any]:
        """
        Returns:
            dict: with bond index (a tuple of atom indices) as the key and and bond
                attributes as the value.
        """
        if self.manual_bonds is not None:
            #    print("getting manual bonds")
            # return self.manual_bonds
            return {tuple(sorted([i, j])): {} for i, j in self.manual_bonds}
        else:
            return {
                tuple(sorted([i, j])): attr for i, j, attr in self.graph.edges.data()
            }

    @property
    def graph(self) -> nx.Graph:
        """
        Returns:
            networkx graph used by mol_graph
        """
        return self.mol_graph.graph

    def is_atom_in_ring(self, atom: int) -> bool:
        """
        Whether an atom in ring.

        Args:
            atom (int): atom index

        Returns:
            bool: atom in ring or not
        """
        ring_info = self.mol_graph.find_rings()
        ring_atoms = set([atom for ring in ring_info for bond in ring for atom in bond])
        return atom in ring_atoms

    def is_bond_in_ring(self, bond: Tuple[int, int]) -> bool:
        """
        Whether a bond in ring.

        Args:
            bond (tuple): bond index

        Returns:
            bool: bond in ring or not
        """
        ring_info = self.mol_graph.find_rings()
        ring_bonds = set([tuple(sorted(bond)) for ring in ring_info for bond in ring])
        return tuple(sorted(bond)) in ring_bonds

    def get_sdf_bond_indices(self, zero_based: bool = False, sdf: Optional[str] = None) -> List[Tuple[int, int]]:
        """
        Get the indices of bonds as specified in the sdf file.

        zero_based (bool): If True, the atom index will be converted to zero based.
        sdf (str): the sdf string for parsing. If None, it is created from the mol.

        Returns:
            list of tuple: each tuple specifies a bond.
        """
        sdf = sdf or self.write()

        lines = sdf.split("\n")
        start = end = 0
        for i, ln in enumerate(lines):
            if "BEGIN BOND" in ln:
                start = i + 1
            if "END BOND" in ln:
                end = i
                break

        bonds = [
            tuple(sorted([int(i) for i in ln.split()[4:6]])) for ln in lines[start:end]
        ]

        if zero_based:
            bonds = [(b[0] - 1, b[1] - 1) for b in bonds]

        return bonds

    def get_sdf_bond_indices_v2000(self, sdf: Optional[str] = None) -> List[Tuple[int, int]]:
        """
        Get the indices of bonds as specified in the sdf file.

        Returns:
            list of tuple: each tuple specifies a bond.
        """
        sdf = sdf or self.write(v3000=False)
        lines = sdf.split("\n")
        split_3 = lines[3].split()
        natoms = int(split_3[0])
        nbonds = int(split_3[1])
        bonds = []
        for line in lines[4 + natoms : 4 + natoms + nbonds]:
            bonds.append(tuple(sorted([int(i) for i in line.split()[:2]])))
        return bonds

    def subgraph_atom_mapping(self, bond: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """
        Find the atoms in the two subgraphs by breaking a bond in a molecule.

        Returns:
            tuple of list: each list contains the atoms in one subgraph.
        """

        original = copy.deepcopy(self.mol_graph)
        original.break_edge(bond[0], bond[1], allow_reverse=True)

        # A -> B breaking
        if nx.is_weakly_connected(original.graph):
            mapping = list(range(self.num_atoms))
            return mapping, mapping
        # A -> B + C breaking
        else:
            components = nx.weakly_connected_components(original.graph)
            nodes = [original.graph.subgraph(c).nodes for c in components]
            mapping = tuple([sorted(list(n)) for n in nodes])
            if len(mapping) != 2:
                raise Exception("Mol not split into two parts")
            return mapping

    def find_ring(self, by_species: bool = False) -> List[List[Union[int, str]]]:
        """
        Find all rings in the molecule.

        Args:
            by_species (bool): If False, the rings will be denoted by atom indices. If
                True, denoted by atom species.

        Returns:
            list of list: each inner list holds the atoms (index or specie) of a ring.
        """
        rings = self.mol_graph.find_rings()

        rings_once_per_atom = []
        for r in rings:
            # the ring is given by the connectivity info. For example, for a 1-2-3 ring,
            # r would be something like [(1,2), (2,3), (3,1)]
            # here we remove the repeated atoms and let each atom appear only once
            atoms = []
            for i in r:
                atoms.extend(i)
            atoms = list(set(atoms))
            if by_species:
                atoms = [self.species[j] for j in atoms]
            rings_once_per_atom.append(atoms)

        return rings_once_per_atom

    def write(
        self,
        filename: Optional[str] = None,
        name: Optional[str] = None,
        format: str = "sdf",
        kekulize: bool = True,
        v3000: bool = True,
    ) -> Optional[str]:
        """Write a molecule to file or as string using rdkit.

        Args:
            filename (str): name of the file to write the output. If None, return the
                output as string.
            name (str): name of a molecule. If `file_format` is sdf, this is the first
                line the molecule block in the sdf.
            format (str): format of the molecule, supporting: sdf, pdb, and smi.
            kekulize (bool): whether to kekulize the mol if format is `sdf`
            v3000 (bool): whether to force v3000 form if format is `sdf`
        """
        if filename is not None:
            create_directory(filename)
            filename = str(to_path(filename))

        name = str(self.id) if name is None else name
        self.rdkit_mol.SetProp("_Name", name)

        if format == "sdf":
            if filename is None:
                sdf = Chem.MolToMolBlock(
                    self.rdkit_mol, kekulize=kekulize, forceV3000=v3000
                )
                return sdf + "$$$$\n"
            else:
                return Chem.MolToMolFile(
                    self.rdkit_mol, filename, kekulize=kekulize, forceV3000=v3000
                )
        elif format == "pdb":
            if filename is None:
                sdf = Chem.MolToPDBBlock(self.rdkit_mol)
                return sdf + "$$$$\n"
            else:
                return Chem.MolToPDBFile(self.rdkit_mol, filename)
        elif format == "smi":
            return Chem.MolToSmiles(self.rdkit_mol)
        else:
            raise ValueError(f"format {format} currently not supported")

    def write_custom(self, index: int) -> str:
        bonds = self.bonds
        bond_count = len(bonds)
        atom_count = len(self.pymatgen_mol.sites)
        sdf = ""
        name = "{}_{}_{}_{}_index-{}".format(
            self.id, self.formula, self.charge, self.free_energy, index
        )
        sdf += name + "\n"
        sdf += "     RDKit          3D\n\n"
        sdf += "  0  0  0  0  0  0  0  0  0  0999 V3000\n"
        sdf += "M  V30 BEGIN CTAB\n"
        sdf += "M  V30 COUNTS {} {} 0 0 0\n".format(atom_count, bond_count)
        sdf += "M  V30 BEGIN ATOM\n"
        # this is done
        for ind in range(len(self.pymatgen_mol.sites)):
            charge = self.rdkit_mol.GetAtomWithIdx(ind).GetFormalCharge()
            element = self.pymatgen_mol[ind].as_dict()["species"][0]["element"]
            x, y, z = self.pymatgen_mol[ind].as_dict()["xyz"]
            if charge != 0:
                sdf += "M  V30 {} {} {:.5f} {:.5f} {:.5f} 0 CHG={}\n".format(
                    ind + 1, element, x, y, z, charge
                )
            else:
                sdf += "M  V30 {} {} {:.5f} {:.5f} {:.5f} 0\n".format(
                    ind + 1, element, x, y, z
                )

        sdf += "M  V30 END ATOM\n"
        if atom_count > 1:
            sdf += "M  V30 BEGIN BOND\n"
            """
            if(bond_count == 0): 
                a_atom = self.pymatgen_mol[0].as_dict()["species"][0]['element']
                b_atom = self.pymatgen_mol[1].as_dict()["species"][0]['element']
                if(a_atom=='H' or b_atom=='H' ): order = 1
                if(a_atom=='F' or b_atom=='F' or a_atom == 'Cl' or b_atom == 'Cl'): order = 1
                if(a_atom=='N' or b_atom=='N'): order = 3
                if(a_atom=="O" or b_atom=='O'): order = 2
                sdf += "M  V30 {} {} {} {}\n".format(1, order, 1, 2)
            """
            for ind, bond in enumerate(bonds):
                double_cond = False
                a, b = bond
                try:
                    double_cond = "DOUBLE" == str(
                        self.rdkit_mol.GetBondBetweenAtoms(a, b).GetBondType()
                    )
                except:
                    pass
                if double_cond:
                    order = 2
                else:
                    order = 1
                sdf += "M  V30 {} {} {} {}\n".format(ind + 1, order, a + 1, b + 1)

            sdf += "M  V30 END BOND\n"
        sdf += "M  V30 END CTAB\n"
        sdf += "M  END\n"
        sdf += "$$$$\n"

        return sdf

    def draw(self, filename: Optional[str] = None, show_atom_idx: bool = False) -> Any:
        """
        Draw the molecule.

        Args:
            filename (str): path to the save the generated image. If `None` the
                molecule is returned and can be viewed in Jupyter notebook.
        """
        m = copy.deepcopy(self.rdkit_mol)
        AllChem.Compute2DCoords(m)

        if show_atom_idx:
            for a in m.GetAtoms():
                a.SetAtomMapNum(a.GetIdx() + 1)
        # d.drawOptions().addAtomIndices = True

        if filename is None:
            return m
        else:
            create_directory(filename)
            filename = str(to_path(filename))
            Draw.MolToFile(m, filename)

    def draw_with_bond_note(
        self,
        bond_note: Dict[Tuple[int, int], Any],
        filename: str = "mol.png",
        show_atom_idx: bool = True,
    ) -> None:
        """
        Draw molecule using rdkit and show bond annotation, e.g. bond energy.

        Args:
            bond_note (dict): {bond_index: note}. The note to show for the
                corresponding bond.
            filename (str): path to the save the generated image. If `None` the
                molecule is returned and can be viewed in Jupyter notebook.
        """
        m = self.draw(show_atom_idx=show_atom_idx)

        # set bond annotation
        highlight_bonds = []
        for bond, note in bond_note.items():
            if isinstance(note, (float, np.floating)):
                note = "{:.3g}".format(note)
            idx = m.GetBondBetweenAtoms(*bond).GetIdx()
            m.GetBondWithIdx(idx).SetProp("bondNote", note)
            highlight_bonds.append(idx)

        # set highlight color
        bond_colors = {b: (192 / 255, 192 / 255, 192 / 255) for b in highlight_bonds}

        d = rdMolDraw2D.MolDraw2DCairo(400, 300)

        # smaller font size
        d.SetFontSize(0.8 * d.FontSize())

        rdMolDraw2D.PrepareAndDrawMolecule(
            d, m, highlightBonds=highlight_bonds, highlightBondColors=bond_colors
        )
        d.FinishDrawing()

        create_directory(filename)
        with open(to_path(filename), "wb") as f:
            f.write(d.GetDrawingText())

    def pack_features(self, broken_bond: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        feats = dict()
        feats["charge"] = self.charge
        return feats

    def __expr__(self) -> str:
        return f"{self.id}_{self.formula}"

    def __str__(self) -> str:
        return self.__expr__()


def create_wrapper_mol_from_atoms_and_bonds(
    species: List[str],
    coords: Union[List[List[float]], np.ndarray],
    bonds: List[Tuple[int, int]],
    charge: int = 0,
    free_energy: Optional[float] = None,
    functional_group: Optional[str] = None,
    identifier: Optional[str] = None,
    original_atom_ind: Optional[List[int]] = None,
    original_bond_ind: Optional[List[int]] = None,
    atom_features: Optional[Dict[str, Any]] = {},
    bond_features: Optional[Dict[str, Any]] = {},
    global_features: Optional[Dict[str, Any]] = {},
) -> MoleculeWrapper:
    """
    Create a :class:`MoleculeWrapper` from atoms and bonds.

    Args:
        species (list of str): atom species str
        coords (2D array): positions of atoms
        bonds (list of tuple): each tuple is a bond (atom indices)
        charge (int): charge of the molecule
        free_energy (float): free energy of the molecule
        identifier (str): (unique) identifier of the molecule
        original_atom_ind(list of indices):  atoms, in order

    Returns:
        MoleculeWrapper instance
    """

    pymatgen_mol = Molecule(species, coords, charge)
    bonds = {tuple(sorted(b)): None for b in bonds}
    mol_graph = MoleculeGraph.with_edges(pymatgen_mol, bonds)
    mol_wrapper = MoleculeWrapper(
        mol_graph,
        free_energy=free_energy,
        functional_group=functional_group,
        id=identifier,
        original_atom_ind=original_atom_ind,
        original_bond_mapping=original_bond_ind,
        atom_features=atom_features,
        bond_features=bond_features,
        global_features=global_features,
    )

    return mol_wrapper
