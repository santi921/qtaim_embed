import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger

import openbabel as ob
import pandas as pd

ob_log_handler = ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
RDLogger.DisableLog("rdApp.*")


def get_molecule_translation_dimenet_qm8(df, file):
    z_list = []
    n_list = []
    r_list = []
    id_list = []
    e1_cc2 = []
    e2_cc2 = []
    f1_cc2 = []
    f2_cc2 = []
    e1_pbe0 = []
    e2_pbe0 = []
    f1_pbe0 = []
    f2_pbe0 = []
    e1_cam = []
    e2_cam = []
    f1_cam = []
    f2_cam = []

    works_count = 0
    for ind, row in df.iterrows():
        combined_graph = row["molecule"]
        # pmg = Molecule.from_dict(combined_graph)
        # xyz = combined_graph.to(fmt="sdf")
        # mol = Chem.MolFromMolBlock(xyz, removeHs=False, sanitize=True)
        Z = combined_graph.atomic_numbers
        R = combined_graph.cart_coords
        N = len(combined_graph.atomic_numbers)

        [z_list.append(z) for z in Z]
        n_list.append(N)
        # r_list.append(R)
        [r_list.append(r) for r in R]
        id_list.append(row["ids"])
        e1_cc2.append(row["E1-CC2"])
        e2_cc2.append(row["E2-CC2"])
        f1_cc2.append(row["f1-CC2"])
        f2_cc2.append(row["f2-CC2"])
        e1_pbe0.append(row["E1-PBE0"])
        e2_pbe0.append(row["E2-PBE0"])
        f1_pbe0.append(row["f1-PBE0"])
        f2_pbe0.append(row["f2-PBE0"])
        e1_cam.append(row["E1-CAM"])
        e2_cam.append(row["E2-CAM"])
        f1_cam.append(row["f1-CAM"])
        f2_cam.append(row["f2-CAM"])
        works_count += 1
    print(works_count)
    # print(z_list)
    # print(n_list)

    # format as npz file
    np.savez(
        file,
        Z=np.array(z_list),
        N=np.array(n_list),
        R=np.array(r_list),
        id=np.array(id_list),
        E1_CC2=np.array(e1_cc2),
        E2_CC2=np.array(e2_cc2),
        f1_CC2=np.array(f1_cc2),
        f2_CC2=np.array(f2_cc2),
        E1_PBE0=np.array(e1_pbe0),
        E2_PBE0=np.array(e2_pbe0),
        f1_PBE0=np.array(f1_pbe0),
        f2_PBE0=np.array(f2_pbe0),
        E1_CAM=np.array(e1_cam),
        E2_CAM=np.array(e2_cam),
        f1_CAM=np.array(f1_cam),
        f2_CAM=np.array(f2_cam),
    )


def get_molecule_translation_dimenet_qm9(df, file):
    z_list = []
    n_list = []
    r_list = []
    homo_list = []
    lumo_list = []
    gap_list = []
    U0_list = []
    id_list = []

    works_count = 0
    for ind, row in df.iterrows():
        combined_graph = row["molecule"]
        # pmg = Molecule.from_dict(combined_graph)
        # xyz = combined_graph.to(fmt="sdf")
        # mol = Chem.MolFromMolBlock(xyz, removeHs=False, sanitize=True)
        Z = combined_graph.atomic_numbers
        R = combined_graph.cart_coords
        N = len(combined_graph.atomic_numbers)

        [z_list.append(z) for z in Z]
        n_list.append(N)
        # r_list.append(R)
        [r_list.append(r) for r in R]
        id_list.append(row["ids"])
        homo_list.append(row["homo"])
        lumo_list.append(row["lumo"])
        gap_list.append(row["gap"])
        U0_list.append(row["u0"])
        works_count += 1
    print(works_count)
    # print(z_list)
    # print(n_list)

    # format as npz file
    np.savez(
        file,
        Z=np.array(z_list),
        N=np.array(n_list),
        R=np.array(r_list),
        id=np.array(id_list),
        homo=np.array(homo_list),
        lumo=np.array(lumo_list),
        gap=np.array(gap_list),
        U0=np.array(U0_list),
    )


def get_reaction_smarts_from_df_qm8(df, extra_feats=[]):
    molecule_list = []

    label_list = [
        "E1-CC2",
        "E2-CC2",
        "f1-CC2",
        "f2-CC2",
        "E1-PBE0",
        "E2-PBE0",
        "f1-PBE0",
        "f2-PBE0",
        "E1-CAM",
        "E2-CAM",
        "f1-CAM",
        "f2-CAM",
    ]
    e1_cc2 = []
    e2_cc2 = []
    f1_cc2 = []
    f2_cc2 = []
    e1_pbe0 = []
    e2_pbe0 = []
    f1_pbe0 = []
    f2_pbe0 = []
    e1_cam = []
    e2_cam = []
    f1_cam = []
    f2_cam = []
    extra_feat_list = []

    works_count = 0
    for ind, row in df.iterrows():
        combined_graph = row["molecule"]
        # pmg = Molecule.from_dict(combined_graph)
        xyz = combined_graph.to(fmt="sdf")
        mol = Chem.MolFromMolBlock(xyz, removeHs=False, sanitize=True)

        if mol is not None:
            rxn_smiles_mapped_manual = Chem.MolToSmiles(mol)

            molecule_list.append(rxn_smiles_mapped_manual)
            e1_cc2.append(row["E1-CC2"])
            e2_cc2.append(row["E2-CC2"])
            f1_cc2.append(row["f1-CC2"])
            f2_cc2.append(row["f2-CC2"])
            e1_pbe0.append(row["E1-PBE0"])
            e2_pbe0.append(row["E2-PBE0"])
            f1_pbe0.append(row["f1-PBE0"])
            f2_pbe0.append(row["f2-PBE0"])
            e1_cam.append(row["E1-CAM"])
            e2_cam.append(row["E2-CAM"])
            f1_cam.append(row["f1-CAM"])
            f2_cam.append(row["f2-CAM"])
            if extra_feats != []:
                extra_feats_temp = []
                for feat_extra in extra_feats:
                    extra_feats_temp.append(row[feat_extra])
                extra_feat_list.append(extra_feats_temp)

            works_count += 1

    if extra_feats == []:
        return (
            molecule_list,
            e1_cc2,
            e2_cc2,
            f1_cc2,
            f2_cc2,
            e1_pbe0,
            e2_pbe0,
            f1_pbe0,
            f2_pbe0,
            e1_cam,
            e2_cam,
            f1_cam,
            f2_cam,
        )
    else:
        return (
            molecule_list,
            e1_cc2,
            e2_cc2,
            f1_cc2,
            f2_cc2,
            e1_pbe0,
            e2_pbe0,
            f1_pbe0,
            f2_pbe0,
            e1_cam,
            e2_cam,
            f1_cam,
            f2_cam,
            extra_feat_list,
        )


def get_reaction_smarts_from_df_qm9(df, extra_feats=[]):
    molecule_list = []

    # label_list = [
    #    "homo",
    #    "lumo",
    #    "gap",
    #    "U0",
    # ]
    homo_list = []
    lumo_list = []
    gap_list = []
    U0_list = []
    extra_feats_list = []

    works_count = 0
    for ind, row in df.iterrows():
        combined_graph = row["molecule"]
        # pmg = Molecule.from_dict(combined_graph)
        xyz = combined_graph.to(fmt="sdf")
        mol = Chem.MolFromMolBlock(xyz, removeHs=False, sanitize=True)

        if mol is not None:
            rxn_smiles_mapped_manual = Chem.MolToSmiles(mol)

            molecule_list.append(rxn_smiles_mapped_manual)
            homo_list.append(row["homo"])
            lumo_list.append(row["lumo"])
            gap_list.append(row["gap"])
            U0_list.append(row["u0"])
            works_count += 1
            if extra_feats != []:
                extra_feats_temp = []
                for feat_extra in extra_feats:
                    extra_feats_temp.append(row[feat_extra])
                extra_feats_list.append(extra_feats_temp)

    if extra_feats == []:
        return (molecule_list, homo_list, lumo_list, gap_list, U0_list)
    else:
        return (
            molecule_list,
            homo_list,
            lumo_list,
            gap_list,
            U0_list,
            extra_feats_list,
        )


def write_csv_qm8(
    file,
    file_feats,
    smi_list,
    e1_cc2,
    e2_cc2,
    f1_cc2,
    f2_cc2,
    e1_cam,
    e2_cam,
    f1_cam,
    f2_cam,
    e1_pbe,
    e2_pbe,
    f1_pbe,
    f2_pbe,
    extra_feats=[],
    extra_feat_names=[],
    header=True,
):
    with open(file, "w") as f:
        if header:
            header_str = "AAM,CC2_E1,CC2_E2,CC2_f1,CC2_f2,CAM_E1,CAM_E2,CAM_f1,CAM_f2,PBE_E1,PBE_E2,PBE_f1,PBE_f2\n"
            f.write(header_str)

        print(".... > writing csv")

        for i in tqdm(range(len(smi_list))):
            str_line = (
                smi_list[i]
                + ","
                + str(e1_cc2[i])
                + ","
                + str(e2_cc2[i])
                + ","
                + str(f1_cc2[i])
                + ","
                + str(f2_cc2[i])
                + ","
                + str(e1_cam[i])
                + ","
                + str(e2_cam[i])
                + ","
                + str(f1_cam[i])
                + ","
                + str(f2_cam[i])
                + ","
                + str(e1_pbe[i])
                + ","
                + str(e2_pbe[i])
                + ","
                + str(f1_pbe[i])
                + ","
                + str(f2_pbe[i])
                + "\n"
            )
            f.write(str_line)

    header_list = []
    if header:
        # header_str = "AAM"
        if extra_feats != []:
            for feat in extra_feat_names:
                header_list.append(feat)

    rows = []
    smiles_list = []
    for i in tqdm(range(len(smi_list))):
        row = []
        str_line = smi_list[i]
        smiles_list.append(str_line)
        if extra_feats != []:
            for feat in extra_feats[i]:
                row.append(np.array(feat))
        rows.append(row)
    # convert to dataframe
    # remove \n from header
    df = pd.DataFrame(rows, index=smiles_list, columns=header_list)
    df.to_pickle(file_feats)


def write_csv_qm9(
    file,
    file_feats,
    smi_list,
    homo_list,
    lumo_list,
    gap_list,
    u0_list,
    header=True,
    extra_feats=[],
    extra_feat_names=[],
):
    with open(file, "w") as f:
        if header:
            header_str = "AAM,homo,lumo,gap,U0\n"
            f.write(header_str)
        print(".... > writing csv")

        for i in tqdm(range(len(smi_list))):
            str_line = (
                smi_list[i]
                + ","
                + str(homo_list[i])
                + ","
                + str(lumo_list[i])
                + ","
                + str(gap_list[i])
                + ","
                + str(u0_list[i])
                + "\n"
            )

            f.write(str_line)

    if header:
        header_list = []
        if extra_feats != []:
            for feat in extra_feat_names:
                header_list.append(feat)

    rows = []
    smiles_list = []
    for i in tqdm(range(len(smi_list))):
        row = []
        str_line = smi_list[i]
        smiles_list.append(str_line)
        # row.append(str_line)
        # row.append(str_line)
        if extra_feats != []:
            for feat in extra_feats[i]:
                row.append(np.array(feat))
        rows.append(row)
    # convert to dataframe

    df = pd.DataFrame(rows, index=smiles_list, columns=header_list)
    df.to_pickle(file_feats)


def translate_qm9(df, out_file, out_feats_file, extra_feats=[]):
    # print(extra_feats)
    (
        smi_list,
        homo_list,
        lumo_list,
        gap_list,
        u0_list,
        extra_feat_list,
    ) = get_reaction_smarts_from_df_qm9(df, extra_feats=extra_feats)

    write_csv_qm9(
        out_file,
        out_feats_file,
        smi_list,
        homo_list,
        lumo_list,
        gap_list,
        u0_list,
        extra_feats=extra_feat_list,
        extra_feat_names=extra_feats,
    )


def translate_qm8(df, out_file, out_feats_file, extra_feats=[]):
    (
        smi_list,
        e1_cc2_list,
        e2_cc2_list,
        f1_cc2_list,
        f2_cc2_list,
        e1_pbe0_list,
        e2_pbe0_list,
        f1_pbe0_list,
        f2_pbe0_list,
        e1_cam_list,
        e2_cam_list,
        f1_cam_list,
        f2_cam_list,
        extra_feat_list,
    ) = get_reaction_smarts_from_df_qm8(df, extra_feats=extra_feats)

    write_csv_qm8(
        out_file,
        out_feats_file,
        smi_list,
        e1_cc2_list,
        e2_cc2_list,
        f1_cc2_list,
        f2_cc2_list,
        e1_cam_list,
        e2_cam_list,
        f1_cam_list,
        f2_cam_list,
        e1_pbe0_list,
        e2_pbe0_list,
        f1_pbe0_list,
        f2_pbe0_list,
        extra_feats=extra_feat_list,
        extra_feat_names=extra_feats,
    )
