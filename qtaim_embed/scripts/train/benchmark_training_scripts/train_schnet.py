import argparse
import pandas as pd
from ase import Atoms
import numpy as np
import os

import torch, torchmetrics
import pytorch_lightning as pl

from schnetpack.transform import ASENeighborList
from schnetpack.data import ASEAtomsData
import schnetpack as spk
import schnetpack.transform as trn


def convert_to_ase_props_qm8(file_path):
    """
    Converts the npz file to a list of ASE Atoms objects and a list of dicts of properties
    """

    data = np.load(file_path)
    atoms_list = []
    property_list = []
    numbers = data["N"]
    R = data["R"]
    Z = data["Z"]
    atom_count = 0
    for mol_ind, mol in enumerate(numbers):
        ats = Atoms(
            positions=R[atom_count : atom_count + mol],
            numbers=Z[atom_count : atom_count + mol],
        )
        atoms_list.append(ats)
        atom_count += mol
        properties = {
            "E1_CC2": [data["E1_CC2"][mol_ind]],
            "E2_CC2": [data["E2_CC2"][mol_ind]],
        }
        property_list.append(properties)
    return atoms_list, property_list


def convert_to_ase_props_qm9(file_path):
    """
    Converts the npz file to a list of ASE Atoms objects and a list of dicts of properties
    """

    data = np.load(file_path)
    atoms_list = []
    property_list = []
    numbers = data["N"]
    R = data["R"]
    Z = data["Z"]
    atom_count = 0
    for mol_ind, mol in enumerate(numbers):
        ats = Atoms(
            positions=R[atom_count : atom_count + mol],
            numbers=Z[atom_count : atom_count + mol],
        )
        atoms_list.append(ats)
        atom_count += mol
        properties = {
            "U0": [data["U0"][mol_ind]],
            "homo": [data["homo"][mol_ind]],
            "lumo": [data["lumo"][mol_ind]],
            "gap": [data["gap"][mol_ind]],
        }
        property_list.append(properties)
    return atoms_list, property_list


def main():

    # create argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-npz_in", default="./qm8_train_dimenet.npz")
    parser.add_argument("-npz_in_test", default="./qm8_test_dimenet.npz")
    parser.add_argument("-processed_file_name", default="./qm8_train_schnet.db")
    parser.add_argument("-processed_file_name_test", default="./qm8_test_schnet.db")
    parser.add_argument("-dataset", default="qm8")
    parser.add_argument("-epochs", default=3)

    args = parser.parse_args()
    npz_in = str(args.npz_in)
    npz_in_test = str(args.npz_in_test)
    processed_file_name = str(args.processed_file_name)
    processed_file_name_test = str(args.processed_file_name_test)
    epochs = int(args.epochs)
    dataset = str(args.dataset)

    if dataset == "qm8":
        atoms_list, property_list = convert_to_ase_props_qm8(npz_in)
        atoms_list_test, property_list_test = convert_to_ase_props_qm8(npz_in_test)
        prop_dict = {
            "E1_CC2": "Hartree",
            "E2_CC2": "Hartree",
        }
        split_file_name = "./qm8_split.npz"
        split_file_name_test = "./qm8_split_test.npz"
        model_save_dir = "./qm8/"

    else:
        atoms_list, property_list = convert_to_ase_props_qm9(npz_in)
        atoms_list_test, property_list_test = convert_to_ase_props_qm9(npz_in_test)
        prop_dict = {
            "homo": "Hartree",
            "lumo": "Hartree",
            "gap": "Hartree",
            "U0": "Hartree",
        }
        split_file_name = "./qm9_split.npz"
        split_file_name_test = "./qm9_split_test.npz"
        model_save_dir = "./qm9/"

    dataset_train = ASEAtomsData.create(
        processed_file_name,
        distance_unit="Ang",
        property_unit_dict=prop_dict,
    )

    dataset_test = ASEAtomsData.create(
        processed_file_name_test,
        distance_unit="Ang",
        property_unit_dict=prop_dict,
    )
    dataset_train.add_systems(property_list, atoms_list)
    dataset_test.add_systems(property_list, atoms_list)

    # len_dataset = len(dataset_train)
    # len_dataset_test = len(dataset_test)
    # len_train = int(0.8 * len_dataset)
    # len_val = int(0.2 * len_dataset)

    datamodule_train = spk.data.AtomsDataModule(
        processed_file_name,
        batch_size=512,
        distance_unit="Ang",
        property_units=prop_dict,
        num_train=0.8,
        num_val=0.2,
        transforms=[
            trn.ASENeighborList(cutoff=5.0),
            trn.CastTo32(),
        ],
        num_workers=4,
        split_file=split_file_name,
        pin_memory=True,  # set to false, when not using a GPU
    )

    datamodule_test = spk.data.AtomsDataModule(
        processed_file_name_test,
        batch_size=512,
        distance_unit="Ang",
        property_units=prop_dict,
        num_train=0.01,
        num_val=0.99,
        transforms=[
            trn.ASENeighborList(cutoff=5.0),
            trn.CastTo32(),
        ],
        num_workers=4,
        split_file=split_file_name_test,
        pin_memory=True,  # set to false, when not using a GPU
    )

    datamodule_train.prepare_data()
    datamodule_train.setup()
    datamodule_test.prepare_data()
    datamodule_test.setup()

    cutoff = 5.0
    n_atom_basis = 30
    pairwise_distance = (
        spk.atomistic.PairwiseDistances()
    )  # calculates pairwise distances between atoms
    radial_basis = spk.nn.GaussianRBF(n_rbf=30, cutoff=cutoff)
    schnet = spk.representation.SchNet(
        n_atom_basis=n_atom_basis,
        n_interactions=5,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )

    if dataset == "qm8":
        pred_e1 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="E1_CC2")
        pred_e2 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="E2_CC2")
        output_e1 = spk.task.ModelOutput(
            name="E1_CC2",
            loss_fn=torch.nn.MSELoss(),
            loss_weight=1.0,
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError(),
                "r2": torchmetrics.R2Score(),
            },
        )

        output_e2 = spk.task.ModelOutput(
            name="E2_CC2",
            loss_fn=torch.nn.MSELoss(),
            loss_weight=1.0,
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError(),
                "r2": torchmetrics.R2Score(),
            },
        )
        output_list = [pred_e1, pred_e2]
        model_outputs = [output_e1, output_e2]

    else:
        pred_homo = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="homo")
        pred_lumo = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="lumo")
        pred_gap = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="gap")
        pred_u0 = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key="U0")
        output_homo = spk.task.ModelOutput(
            name="homo",
            loss_fn=torch.nn.MSELoss(),
            loss_weight=1.0,
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError(),
                "r2": torchmetrics.R2Score(),
            },
        )
        output_lumo = spk.task.ModelOutput(
            name="lumo",
            loss_fn=torch.nn.MSELoss(),
            loss_weight=1.0,
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError(),
                "r2": torchmetrics.R2Score(),
            },
        )
        output_gap = spk.task.ModelOutput(
            name="gap",
            loss_fn=torch.nn.MSELoss(),
            loss_weight=1.0,
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError(),
                "r2": torchmetrics.R2Score(),
            },
        )
        output_u0 = spk.task.ModelOutput(
            name="U0",
            loss_fn=torch.nn.MSELoss(),
            loss_weight=1.0,
            metrics={
                "MAE": torchmetrics.MeanAbsoluteError(),
                "r2": torchmetrics.R2Score(),
            },
        )
        output_list = [pred_homo, pred_lumo, pred_gap, pred_u0]
        model_outputs = [output_homo, output_lumo, output_gap, output_u0]

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=output_list,
        postprocessors=[
            trn.CastTo64(),
        ],
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=model_outputs,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1e-4},
    )

    logger = pl.loggers.TensorBoardLogger(save_dir=model_save_dir)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(model_save_dir, "best_inference_model"),
            save_top_k=1,
            monitor="val_loss",
        )
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=model_save_dir,
        max_epochs=epochs,  # for testing, we restrict the number of epochs
    )
    trainer.fit(task, datamodule=datamodule_train)
    trainer.validate(task, datamodule=datamodule_test)


main()
