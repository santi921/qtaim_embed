import json
import torch 
from sklearn.metrics import r2_score
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping
)

from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.core.dataset import HeteroGraphGraphLabelDataset
from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask
from qtaim_embed.scripts.train.learning_utils import get_datasets_qm8

torch.set_float32_matmul_precision('high')


# main 
def main(): 
    results_dict = {}
    loc_dict = {
        "10": "../../../data/splits_1205/train_qm8_qtaim_1205_labelled_10.pkl",
        "100": "../../../data/splits_1205/train_qm8_qtaim_1205_labelled_100.pkl",
        "1000": "../../../data/splits_1205/train_qm8_qtaim_1205_labelled_1000.pkl",
        "10000": "../../../data/splits_1205/train_qm8_qtaim_1205_labelled_10000.pkl",
        "all": "../../../data/splits_1205/train_qm8_qtaim_1205_labelled.pkl",
        "test": "../../../data/splits_1205/test_qm8_qtaim_1205_labelled.pkl"
    }
    model_dict, dict_keys, dict_datasets = get_datasets_qm8(loc_dict)


    for keys in dict_datasets.keys():

        model_temp = load_graph_level_model_from_config(model_dict[keys])
        test_dataset = dict_datasets[keys]["test"]
        
        for name in dict_datasets[keys].keys():
            if name != "test":
                dataloader_train = DataLoaderMoleculeGraphTask(dict_datasets[keys][name], batch_size=256, shuffle=True, num_workers=0)
                dataloader_test = DataLoaderMoleculeGraphTask(test_dataset, batch_size=len(test_dataset.graphs), shuffle=False, num_workers=0)
                early_stopping_callback = EarlyStopping(
                    monitor="val_mae", min_delta=0.00, patience=100, verbose=False, mode="min"
                )
                lr_monitor = LearningRateMonitor(logging_interval="step")

                trainer = pl.Trainer(
                    max_epochs=1000,
                    accelerator="gpu",
                    gradient_clip_val=2.0,
                    devices=1,
                    accumulate_grad_batches=1,
                    enable_progress_bar=True,
                    callbacks=[
                        early_stopping_callback,
                        lr_monitor,
                    ],
                    enable_checkpointing=True,
                    strategy="auto",
                    #default_root_dir=model_save_string,
                    default_root_dir="./test/",
                    precision="bf16-mixed",
                )
                
                trainer.fit(model_temp, dataloader_train)
                trainer.save_checkpoint(f"./libe_learning_test/{keys}_{name}.ckpt")
                
                batch_graph, batched_labels = next(iter(dataloader_test))
                r2_metrics, mae_metrics, mse_metrics, _, _ = model_temp.evaluate_manually(
                    batch_graph,
                    batched_labels,
                    scaler_list=test_dataset.label_scalers,
                )
                # convert to numpy
                r2_metrics = r2_metrics.cpu().numpy()
                mae_metrics = mae_metrics.cpu().numpy()
                mse_metrics = mse_metrics.cpu().numpy()

                # convert to list 
                r2_metrics = r2_metrics.tolist()
                mae_metrics = mae_metrics.tolist()
                mse_metrics = mse_metrics.tolist()
                
                results_dict[f"{keys}_{name}"] = {
                    "r2_metrics": r2_metrics,
                    "mae_metrics": mae_metrics,
                    "mse_metrics": mse_metrics,
                }


    print(results_dict)
    # save results dict
    json.dump(results_dict, open("./qm8_learning_results_dict.json", "w"), indent=4)

main()
