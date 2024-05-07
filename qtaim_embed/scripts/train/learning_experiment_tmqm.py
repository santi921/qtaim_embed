import json
import torch 
from sklearn.metrics import r2_score
import argparse 
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping
)

from qtaim_embed.models.utils import load_graph_level_model_from_config
from qtaim_embed.core.dataset import HeteroGraphGraphLabelDataset
from qtaim_embed.data.dataloader import DataLoaderMoleculeGraphTask


from qtaim_embed.scripts.train.learning_utils import (
    get_datasets_tmqm_low, 
    get_datasets_tmqm_mid, 
    get_datasets_tmqm_high
)

torch.set_float32_matmul_precision('high')


def manual_statistics(model, batch_graph,batched_labels,scaler_list):
    preds = model.forward(batch_graph, batch_graph.ndata["feat"])

    preds_unscaled = deepcopy(preds.detach())
    labels_unscaled = deepcopy(batched_labels)
    # print("preds unscaled", preds_unscaled)  # * this looks good
    # print("labels unscaled", labels_unscaled)  # * this looks good
    for scaler in scaler_list:
        labels_unscaled = scaler.inverse_feats(labels_unscaled)
        preds_unscaled = scaler.inverse_feats({"global": preds_unscaled})

    preds_unscaled = preds_unscaled["global"].view(-1, model.hparams.ntasks)
    labels_unscaled = labels_unscaled["global"].view(-1, model.hparams.ntasks)
    # manually compute mae and r2
    mae = torch.mean(torch.abs(preds_unscaled - labels_unscaled))
    # convert to numpy 
    preds_unscaled = preds_unscaled.cpu().numpy()
    r2 = r2_score(labels_unscaled, preds_unscaled)
    # convert to single float
    mae = mae.cpu().numpy().tolist()
    return mae, r2



def main(): 

    parser = argparse.ArgumentParser(description='select what level of dataset to run experiment on')
    parser.add_argument('--level', type=str, help='select what level of dataset to run experiment on', required=True)
    args = parser.parse_args()
    level = str(args.level)
    print("level: ", level)

    results_dict = {}


    if level == "low":
        loc_dict = {
            "50": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_low/low_train_50.pkl",
            #"500": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_low/low_train_500.pkl",
            #"5000": 
            #"10000": 
            #"all": 
            #"test": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_low/tmQM_wB97MV_TPSS_QTAIM_corrected_test.pkl"
            "test": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_low/low_train_50.pkl",
        }
        model_dict, dict_keys, dict_datasets = get_datasets_tmqm_low(loc_dict)
    elif level == "mid":
        loc_dict = {
            "50": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_mid/mid_train_50.pkl",
            #"500": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_mid/mid_train_500.pkl",
            #"5000": 
            #"10000": 
            #"all": 
            #"test": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_mid/tmQMg_qtaim_corrected_test.pkl"
            "test": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_mid/mid_train_50.pkl",
        }   
        model_dict, dict_keys, dict_datasets = get_datasets_tmqm_mid(loc_dict)
    elif level == "high":
        loc_dict = {
            "50": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_high/high_train_50.pkl",
            #"500": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_high/high_train_500.pkl", 
            #"5000": 
            #"10000": 
            #"all": 
            #"test": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_high/test_tmQMg_qtaim_best_corrected.pkl"
            "test": "/home/santiagovargas/dev/qtaim_embed/data/tmqm_all/stratified_high/high_train_50.pkl",
        }
        model_dict, dict_keys, dict_datasets = get_datasets_tmqm_high(loc_dict)
    else:
        print("level not found")
        return
    

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

                mean_mae, mean_mse, ewt, _, _ = model_temp.evaluate_manually(
                       batch_graph, batched_labels, test_dataset.label_scalers, per_atom=True
                )
                # convert to numpy
                r2_metrics = list(r2_metrics.cpu().numpy())[0]
                mae_metrics = list(mae_metrics.cpu().numpy())[0]
                mse_metrics = list(mse_metrics.cpu().numpy())[0]
                mean_mae = float(mean_mae.cpu().numpy())
                mean_mse = float(mean_mse.cpu().numpy())
                ewt = float(ewt.cpu().numpy())


                # convert to list 
                r2_metrics = r2_metrics.tolist()
                mae_metrics = mae_metrics.tolist()
                mse_metrics = mse_metrics.tolist()

                mae_man, r2_man = manual_statistics(
                    model_temp, 
                    batch_graph,
                    batched_labels,
                    scaler_list=test_dataset.label_scalers,
                )
                
                results_dict[f"{keys}_{name}"] = {
                    "r2_metrics": r2_metrics,
                    "mae_metrics": mae_metrics,
                    "mse_metrics": mse_metrics,
                    "r2_manual": r2_man, 
                    "mae_manual": mae_man,
                    "mean_mae_per_atom": mean_mae,
                    "mean_rmse_per_atom": mean_mse,
                    "ewt": ewt,
                }


    print(results_dict)
    # save results dict
    json.dump(results_dict, open("./tmqm_{}_learning_results_dict.json".format(level), "w"), indent=4)

main()
    
    