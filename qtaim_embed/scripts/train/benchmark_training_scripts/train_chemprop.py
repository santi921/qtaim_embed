import chemprop
import pandas as pd


# hyper params
arguments = [
    "--epochs",
    "100",
    "--batch_size",
    "512",
    "--num_iters",
    "30",
    "--gpu",
    "5",
    "--log_dir",
    "./logs/",
    "--explicit_h",
    "--target_columns",
    "homo lumo gap U0",
    "--data_path",
    "../../barriers_data/rapter_train.csv",  # update
    "--dataset_type",
    "regression",
    "--config_save_path",
    "./bayes_qm8/best.json",
]

args = chemprop.args.HyperoptArgs().parse_args(arguments)
chemprop.hyperparameter_optimization.hyperopt(args)


arguments = [
    "--gpu",
    "5",
    "--data_path",
    "../../barriers_data/rapter_train.csv",  # update
    "--target_columns",
    "homo lumo gap U0",
    "--dataset_type",
    "regression",
    "--config_path",
    "./bayes_qm8/best.json" "--save_dir",
    "test_best_qm8",
]


args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(
    args=args, train_func=chemprop.train.run_training
)


# test
arguments = [
    "--test_path",
    "../../barriers_data/rapter_test.csv",  # update
    "--target_columns",
    "homo lumo gap U0",
    "--preds_path",
    "test_preds_qm8.csv",
    "--checkpoint_dir",
    "test_best_qm8",
]


args = chemprop.args.PredictArgs().parse_args(arguments)
preds = chemprop.train.make_predictions(args=args)
