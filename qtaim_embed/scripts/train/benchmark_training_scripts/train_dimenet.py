import tensorflow as tf
import numpy as np
import os
import ast
import logging
import string
import random
import yaml
from datetime import datetime

from dimenet.model.dimenet import DimeNet
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
from dimenet.training.trainer import Trainer
from dimenet.training.metrics import Metrics
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider


def id_generator(
    size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits
):
    return "".join(random.SystemRandom().choice(chars) for _ in range(size))


def main():
    # Set up logger
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s (%(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel("INFO")

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    tf.get_logger().setLevel("WARN")
    tf.autograph.set_verbosity(2)

    config = {
        "model_name": "dimenet++",
        "emb_size": 128,
        "out_emb_size": 256,
        "int_emb_size": 64,
        "basis_emb_size": 8,
        "num_blocks": 4,
        "num_spherical": 7,
        "num_radial": 6,
        "output_init": "GlorotOrthogonal",
        "extensive": False,
        "cutoff": 5.0,
        "envelope_exponent": 5,
        "num_before_skip": 1,
        "num_after_skip": 2,
        "num_dense_output": 3,
        "num_train": 108463,  # adjust
        "num_valid": 12000,  # adjust
        "num_test": 10000,  # adjust
        "data_seed": 42,
        "dataset": "./benchmark_data/qm9_train_dimenet.npz",
        "dataset_test": "./benchmark_data/qm9_test_dimenet.npz",
        "logdir": "logs",
        "num_steps": 3000000,
        "ema_decay": 0.999,
        "learning_rate": 0.001,
        "warmup_steps": 3000,
        "decay_rate": 0.01,
        "decay_steps": 4000000,
        "batch_size": 32,
        "evaluation_interval": 100,
        "save_interval": 100,
        "restart": None,
        "comment": "final",
        "targets": ["homo", "lumo", "gap", "U0"],  # edit for multitask
    }

    # For strings that yaml doesn't parse (e.g. None)
    for key, val in config.items():
        if type(val) is str:
            try:
                config[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass

    model_name = config["model_name"]

    if model_name == "dimenet":
        num_bilinear = config["num_bilinear"]
    elif model_name == "dimenet++":
        out_emb_size = config["out_emb_size"]
        int_emb_size = config["int_emb_size"]
        basis_emb_size = config["basis_emb_size"]
        extensive = config["extensive"]
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")

    emb_size = config["emb_size"]
    num_blocks = config["num_blocks"]

    num_spherical = config["num_spherical"]
    num_radial = config["num_radial"]
    output_init = config["output_init"]

    cutoff = config["cutoff"]
    envelope_exponent = config["envelope_exponent"]

    num_before_skip = config["num_before_skip"]
    num_after_skip = config["num_after_skip"]
    num_dense_output = config["num_dense_output"]

    num_train = config["num_train"]
    num_valid = config["num_valid"]
    num_test = config["num_test"]
    data_seed = config["data_seed"]
    dataset = config["dataset"]
    dataset_test = config["dataset_test"]
    logdir = config["logdir"]

    num_steps = config["num_steps"]
    ema_decay = config["ema_decay"]

    learning_rate = config["learning_rate"]
    warmup_steps = config["warmup_steps"]
    decay_rate = config["decay_rate"]
    decay_steps = config["decay_steps"]

    batch_size = config["batch_size"]
    evaluation_interval = config["evaluation_interval"]
    save_interval = config["save_interval"]
    restart = config["restart"]
    comment = config["comment"]
    targets = config["targets"]

    # Create directories
    # A unique directory name is created for this run based on the input
    if restart is None:
        directory = (
            logdir
            + "/"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + id_generator()
            + "_"
            + os.path.basename(dataset)
            + "_"
            + "-".join(targets)
            + "_"
            + comment
        )
    else:
        directory = restart
    logging.info(f"Directory: {directory}")

    if not os.path.exists(directory):
        os.makedirs(directory)
    best_dir = os.path.join(directory, "best")
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    log_dir = os.path.join(directory, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    best_loss_file = os.path.join(best_dir, "best_loss.npz")
    best_ckpt_file = os.path.join(best_dir, "ckpt")
    step_ckpt_folder = log_dir

    summary_writer = tf.summary.create_file_writer(log_dir)

    test = {}
    train = {}
    validation = {}

    train["metrics"] = Metrics("train", targets)
    validation["metrics"] = Metrics("val", targets)
    test["metrics"] = Metrics("test", targets)

    data_container = DataContainer(dataset, cutoff=cutoff, target_keys=targets)
    data_container_test = DataContainer(
        dataset_test, cutoff=cutoff, target_keys=targets
    )

    # Initialize DataProvider (splits dataset into 3 sets based on data_seed and provides tf.datasets)
    data_provider = DataProvider(
        data_container,
        num_train,
        num_valid,
        batch_size,
        seed=data_seed,
        randomized=True,
    )

    data_provider_test = DataProvider(
        data_container_test,
        0,
        len(data_container_test),
        batch_size,
        seed=data_seed,
        randomized=True,
    )

    # Initialize datasets
    train["dataset"] = data_provider.get_dataset("train").prefetch(
        tf.data.experimental.AUTOTUNE
    )
    validation["dataset"] = data_provider.get_dataset("val").prefetch(
        tf.data.experimental.AUTOTUNE
    )
    test["dataset"] = data_provider_test.get_dataset("val").prefetch(
        tf.data.experimental.AUTOTUNE
    )

    train["dataset_iter"] = iter(train["dataset"])
    validation["dataset_iter"] = iter(validation["dataset"])
    test["dataset_iter"] = iter(test["dataset"])

    if model_name == "dimenet":
        model = DimeNet(
            emb_size=emb_size,
            num_blocks=num_blocks,
            num_bilinear=num_bilinear,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_dense_output=num_dense_output,
            num_targets=len(targets),
            activation=swish,
            output_init=output_init,
        )
    elif model_name == "dimenet++":
        model = DimeNetPP(
            emb_size=emb_size,
            out_emb_size=out_emb_size,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            num_blocks=num_blocks,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_dense_output=num_dense_output,
            num_targets=len(targets),
            activation=swish,
            extensive=extensive,
            output_init=output_init,
        )
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")

    if os.path.isfile(best_loss_file):
        loss_file = np.load(best_loss_file)
        metrics_best = {k: v.item() for k, v in loss_file.items()}
    else:
        metrics_best = validation["metrics"].result()
        for key in metrics_best.keys():
            metrics_best[key] = np.inf
        metrics_best["step"] = 0
        np.savez(best_loss_file, **metrics_best)

    trainer = Trainer(
        model,
        learning_rate,
        warmup_steps,
        decay_steps,
        decay_rate,
        ema_decay=ema_decay,
        max_grad_norm=1000,
    )
    # Set up checkpointing
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), optimizer=trainer.optimizer, model=model
    )
    manager = tf.train.CheckpointManager(ckpt, step_ckpt_folder, max_to_keep=3)

    # Restore latest checkpoint
    ckpt_restored = tf.train.latest_checkpoint(log_dir)
    if ckpt_restored is not None:
        ckpt.restore(ckpt_restored)

    with summary_writer.as_default():
        steps_per_epoch = int(np.ceil(num_train / batch_size))

        if ckpt_restored is not None:
            step_init = ckpt.step.numpy()
        else:
            step_init = 1
        for step in range(step_init, num_steps + 1):
            # Update step number
            ckpt.step.assign(step)
            tf.summary.experimental.set_step(step)

            # Perform training step
            trainer.train_on_batch(train["dataset_iter"], train["metrics"])

            # Save progress
            if step % save_interval == 0:
                manager.save()

            # Evaluate model and log results
            if step % evaluation_interval == 0:
                print("eval step", step)
                # Save backup variables and load averaged variables
                trainer.save_variable_backups()
                # trainer.load_averaged_variables()

                # Compute results on the validation set
                for i in range(int(np.ceil(num_valid / batch_size))):
                    trainer.test_on_batch(
                        validation["dataset_iter"], validation["metrics"]
                    )

                # for i in range(int(np.ceil(num_valid / batch_size))):
                #    trainer.test_on_batch(test["dataset_iter"], test["metrics"])
                # Update and save best result
                if validation["metrics"].mean_mae < metrics_best["mean_mae_val"]:
                    metrics_best["step"] = step
                    metrics_best.update(validation["metrics"].result())

                    np.savez(best_loss_file, **metrics_best)
                    model.save_weights(best_ckpt_file)

                for key, val in metrics_best.items():
                    if key != "step":
                        tf.summary.scalar(key + "_best", val)

                epoch = step // steps_per_epoch
                logging.info(
                    f"{step}/{num_steps} (epoch {epoch+1}): "
                    f"Loss: train={train['metrics'].loss:.6f}, val={validation['metrics'].loss:.6f}; "
                    # f"r2: train={train['metrics'].r2:.6f}, val={validation['metrics'].r2:.6f}; "
                    f"r2: train={train['metrics'].mean_r2:.6f}, val={validation['metrics'].mean_r2:.6f}; "
                    f"mae: train={train['metrics'].mean_mae:.6f}, val={validation['metrics'].mean_mae:.6f}; "
                    f"val={validation['metrics'].mean_log_mae:.6f}, "
                )

                train["metrics"].write()
                validation["metrics"].write()

                train["metrics"].reset_states()
                validation["metrics"].reset_states()

                # Restore backup variables
                trainer.restore_variable_backups()

        # Compute results on the validation set
        for i in range(int(np.ceil(num_test / batch_size))):
            trainer.test_on_batch(test["dataset_iter"], test["metrics"])

        epoch = step // steps_per_epoch
        logging.info(
            f"{step}/{num_steps} (epoch {epoch+1}): "
            f"Loss: test={test['metrics'].loss:.6f}; "
            f"r2: test={test['metrics'].mean_r2:.6f}; "
            f"MAE: test={test['metrics'].mean_mae:.6f}, "
        )
        print(
            f"{step}/{num_steps} (epoch {epoch+1}): "
            f"Loss: test={test['metrics'].loss:.6f}; "
            f"r2: test={test['metrics'].mean_r2:.6f}; "
            f"MAE: test={test['metrics'].mean_mae:.6f}, "
        )
        print(test["metrics"].maes)
        test["metrics"].write()
        test["metrics"].reset_states()


main()
