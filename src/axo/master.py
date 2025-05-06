import sys
import numpy as np
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import mlflow  # Import MLflow

from qkeras.utils import _add_supported_quantized_objects
from qkeras import quantized_bits

co = {}
_add_supported_quantized_objects(co)

import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

K = tf.keras.backend

from . import data_util
from . import losses
from . import models
from . import optim
from . import metric
from . import losses
from . import callbacks as axo_callbacks
from . import utilities

import gc
import re
import pprint
import json

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input
from tensorflow.keras.models import Sequential, Model
import mplhep as hep
from tqdm.auto import tqdm
import argparse
import yaml
import os


def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.add_constructor("tag:yaml.org,2002:python/tuple", tuple_constructor)


def main(slave={}, experiment_name="Experiment"):
    ########################################################################################################
    # MLflow Setup
    ########################################################################################################
    import mlflow  # Ensure MLflow is imported

    mlflow.set_tracking_uri(
        "https://mlflow-deploy-mlflow.app.cern.ch/"
    )  # Set the tracking URI
    mlflow.set_experiment(experiment_name)  # Set the experiment name

    ########################################################################################################
    # Dictionary merging and config checking
    ########################################################################################################
    module_dir = os.path.dirname(__file__)

    master = yaml.load(
        open(os.path.join(module_dir, "utilities/config.yml"), "r"), Loader=yaml.Loader
    )

    if utilities.check_compartibility(master=master, slave=slave) == 1:
        print("Configurations are compatible")
        print("Generating new config ....")
        daughter = utilities.merge_dict(master=master, slave=slave)
    else:
        print("Smoke test failed !!! check the dictionary and the documentations")
        return 0
    config = daughter.copy()  # Setting Daughter as default

    # Extract beta value for MLflow run name
    beta_value = config["train"]["VAE_config"]["beta"]

    # Start the parent MLflow run
    with mlflow.start_run(run_name="training") as parent_run:
        # Start a nested MLflow run identified by the beta value
        with mlflow.start_run(run_name=f"beta_{beta_value}", nested=True) as child_run:
            ########################################################################################################
            # Reproducibility
            ########################################################################################################
            seed = config["determinism"]["global_seed"]
            if config["determinism"]["python_determinism"]:
                os.environ["PYTHONHASHSEED"] = str(seed)
            if config["determinism"]["numpy_determinism"]:
                np.random.seed(seed)
            if config["determinism"]["tf_op_determinism"]:
                tf.random.set_seed(seed)
                tf.config.experimental.enable_op_determinism()
            ########################################################################################################
            # Data creation
            ########################################################################################################
            processed_data_path = config["data_config"]["Processed_data_path"]
            if os.path.isfile(processed_data_path):
                print("File already exists, checking if the config match")
                f = h5py.File(config["data_config"]["Processed_data_path"], "r")
                exisitng_ser_config = f.attrs["config"]
                f.close()
                present_ser_config = json.dumps(config["data_config"])

                # Uncomment the following lines if you want to regenerate data when configs don't match
                if exisitng_ser_config == present_ser_config:
                    print("Configs match, skipping")
                else:
                    print("[WARNING]: CONFIG DO NOT MATCH, OVERWRITING!!!")
                    data_util.data.get_data(config_master=config["data_config"])
            else:
                print("File does not exist, creating data file")
                data_util.data.get_data(config_master=config["data_config"])
            ########################################################################################################
            # Data reading
            ########################################################################################################

            f = h5py.File(config["data_config"]["Processed_data_path"], "r")

            x_train = f["Background_data"]["Train"]["DATA"][:]
            x_test = f["Background_data"]["Test"]["DATA"][:]

            scale = f["Normalisation"]["norm_scale"][:]
            bias = f["Normalisation"]["norm_bias"][:]

            f.close()

            x_train = np.reshape(x_train, (x_train.shape[0], -1))
            x_test = np.reshape(x_test, (x_test.shape[0], -1))
            ########################################################################################################
            # Log Parameters to MLflow
            ########################################################################################################
            mlflow.log_params(
                {
                    "beta": beta_value,
                    "learning_rate": config["train"]["common"]["optimiser_config"].get(
                        "learning_rate"
                    ),
                    "batch_size": config["train"]["common"]["batch_size"],
                    "n_epochs": config["train"]["common"]["n_epochs"],
                    # Add other hyperparameters as needed
                }
            )
            ########################################################################################################
            # Loss Setup
            ########################################################################################################
            loss_name = config["train"]["common"]["reconstruction_loss"].split("_loss")[
                0
            ]  # For users
            constituents = config["data_config"]["Read_configs"]["BACKGROUND"][
                "constituents"
            ]

            compute_loss = getattr(losses, f"{loss_name}_loss")

            loss_reco = compute_loss(
                norm_scales=scale, norm_biases=bias, mask=constituents, name="Reco_loss"
            )
            loss_kld = losses.kld()
            ########################################################################################################
            # Model Setup
            ########################################################################################################
            config_to_vae = {
                **config["model"].copy(),
                **config["train"]["VAE_config"].copy(),
            }
            vae = models.VariationalAutoEncoder(
                config=config_to_vae, reco_loss=loss_reco, kld_loss=loss_kld
            )
            ########################################################################################################
            # Optimiser Setup
            ########################################################################################################
            optim_config = config["train"]["common"]["optimiser_config"]
            optim_name = optim_config["optmiser"]

            try:
                compute_optim = getattr(optim, optim_name)
            except AttributeError:
                print("Optimizer Not found in AXO, looking in TF")
                compute_optim = getattr(tf.keras.optimizers, optim_name)

            allowed_params_for_optim = utilities.allowed_params(compute_optim)
            allowed_keys = list(
                set(optim_config.keys()).intersection(set(allowed_params_for_optim))
            )
            config_to_optim = {
                k: v for k, v in optim_config.items() if k in allowed_keys
            }
            opt = compute_optim(**config_to_optim)
            ########################################################################################################
            # Model Compilation
            ########################################################################################################
            vae.compile(optimizer=opt)
            ########################################################################################################
            # Callback Setup
            ########################################################################################################
            # For now, only dealing with lr_scheduler; more can be added as needed
            callbacks = []
            callback_config = config["callback"]

            # Checking for LR Scheduler
            try:
                lrsc_config = callback_config["lr_schedule"]
            except KeyError:
                print("Learning rate scheduler not found; training with constant loss")
            else:
                lrsc_name = lrsc_config["name"]
                lrsc_config = lrsc_config["config"]
                try:
                    compute_lrsc = getattr(axo_callbacks.lr_scheduler, lrsc_name)
                except AttributeError:
                    print("Scheduler Not found in AXO, looking in TF callbacks")
                    compute_lrsc = getattr(tf.keras.callbacks, lrsc_name)

                # Implementing the lrsc
                allowed_params_for_lrsc = utilities.allowed_params(compute_lrsc)
                allowed_keys = list(
                    set(lrsc_config.keys()).intersection(set(allowed_params_for_lrsc))
                )
                config_to_lrsc = {
                    k: v for k, v in lrsc_config.items() if k in allowed_keys
                }
                lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
                    compute_lrsc(**config_to_lrsc), verbose=1
                )
                callbacks.append(lr_scheduler)

            ########################################################################################################
            # Model Training
            ########################################################################################################
            history = vae.fit(
                x_train,
                x_train,
                callbacks=callbacks,
                batch_size=config["train"]["common"]["batch_size"],
                epochs=config["train"]["common"]["n_epochs"],
                validation_split=0.1,
                shuffle=True,
                verbose=2,
            )
            ########################################################################################################
            # Log Metrics to MLflow
            ########################################################################################################
            # Log the training and validation losses per epoch
            for epoch in range(len(history.history["loss"])):
                mlflow.log_metric(
                    "train_loss", history.history["loss"][epoch], step=epoch
                )
                mlflow.log_metric(
                    "val_loss", history.history["val_loss"][epoch], step=epoch
                )
            ########################################################################################################
            # Plot and Log Training Curves
            ########################################################################################################
            # Plot training & validation loss values
            plt.figure()
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.legend(["Train", "Validation"], loc="upper right")
            # Save plot to a temporary file
            loss_plot_path = "loss_plot.png"
            plt.savefig(loss_plot_path)
            plt.close()
            # Log the plot
            mlflow.log_artifact(loss_plot_path, artifact_path="plots")

            ########################################################################################################
            # Model Trimming
            ########################################################################################################
            model = tf.keras.Sequential(vae.encoder.layers[:-2])
            ########################################################################################################
            # Score plots and result generation
            ########################################################################################################
            config_thres = config["threshold"]
            config_thres["data_path"] = config["data_config"]["Processed_data_path"]
            axo_man = metric.axo_threshold_manager(model=model, config=config_thres)
            dist_plot = metric.distribution_plots(model=model, config=config_thres)

            # Log distribution plots if available
            # Assuming dist_plot generates and saves plots; adjust as necessary
            distribution_plot_path = "distribution_plot.png"
            if os.path.exists(distribution_plot_path):
                mlflow.log_artifact(distribution_plot_path, artifact_path="plots")

            ########################################################################################################
            # Storage and report generation
            ########################################################################################################
            utilities.store_axo(
                config=config,
                model=vae,
                model_trim=model,
                axo_man=axo_man,
                dist_plot=dist_plot,
                history_dict=history.history,
            )

            threshold_dict = utilities.retrieve.get_threshold_dict(
                config["store"]["lite_path"]
            )
            dict_axo = utilities.retrieve.get_axo_score_dataframes(
                config["store"]["lite_path"]
            )
            histogram_dict = utilities.retrieve.get_histogram_dict(
                config["store"]["lite_path"]
            )
            history_dict = utilities.retrieve.get_history_dict(
                config["store"]["lite_path"]
            )

            report_config = config["report"]
            if (
                report_config["html_report"]["generate"]
                or report_config["pdf_report"]["generate"]
            ):
                print("Report generation flag found !!")
                html_path = report_config["html_report"]["path"]
                pdf_path = report_config["pdf_report"]["path"]
                utilities.generate_axolotl_html_report(
                    config=config,
                    dict_axo=dict_axo,
                    histogram_dict=histogram_dict,
                    threshold_dict=threshold_dict,
                    history_dict=history_dict,
                    output_file=html_path,
                    pdf_file=pdf_path,
                )

                # Log the report as an artifact
                mlflow.log_artifact(html_path, artifact_path="Model report")

            print("Run completing exiting ...")
