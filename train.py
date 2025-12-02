import argparse
import os
import random
import sys
import numpy as np
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch import Trainer

from models.mlp import MLP
from models.bias_modulation import DemographicPredictor, BiasModulated
import torch
import pandas as pd

import dataloaders.adult
from dataloaders.datamodule import BaseDataModule

import logging
from utils import save_test_results, seed_everything
import copy


class CustomCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--data_name",
            type=str,
            default="adult",
            help="Name of the dataset to use (e.g., adult, compas)",
        )

        parser.add_argument(
            "--output_dir",
            type=str,
            default="./results",
            help="Directory to save outputs",
        )
        parser.link_arguments(
            "data.input_dim", "model.init_args.input_dim", apply_on="instantiate"
        )

        parser.add_argument(
            "--save_weights",
            type=bool,
            default=False,
            help="Whether to save model weights after training",
        )

    def after_instantiate_classes(self):
        self.datamodule.setup()
        input_dim = (
            self.datamodule.dataset.input_dim
        )  # Set input dim of model based on datamodule
        self.config["fit"]["model"]["init_args"]["input_dim"] = input_dim
        self.config["fit"]["model"]["init_args"][
            "nb_groups"
        ] = self.datamodule.dataset.nb_groups
        self.config["fit"]["model"]["init_args"][
            "nb_group_labels"
        ] = self.datamodule.dataset.nb_group_labels
        self.model = self.model.__class__(
            **self.config["fit"]["model"]["init_args"]
        )  # Re-instantiate model with correct input dim and nb_groups, which depend on the datamodule.
        self.model.group_names = self.datamodule.dataset.group_label_names
        return super().after_instantiate_classes()

    def before_instantiate_classes(self):
        seed_everything(
            self.config["fit"]["seed_everything"]
        )  # Set random seed for reproducibility
        return super().before_instantiate_classes()

    def before_test(self):
        seed_everything(
            self.config["fit"]["seed_everything"]
        )  # Set random seed for reproducibility

    def after_fit(self):
        sensitive_attrib_list = self.config["fit"]["data"]["init_args"][
            "sensitive_attr"
        ]
        out_dir = os.path.join(
            self.config["fit"]["output_dir"], "_".join(sensitive_attrib_list)
        )
        # Create output directory if it doesn't exist
        seed = self.config["fit"]["seed_everything"]
        if self.config["fit"]["save_weights"]:
            model_dir = os.path.join(out_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            # Save the trained model
            # model_path = os.path.join(model_dir, "best.pth")
            # torch.save(self.model.state_dict(), model_path)

            self.trainer.save_checkpoint(os.path.join(model_dir, f"best_{seed}.ckpt"))

        # Recreate the trainer from the existing trainer's arguments, but force devices=1
        self.trainer.devices = 1
        results = self.trainer.test(ckpt_path="best", datamodule=self.datamodule)

        dir_results = os.path.join(out_dir, "results")
        save_test_results(dir_results, results, seed)

    def before_fit(self):
        seed_everything(
            self.config["fit"]["seed_everything"]
        )  # Set random seed for reproducibility

        if isinstance(self.model, BiasModulated):
            # train the demographic predictor first
            logging.info("Training Demographic Predictor before Bias Modulated Model")
            init_args = self.config["fit"]["model"]["init_args"]
            dp_model = DemographicPredictor(
                input_dim=init_args["input_dim"],
                hidden_layers=init_args["hidden_layers"],
                output_dim=init_args["output_dim"],
                dropout=init_args["dropout"],
                learning_rate=init_args["learning_rate"],
                weight_decay=init_args["weight_decay"],
                activation=init_args.get("activation", "relu"),
                nb_groups=init_args["nb_groups"],
                nb_group_labels=init_args["nb_group_labels"],
            )

            dp_trainer = copy.deepcopy(
                self.trainer
            )  # to avoid modifying the main trainer

            out_dir = os.path.join(
                self.config["fit"]["output_dir"],
                "_".join(self.config["fit"]["data"]["init_args"]["sensitive_attr"]),
                "demographic_predictor",
            )

            model_path = os.path.join(
                out_dir,
                "models",
                f"best_{self.config['fit']['seed_everything']}.ckpt",
            )

            if os.path.exists(model_path):
                dp_model.load_state_dict(torch.load(model_path)["state_dict"])
                logging.info(
                    f"Loaded pre-trained Demographic Predictor from {model_path}"
                )
                return
                dp_trainer.devices = 1
                results = dp_trainer.test(
                    ckpt_path=model_path, datamodule=self.datamodule
                )
            else:
                dp_trainer.fit(dp_model, datamodule=self.datamodule)
                dp_trainer.devices = 1
                # load the best checkpoint
                results = dp_trainer.test(ckpt_path="best", datamodule=self.datamodule)

                # save checkppoint and results

                dp_trainer.save_checkpoint(model_path)

            logging.info(f"Demographic Predictor Test Results: {results}")
            dir_results = os.path.join(
                out_dir,
                "results",
            )
            save_test_results(
                dir_results, results, self.config["fit"]["seed_everything"]
            )
            # set the demographic predictor weights to the bias modulated model
            self.model.set_demographic_model_weights(dp_trainer.model.state_dict())

    def after_test(self):
        results = self._results
        self.save_test_results(results)

    def save_test_results_(self, results):
        # Create output directory if it doesn't exist
        result_dir = os.path.join(
            self.config["fit"]["output_dir"],
            self.datamodule.dataset.name,
            # self.config["fit"]["data"]["init_args"]["sensitive_attr_name"],
        )
        os.makedirs(result_dir, exist_ok=True)

        # Save test results
        result_path = os.path.join(
            result_dir, f"model_{self.config['fit']["seed_everything"]}.csv"
        )
        df = pd.DataFrame(results)
        df.to_csv(result_path, index=False)


def cli_main():
    cli = CustomCLI(
        MLP,
        BaseDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    cli_main()
