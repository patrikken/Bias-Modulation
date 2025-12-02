import argparse
import os
import random
import sys
import numpy as np
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from lightning.pytorch import Trainer
from models.decoupled import DecoupledModel
from models.mlp import MLP
import logging
from lightning.pytorch import Trainer
from jsonargparse import ArgumentParser, auto_cli
from dataloaders.heart import Heart, BaseDataModule
import torch
import copy
from lightning.pytorch.callbacks import ModelCheckpoint
from utils import save_test_results, seed_everything


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = LightningArgumentParser()
    parser.add_argument("--c", required=True, action="config")
    parser.add_lightning_class_args(MLP, "model", subclass_mode=True)
    parser.add_lightning_class_args(BaseDataModule, "data", subclass_mode=True)
    parser.add_lightning_class_args(Trainer, "trainer", subclass_mode=True)
    parser.add_optimizer_args(torch.optim.Adam, "optimizer.init_args")
    parser.add_argument(
        "--data_name",
        type=str,
        default="adult",
        help="Name of the dataset to use (e.g., adult, compas)",
    )

    parser.add_argument(
        "--seed_everything",
        type=int,
        default=42,
        help="Name of the dataset to use (e.g., adult, compas)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save outputs",
    )

    parser.add_argument(
        "--save_weights",
        type=bool,
        default=False,
        help="Whether to save model weights after training",
    )

    parser.link_arguments(
        "data.input_dim", "model.init_args.input_dim", apply_on="instantiate"
    )
    cfg = parser.parse_args()

    init = parser.instantiate_classes(cfg, instantiate_groups=True)

    seed = cfg["seed_everything"]
    seed_everything(seed)

    model = init["model"]
    datamodule = init["data"]
    trainer = init["trainer"]

    datamodule.setup(stage="fit")
    input_dim = datamodule.dataset.input_dim
    nb_groups = datamodule.dataset.nb_groups  #
    cfg["model"]["init_args"]["input_dim"] = input_dim
    cfg["model"]["init_args"][
        "nb_groups"
    ] = (
        datamodule.dataset.nb_groups
    )  # for defining number of heads in multitask setting or number of models in decoupled setting
    cfg["model"]["init_args"][
        "nb_group_labels"
    ] = datamodule.dataset.nb_group_labels  # for reporting per group metrics
    model = model.__class__(
        **cfg["model"]["init_args"],
    )
    model.group_names = datamodule.dataset.group_label_names
    group_idd = np.unique(datamodule.dataset.group_idx)

    sensitive_attrib_list = cfg["data"]["init_args"]["sensitive_attr"]
    out_dir = os.path.join(cfg["output_dir"], "_".join(sensitive_attrib_list))
    dir_results = os.path.join(out_dir, "results")
    dir_models = os.path.join(out_dir, "models")

    for g in group_idd:
        logging.info(
            f"Training for Group {g} has {sum(datamodule.dataset.group_idx == g)} samples."
        )
        group_loader = datamodule.set_current_group(group_id=g)
        model.set_current_group(g)
        current_trainer = copy.deepcopy(trainer)
        checkpoint = current_trainer.callbacks[0]
        current_trainer.fit(model.models_per_group[g], datamodule=datamodule)
        if cfg["save_weights"]:
            current_trainer.save_checkpoint(
                os.path.join(dir_models, f"best_group_{g}_seed_{seed}.ckpt")
            )
    # datamodule = parser.instantiate_classes(cfg.data)

    test_results = trainer.test(
        model, datamodule=datamodule, ckpt_path=cfg.trainer.get("ckpt_path", None)
    )
    logging.info(f"Test Results: {test_results}")

    save_test_results(dir_results, test_results, seed)
    # cli_main()
