from lightning.pytorch import Trainer
from models.mlp import MLP
from models.bias_modulation import BiasModulated
from dataloaders.datamodule import BaseDataModule
from models.decoupled import DecoupledModel
from models.generic import GenericModel
from dataloaders.heart import Heart
from dataloaders.mimic import Mimic
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from utils import save_test_results, seed_everything
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import copy
import torch
import os
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping


def train_model(model_instance, trainer, datamodule):
    model_instance.group_names = datamodule.dataset.group_label_names
    # train_model(cfg, model_instance, datamodule, trainer)
    if isinstance(model_instance, BiasModulated):
        logging.info("Training Demographic Predictor first...")
        demog_pred = model_instance.demographic_model
        demog_trainer = Trainer(
            max_epochs=100,
            callbacks=[copy.deepcopy(cb) for cb in trainer.callbacks],
            # logger=trainer.logger,
            enable_progress_bar=True,
            enable_model_summary=False,
            devices=1,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            log_every_n_steps=5,
        )
        demog_trainer.fit(model_instance.demographic_model, datamodule=datamodule)
        logging.info("Demographic Predictor training completed.")
        model_instance.demographic_model = demog_pred
    elif isinstance(model_instance, DecoupledModel):
        logging.info("Training Decoupled Model...")

        group_labels = datamodule.dataset.nb_group_labels
        val_loss = 0.0
        for g in range(group_labels):
            logging.info(f"Training model for group {g}...")
            model_instance.set_current_group(g)
            group_trainer = Trainer(
                max_epochs=100,
                callbacks=[
                    copy.deepcopy(cb) for cb in trainer.callbacks
                ],  # ensure each group has its own callbacks
                # logger=trainer.logger,
                enable_progress_bar=True,
                enable_model_summary=False,
                devices=1,
                accelerator="cuda" if torch.cuda.is_available() else "cpu",
                log_every_n_steps=5,
            )
            group_trainer.fit(model_instance, datamodule=datamodule)
            logging.info(f"Model training for group {g} completed.")

        test_results = trainer.test(model_instance, datamodule=datamodule)
        logging.info(f"Test Results: {test_results}")
        return test_results

    # add model checkpoint callback to save best model based on worst-group validation accuracy also EarlyStopping callback based on worst-group validation accuracy
    
    if not isinstance(model_instance, DecoupledModel) and not isinstance(model_instance, GenericModel):
        trainer.callbacks = [
            cb
            for cb in trainer.callbacks
            if not isinstance(cb, ModelCheckpoint) and not isinstance(cb, EarlyStopping)
        ]
        EarlyStopping_callback = EarlyStopping(
            monitor="val_worst_group_loss",
            patience=10,
            mode="min",
            verbose=True,
            min_delta=0.001,
        )
        trainer.callbacks.append(EarlyStopping_callback)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_worst_group_loss",
            mode="min",
            filename="best_{epoch}_{val_worst_group_loss:.4f}",
            save_top_k=1,
        )
        trainer.callbacks.append(checkpoint_callback)
    trainer.fit(model_instance, datamodule=datamodule)
    test_results = trainer.test(model_instance, datamodule=datamodule, ckpt_path="best")
    logging.info(f"Test Results: {test_results}")

    return test_results


if __name__ == "__main__":
    import logging

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
        "--method_name",
        type=str,
        default="generic",
        help="Name of the method to use (e.g., 1hot, bias_modulation)",
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
        "--dataset_name",
        help="Name of the dataset to use",
        choices=["heart", "mimic", "sap"],
    )

    """ parser.add_argument(
        "--trail",
        type=int,
        default=100,
        help="Number of hyperparameter optimization trials",
    ) """

    cfg = parser.parse_args()
    seed = cfg["seed_everything"]

    if cfg["dataset_name"] == "heart":
        cfg["data"]["class_path"] = "dataloaders.heart.Heart"
    elif cfg["dataset_name"] == "mimic":
        cfg["data"]["class_path"] = "dataloaders.mimic.Mimic"
    elif cfg["dataset_name"] == "sap":
        cfg["data"]["class_path"] = "dataloaders.sap.Sap"
    else:
        raise ValueError(f"Unsupported dataset: {cfg['dataset_name']}")

    if cfg["method_name"] == "generic":
        cfg["model"]["class_path"] = "models.generic.GenericModel"
    elif cfg["method_name"] == "decoupled":
        cfg["model"]["class_path"] = "models.decoupled.DecoupledModel"
    elif cfg["method_name"] == "bias_modulation":
        cfg["model"]["class_path"] = "models.bias_modulation.BiasModulated"
    elif cfg["method_name"] == "1hot":
        cfg["model"]["class_path"] = "models.generic.GenericModel"
        cfg["data"]["init_args"]["is_1hot"] = True
    elif cfg["method_name"] == "intersectional":
        cfg["model"]["class_path"] = "models.generic.GenericModel"
        cfg["data"]["init_args"]["is_intersectional"] = True
    elif cfg["method_name"] == "multitask":
        cfg["model"]["class_path"] = "models.multitask.MultiTaskModel"
    else:
        raise ValueError(f"Unsupported method: {cfg['method_name']}")

    seed_everything(seed)
    init = parser.instantiate_classes(cfg, instantiate_groups=True)

    model = init["model"]
    datamodule = init["data"]
    trainer = init["trainer"]

    datamodule.setup(stage="fit")
    input_dim = datamodule.dataset.input_dim
    nb_groups = datamodule.dataset.nb_groups  #
    sensitive_attr = "_".join(datamodule.dataset.sensitive_attr)
    best_hyperparam_path = f"configs/best_hyperparameters/{cfg['dataset_name']}/{cfg['method_name']}/{sensitive_attr}/best_hyperparameters.yaml"

    if os.path.exists(best_hyperparam_path):
        logging.info(f"Loading best hyperparameters from {best_hyperparam_path}")
        import yaml

        with open(best_hyperparam_path, "r") as f:
            for key, value in yaml.safe_load(f).items():
                # if key in cfg["model"]["init_args"]:
                logging.info(f"Best hyperparameter - {key}: {value}")
                cfg["model"]["init_args"][key] = value
    else:
        logging.info(
            f"No best hyperparameters found at {best_hyperparam_path}. Using default parameters."
        )
    cfg["model"]["init_args"]["input_dim"] = input_dim
    cfg["model"]["init_args"][
        "nb_groups"
    ] = (
        datamodule.dataset.nb_groups
    )  # for defining number of heads in multitask setting or number of models in decoupled setting
    cfg["model"]["init_args"]["nb_group_labels"] = datamodule.dataset.nb_group_labels

    model_instance = model.__class__(**cfg["model"]["init_args"])
    test_results = train_model(model_instance, trainer, datamodule)

    out_dir = os.path.join("results", cfg["dataset_name"], cfg["method_name"])
    dir_results = os.path.join(out_dir, sensitive_attr)
    save_test_results(dir_results, test_results, seed=seed)
