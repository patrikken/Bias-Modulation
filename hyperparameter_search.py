from lightning.pytorch import Trainer
from models.mlp import MLP
from models.bias_modulation import BiasModulated
from dataloaders.datamodule import BaseDataModule
from models.decoupled import DecoupledModel
from dataloaders.heart import Heart
from dataloaders.mimic import Mimic
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from utils import save_test_results, seed_everything
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import copy
import torch
import os
from lightning.pytorch.callbacks import RichProgressBar


def train_model(cfg, model, datamodule, trainer):
    datamodule.setup(stage="fit")
    callbacks = [copy.deepcopy(c) for c in trainer.callbacks if not isinstance(c, RichProgressBar) and not isinstance(c, PyTorchLightningPruningCallback)]
    if isinstance(model, BiasModulated):
        # train demographic predictor first
        logging.info("Training Demographic Predictor first...")
        demog_pred = model.demographic_model
        demog_trainer = Trainer(
            max_epochs=100,
            callbacks=callbacks,
            #logger=trainer.logger,
            enable_progress_bar=False,
            enable_model_summary=False,
            devices=cfg["trainer"]['init_args']["devices"],
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            log_every_n_steps=5
        )
        demog_trainer.fit(demog_pred, datamodule=datamodule) 
        logging.info("Demographic Predictor training completed.")
        model.set_demographic_model_weights(demog_pred.state_dict())
    elif isinstance(model, DecoupledModel):
        logging.info("Training Decoupled Model...")

        group_labels = datamodule.dataset.nb_group_labels
        val_loss = 0.0
        for g in range(group_labels):
            logging.info(f"Training model for group {g}...")
            model.set_current_group(g)
            group_trainer = Trainer(
                max_epochs=100,
                callbacks=callbacks,
                #logger=trainer.logger,
                enable_progress_bar=False,
                enable_model_summary=False,
                devices=cfg["trainer"]['init_args']["devices"],
                accelerator="cuda" if torch.cuda.is_available() else "cpu",
                log_every_n_steps=5
            )
            group_trainer.fit(model, datamodule=datamodule)
            logging.info(f"Model training for group {g} completed.")
            val_loss += group_trainer.callback_metrics["val_loss"].item()
        avg_val_loss = val_loss / group_labels
        logging.info(f"Average validation loss across groups: {avg_val_loss}")
        return avg_val_loss

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_loss"].item()


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
        "--dataset_name", help="Name of the dataset to use", choices=["heart", "mimic"]
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
    #cfg["trainer"]['init_args']["strategy"] = "ddp"  # disable ddp for hyperparameter search
    seed_everything(seed)
    init = parser.instantiate_classes(cfg, instantiate_groups=True)

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
    cfg["model"]["init_args"]["nb_group_labels"] = datamodule.dataset.nb_group_labels

    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2)
        hidden_layers = trial.suggest_categorical(
            "hidden_layers", [[32, 32], [64, 32], [128, 64]]
        )
        dropout = trial.suggest_float("dropout", 0.2, 0.5)

        if cfg["method_name"] == "bias_modulation":
            # For bias modulation, we might want to keep a smaller range for weight decay
            head_weights = trial.suggest_categorical(
                "head_weights", [[0.5, 0.5], [0.7, 0.3], [0.3, 0.7]]
            )
            """ hidden_layers_demographic = trial.suggest_categorical(
                "hidden_layers_demographic", [[16, 16], [32, 16], [64, 32]]
            ) 
            cfg["model"]["init_args"][
                "hidden_layers_demographic"
            ] = hidden_layers_demographic"""

            """ if len(hidden_layers) <= 2:
                modulated_layers_indices = trial.suggest_categorical(
                    "modulated_layers_indices", [[0], [1], [0, 1]]
                )
            else: """
           
            """modulated_at_inference = trial.suggest_categorical(
                "modulated_at_inference", [True, False]
            )
            cfg["model"]["init_args"]["modulated_at_inference"] = modulated_at_inference
             remove_bias_in_linear_layers = trial.suggest_categorical(
                "remove_bias_in_linear_layers", [True, False]
            )  
            cfg["model"]["init_args"][
                "remove_bias_in_linear_layers"
            ] = remove_bias_in_linear_layers"""

            modulated_layers_indices = trial.suggest_categorical(
                "modulated_layers_indices", [[0], [1], [0, 1]]
            )
            cfg["model"]["init_args"]["head_weights"] = head_weights
            cfg["model"]["init_args"][
                "modulated_layers_indices"
            ] = modulated_layers_indices

        conf = copy.deepcopy(cfg)

        conf["model"]["init_args"]["hidden_layers"] = hidden_layers
        conf["model"]["init_args"]["learning_rate"] = learning_rate
        conf["model"]["init_args"]["weight_decay"] = weight_decay
        conf["model"]["init_args"]["dropout"] = dropout
        # Create a new model instance with updated hyperparameters
        model_instance = model.__class__(**conf["model"]["init_args"])

        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
        logger = trainer.logger  # Reuse the existing logger 
        trainer_instance = Trainer(
            max_epochs=100,
            callbacks=[copy.deepcopy(cb) for cb in trainer.callbacks] + [pruning_callback],
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=False,
            devices=cfg["trainer"]['init_args']["devices"],
            accelerator=cfg["trainer"]['init_args']["accelerator"],
            log_every_n_steps=1,
            #check_val_every_n_epoch=1,
        )
        val_loss = train_model(conf, model_instance, datamodule, trainer_instance)
        return val_loss

    def optimize_hyperparameters(n_trials=5):
        cfg = parser.parse_args()
        study = optuna.create_study(direction="minimize", study_name=f"dataset_{cfg['dataset_name']}_{cfg['method_name']}_seed{cfg['seed_everything']}") #, storage="sqlite:///personnalisation_exp.db", load_if_exists=True
        study.optimize(objective, n_trials=n_trials)

        logging.info("Number of finished trials: %d", len(study.trials))
        logging.info("Best trial:")
        trial = study.best_trial

        logging.info("  Value: %f", trial.value)
        logging.info("  Params: ")
        
        for key, value in trial.params.items():
            logging.info("    %s: %s", key, value)
            if key in cfg["model"]["init_args"]:
                cfg["model"]["init_args"][key] = value

        # save the best hyperparameters to a yaml file

        import yaml

        sensitive_attr = "_".join(datamodule.dataset.sensitive_attr)
        out_dir = (
            f"configs/best_hyperparameters/{cfg['dataset_name']}/{cfg['method_name']}/{sensitive_attr}"
        )
        os.makedirs(out_dir, exist_ok=True)
        hyperparam_path = os.path.join(out_dir, "best_hyperparameters.yaml")
        with open(hyperparam_path, "w") as f:
            yaml.dump(trial.params, f)

        return trial.params

    def train_final_model(best_params):
        # Update cfg with best hyperparameters
        for key, value in best_params.items():
            if key in cfg["model"]["init_args"]:
                cfg["model"]["init_args"][key] = value

        model_instance = model.__class__(**cfg["model"]["init_args"])
        model_instance.group_names = datamodule.dataset.group_label_names
        # train_model(cfg, model_instance, datamodule, trainer)
        model_instance.group_names = datamodule.dataset.group_label_names
        if isinstance(model_instance, BiasModulated):
            # train demographic predictor first
            logging.info("Training Demographic Predictor first...")
            demog_pred = model_instance.demographic_model
            demog_trainer = Trainer(
                max_epochs=100,
                callbacks=[copy.deepcopy(cb) for cb in trainer.callbacks],
                #logger=trainer.logger,
                enable_progress_bar=True,
                enable_model_summary=False,
                devices=cfg["trainer"]['init_args']["devices"],
                accelerator=cfg["trainer"]['init_args']["accelerator"],
                log_every_n_steps=5
            )
            demog_trainer.fit(demog_pred, datamodule=datamodule) 
            logging.info("Demographic Predictor training completed.")
            model_instance.set_demographic_model_weights(demog_pred.state_dict())
        elif isinstance(model_instance, DecoupledModel):
            logging.info("Training Decoupled Model...")

            group_labels = datamodule.dataset.nb_group_labels
            val_loss = 0.0
            for g in range(group_labels):
                logging.info(f"Training model for group {g}...")
                model.set_current_group(g)
                group_trainer = Trainer(
                    max_epochs=100,
                    callbacks=[copy.deepcopy(cb) for cb in trainer.callbacks],
                    #logger=trainer.logger,
                    enable_progress_bar=True,
                    enable_model_summary=False,
                    devices=1,
                    accelerator="cuda" if torch.cuda.is_available() else "cpu",
                    log_every_n_steps=5
                )
                group_trainer.fit(model, datamodule=datamodule) 
                logging.info(f"Model training for group {g} completed.") 
            
            test_results = trainer.test(model, datamodule=datamodule)
            logging.info(f"Test Results: {test_results}")
            return test_results 
        trainer.fit(model_instance, datamodule=datamodule)
        test_results = trainer.test(model_instance, datamodule=datamodule)
        logging.info(f"Test Results: {test_results}")

        return test_results


    best_params = optimize_hyperparameters(n_trials=50)
    test_results = train_final_model(best_params)
    sensitive_attr = "_".join(datamodule.dataset.sensitive_attr)
    out_dir = os.path.join("results", "hyperparameter_search", cfg["dataset_name"], cfg["method_name"])
    dir_results = os.path.join(out_dir, sensitive_attr)
    save_test_results(dir_results, test_results, seed=seed)
