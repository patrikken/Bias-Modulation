import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryFairness,
    Accuracy,
    AUROC,
    CalibrationError,
)
from metrics.fairness import DemParityMetric, EqualizedOdds, EqualOpportunity
import logging


class MLP(L.LightningModule):
    """
    A simple Multi-Layer Perceptron (MLP) model for classification tasks.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_layers (list of int): List containing the number of units in each hidden layer.
        output_dim (int): Dimension of the output layer (number of classes).
        dropout (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) for the optimizer.
        nb_groups (int): Number of sensitive attribute groups. For example, if sensitive attribute is age with 2 groups (young, old), then nb_groups=1. If sensitive attribute is age and sex with 2 age groups and 2 sex groups, then nb_groups=4.
        nb_group_labels (int): Number of unique sensitive attribute labels. For example, if sensitive attribute is age and sex with 2 age groups and 2 sex groups, then nb_group_labels=4. This helps for reporting per group metrics.
    """

    def __init__(
        self,
        input_dim,
        hidden_layers,
        output_dim,
        dropout,
        learning_rate,
        weight_decay,
        activation="relu",
        nb_groups=2,
        nb_group_labels=4,
    ):
        super(MLP, self).__init__()
        layers = []
        self.nb_groups = nb_groups
        self.nb_group_labels = nb_group_labels
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        logging.info(
            f"Initializing MLP with input_dim={input_dim}, hidden_layers={hidden_layers}, output_dim={output_dim}, dropout={dropout}, learning_rate={learning_rate}, weight_decay={weight_decay}, activation={activation}"
        )
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

        self.test_acc = Accuracy(task="binary", threshold=0.5)
        self.test_auroc = AUROC(task="binary")

        self.train_acc = Accuracy(task="binary", threshold=0.5)
        self.train_auroc = AUROC(task="binary")

        self.val_acc = Accuracy(task="binary", threshold=0.5)
        self.val_auroc = AUROC(task="binary")

        self.per_group_acc = [
            Accuracy(task="binary", threshold=0.5) for _ in range(self.nb_group_labels)
        ]
        self.per_group_auroc = [
            AUROC(task="binary") for _ in range(self.nb_group_labels)
        ]
        self.per_group_ece = [
            CalibrationError(task="multiclass", num_classes=output_dim)
            for _ in range(self.nb_group_labels)
        ]
        self.group_names = None

    def _get_log_model_info(self):
        return f"Instantiated MLP. "

    def _log_model_info(self):
        logging.info(self._get_log_model_info())

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y, s, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log_metrics(preds, y, s, loss, stage)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.compute_per_group_acc(batch)
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def _get_logits(self, x):
        # Childs classes may override this method to provide custom forward logic
        return self(x)

    def compute_per_group_acc(self, batch, stage="test"):
        x, y, s, group_idx = batch
        _preds = []
        _s = []
        _y = []
        for g in range(self.nb_group_labels):
            idx = group_idx == g
            logits = self._get_logits(x[idx])
            # logits = self(x, group_idx=group_idx)
            preds = torch.argmax(logits, dim=1)
            # loss = loss.mean()
            self.log_group_metrics(logits, y[idx], g, stage)
            _preds.append(preds)
            _y.append(y[idx])
            _s.append(s[idx])
        return torch.cat(_preds), torch.cat(_y), torch.cat(_s)

    def log_group_metrics(self, logits, labels, group_id, stage="test"):
        """
        Log metrics for a specific group.
        Args:
            logits: Model predictions for the group.
            labels: True labels for the group.
            group_id: ID of the group.
            stage: Stage of evaluation (train/val/test). Only "test" is implemented here.
        """
        preds = torch.argmax(logits, dim=1)
        group_acc_score = self.per_group_acc[group_id]
        acc = group_acc_score(preds, labels)
        auc = self.per_group_auroc[group_id](preds, labels)
        loss = self.criterion(logits, labels)
        ece = self.per_group_ece[group_id](logits, labels)
        group_name = (
            self.group_names[group_id]
            if self.group_names is not None and len(self.group_names) > group_id
            else group_id
        )
        self.log(f"{stage}_group_{group_name}_auroc", auc, sync_dist=True)
        self.log(f"{stage}_group_{group_name}_acc", acc, sync_dist=True)
        self.log(f"{stage}_group_{group_name}_loss", loss, sync_dist=True)
        self.log(f"{stage}_group_{group_name}_ece", ece, sync_dist=True)

    def log_metrics(self, preds, y, s, loss, stage):
        if stage == "test":
            acc = self.test_acc(preds, y)
            auroc = self.test_auroc(preds, y)
        elif stage == "train":
            acc = self.train_acc(preds, y)
            auroc = self.train_auroc(preds, y)
        else:
            acc = self.val_acc(preds, y)
            auroc = self.val_auroc(preds, y)

        self.log(f"{stage}_loss", loss, prog_bar=stage == "val", sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=stage == "val", sync_dist=True)
        self.log(f"{stage}_auroc", auroc, prog_bar=False, sync_dist=True)


class MLP_Fairness(L.LightningModule):
    """
    A simple Multi-Layer Perceptron (MLP) model for classification tasks.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_layers (list of int): List containing the number of units in each hidden layer.
        output_dim (int): Dimension of the output layer (number of classes).
        dropout (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay (L2 regularization) for the optimizer.
    """

    def __init__(
        self,
        input_dim,
        hidden_layers,
        output_dim,
        dropout,
        learning_rate,
        weight_decay,
        activation="relu",
    ):
        super(MLP, self).__init__()
        layers = []

        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        self.fairness_metric = BinaryFairness(2)
        self.dp_fairness = DemParityMetric()
        self.eo_fairness = EqualizedOdds()
        self.eop_fairness = EqualOpportunity()

    def forward(self, x):
        return self.model(x)

    def log_metrics(self, preds, y, s, loss, stage):
        super().log_metrics(preds, y, loss, stage)
        if stage == "test" or stage == "val":
            fairness = self.fairness_metric(preds, y, s) if stage == "test" else {}
            fairness["dp_diff"] = self.dp_fairness(preds, s)
            fairness["eo_diff"] = self.eo_fairness(preds, y, s)
            fairness["eop_diff"] = self.eop_fairness(preds, y, s)
            for metric_name, metric_value in fairness.items():
                self.log(
                    f"{stage}_{metric_name}",
                    metric_value,
                    prog_bar=False,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
