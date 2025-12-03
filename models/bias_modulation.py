import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, AUROC, CalibrationError
from metrics.fairness import DemParityMetric, EqualizedOdds, EqualOpportunity
from models.mlp import MLP
import copy
import logging
import numpy as np
import torch.nn.functional as F


class ModulatedLinear(nn.Linear):
    def forward(self, x, bias_vect=None):
        """
        offset: tensor of shape (out_features,) or None.
                If None, uses the normal bias.
        """
        if self.bias is None:
            # No bias in the layer, use the provided bias_vect directly
            return F.linear(x, self.weight, bias_vect)

        if bias_vect is None:
            # Normal behavior
            return F.linear(x, self.weight, self.bias)
        else:
            # Modulate the bias with the provided bias_vect
            # with torch.no_grad():
            #    bias = self.bias + bias_vect
            return F.linear(x, self.weight, self.bias + bias_vect)


class MultiTargetDemographicModel(MLP):
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
        head_weights=None,
    ):
        super().__init__(
            input_dim,
            hidden_layers,
            output_dim,
            dropout,
            learning_rate,
            weight_decay,
            activation,
            nb_groups,
            nb_group_labels,
        )

        prev_dim = input_dim
        self.layers = []
        for hidden_dim in hidden_layers:
            layers = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.layers.append(layers)
            prev_dim = hidden_dim

        # self.layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.layers = nn.ModuleList(self.layers)
        del self.model  # remove the base model

        nb_heads = nb_group_labels  # one head per sensitive attribute label

        if head_weights is not None and len(head_weights) > nb_heads:
            self.head_weights = head_weights
        else:
            # uniform weights for each head
            logging.info(
                f"Setting uniform weights for each of the {nb_heads} heads. Ensure length of head_weights matches number of heads {nb_heads}."
            )
            self.head_weights = [1.0 / nb_heads for _ in range(nb_heads)]

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_layers[-1], output_dim) for _ in range(nb_heads)]
        )

        self.per_head_train_acc = [
            Accuracy(task="binary", threshold=0.5) for _ in range(len(self.heads))
        ]
        self.per_head_val_acc = [
            Accuracy(task="binary", threshold=0.5) for _ in range(len(self.heads))
        ]
        self.per_head_test_acc = [
            Accuracy(task="binary", threshold=0.5) for _ in range(len(self.heads))
        ]

        self.per_head_roc = [AUROC(task="binary") for _ in range(len(self.heads))]

        self.head_weights = [1.0 / len(self.heads) for _ in range(len(self.heads))]

    def _get_log_model_info(self):
        return (
            super()._get_log_model_info()
            + f"For sensitive attribute prediction with {len(self.heads)} heads."
        )


class DemographicPredictor(MLP):
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
        head_weights=None,
    ):
        super().__init__(
            input_dim,
            hidden_layers,
            output_dim,
            dropout,
            learning_rate,
            weight_decay,
            activation,
            nb_groups,
            nb_group_labels,
        )

        prev_dim = input_dim
        self.layers = []
        for hidden_dim in hidden_layers:
            layers = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.layers.append(layers)
            prev_dim = hidden_dim

        # self.layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.layers = nn.ModuleList(self.layers)
        del self.model  # remove the base model

        nb_heads = nb_groups  # one head per sensitive attribute

        if head_weights is not None and len(head_weights) > nb_heads:
            self.head_weights = head_weights
        else:
            # uniform weights for each head
            logging.info(
                f"Setting uniform weights for each of the {nb_heads} heads. Ensure length of head_weights matches number of heads {nb_heads}."
            )
            self.head_weights = [1.0 / nb_heads for _ in range(nb_heads)]

        self.heads = nn.ModuleList(
            [nn.Linear(hidden_layers[-1], output_dim) for _ in range(nb_heads)]
        )

        self.per_head_train_acc = [
            Accuracy(task="binary", threshold=0.5) for _ in range(len(self.heads))
        ]
        self.per_head_val_acc = [
            Accuracy(task="binary", threshold=0.5) for _ in range(len(self.heads))
        ]
        self.per_head_test_acc = [
            Accuracy(task="binary", threshold=0.5) for _ in range(len(self.heads))
        ]

        self.per_head_roc = [AUROC(task="binary") for _ in range(len(self.heads))]

        self.head_weights = [1.0 / len(self.heads) for _ in range(len(self.heads))]

    def _get_log_model_info(self):
        return (
            super()._get_log_model_info()
            + f"For sensitive attribute prediction with {len(self.heads)} heads."
        )

    def forward(self, x):
        embeddings = []

        for layer in self.layers:
            x = layer(x)
            embeddings.append(x.detach())  # get internal layers embeddings

        head_logits = []
        # s is matrix of shape (batch_size, num_sensitive_attributes)
        for head in self.heads:
            group_pred = head(x)
            head_logits.append(group_pred)

        return embeddings, head_logits

    def _step(self, batch, stage):
        x, y, s, _ = batch
        # s is matrix of shape (batch_size, num_sensitive_attributes)
        emb, logits = self(x)
        logging.debug(f"Logits shape: {[logit.shape for logit in logits]}")
        loss = 0
        preds = []
        head_losses = []
        for i in range(len(self.heads)):
            labels = s[:, i]
            head_logits = logits[i]  # get logits for head i

            loss = self.head_weights[i] * self.criterion(
                head_logits, labels
            )  # assuming labels is the sensitive attribute label for head i
            head_losses.append(loss)
            # head_losses.append(loss)
            preds = torch.argmax(head_logits, dim=1)
            self.log_metrics(preds, labels, loss, stage, head_idx=i)

        loss = sum(head_losses)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def compute_per_group_acc(self, batch):
        x, y, s, group_idx = batch
        _preds = []
        _s = []
        _y = []
        for g in range(self.nb_groups):
            idx = group_idx == g
            emb, logits = self(x[idx])
            logits = logits[0]  # first head is for label prediction, ignore others
            # logits = self(x, group_idx=group_idx)
            preds = torch.argmax(logits, dim=1)
            # loss = loss.mean()
            self.log_group_metrics(logits, y[idx], g)

            _preds.append(preds)
            _y.append(y[idx])
            _s.append(s[idx])
        return torch.cat(_preds), torch.cat(_y), torch.cat(_s)

    def log_metrics(self, preds, labels, loss, stage, head_idx=None):
        log_label = f"head_{head_idx}" if head_idx is not None else ""
        if stage == "train" and head_idx is not None:
            train_acc = self.per_head_train_acc[head_idx](preds, labels)
            self.log(
                f"{stage}_{log_label}_acc",
                train_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_{log_label}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        elif stage == "val" and head_idx is not None:
            val_acc = self.per_head_val_acc[head_idx](preds, labels)
            self.log(
                f"{stage}_{log_label}_acc",
                val_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                f"{stage}_{log_label}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        elif stage == "test" and head_idx is not None:
            log_label = f"head_{head_idx}"
            test_acc = self.per_head_test_acc[head_idx](preds, labels)
            test_roc = self.per_head_roc[head_idx](preds, labels)

            self.log(
                f"{stage}_{log_label}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_{log_label}_acc",
                test_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            """ self.log(
                f"{stage}_group_{head_idx}_auroc",
                test_roc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
             self.log(
                f"{stage}_group_{head_idx}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            ) """
        return


class BiasModulated(MLP):
    def __init__(
        self,
        input_dim,
        hidden_layers,
        output_dim,
        dropout,
        learning_rate,
        weight_decay,
        activation="relu",
        nb_groups=4,
        nb_group_labels=4,
        demographic_model: DemographicPredictor = None,
        remove_bias_in_linear_layers=False,
        modulated_layers_indices=None,
        modulated_at_inference=False,
        head_weights=None,
    ):
        super().__init__(
            input_dim,
            hidden_layers,
            output_dim,
            dropout,
            learning_rate,
            weight_decay,
            activation,
            nb_groups,
            nb_group_labels,
        )

        if demographic_model is None:
            self.demographic_model = DemographicPredictor(
                input_dim,
                hidden_layers,
                output_dim,
                dropout,
                learning_rate,
                weight_decay,
                activation,
                nb_groups=nb_groups,
                nb_group_labels=nb_group_labels,
                head_weights=head_weights,
            )
        else:
            self.demographic_model = demographic_model

        self.layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers = nn.Sequential(
                ModulatedLinear(
                    prev_dim,
                    hidden_dim,
                    bias=False if remove_bias_in_linear_layers else True,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.layers.append(layers)
            prev_dim = hidden_dim
        self.layers = nn.ModuleList(self.layers)
        # self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)

        self.modulated_layers_indices = (
            modulated_layers_indices
            if modulated_layers_indices is not None
            else list(range(len(self.layers)))
        )

        self.modulated_at_inference = modulated_at_inference

        del self.model  # remove the base model

    def set_demographic_model_weights(self, state_dict):
        if self.demographic_model is None:
            raise ValueError("Demographic model is not initialized.")
        self.demographic_model.load_state_dict(state_dict)

    def forward(self, x, is_bias_modulation=None):
        sensitive_embeddings, _ = self.demographic_model(x)

        for i, layer in enumerate(self.layers):
            if is_bias_modulation and i in self.modulated_layers_indices:
                logging.debug(f"Applying bias modulation at layer {i}")
                # mainly used during training
                bias_vect = sensitive_embeddings[i]
                x = layer[0](x, bias_vect=bias_vect)  # ModulatedLinear layer
                x = layer[1](x)  # ReLU
                x = layer[2](x)  # Dropout
            else:
                logging.debug(f"No bias modulation at layer {i}")
                x = layer(x)
        x = self.output_layer(x)
        return x

    def _get_logits(self, x):
        return self.forward(x, is_bias_modulation=self.modulated_at_inference)

    def _step(self, batch, stage):
        x, y, s, _ = batch
        is_bias_modulation = True if stage == "train" else self.modulated_at_inference
        logits = self(x, is_bias_modulation=is_bias_modulation)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log_metrics(preds, y, s, loss, stage)
        return loss
