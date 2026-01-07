import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, AUROC
from models.mlp import MLP
import copy
import logging
import numpy as np


class MultiTaskModel(MLP):
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

        # remove last layer from parent MLP
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        nb_heads = (
            nb_groups + 1
        )  # one extra head for label prediction and first head for label prediction

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
            + f"For multitask learning with {len(self.heads)} heads."
        )

    def forward(self, x):
        features = self.model(x)
        # get internal layers embeddings
        emb = features.detach()
        head_logits = []
        # s is matrix of shape (batch_size, num_sensitive_attributes)
        for head in self.heads:
            group_pred = head(features)
            head_logits.append(group_pred)

        return emb, head_logits

    def _step(self, batch, stage):
        x, y, s, _ = batch
        # s is matrix of shape (batch_size, num_sensitive_attributes)
        emb, logits = self(x)
        logging.debug(f"Logits shape: {[logit.shape for logit in logits]}")
        loss = 0
        preds = []
        head_losses = []
        for i in range(len(self.heads)):
            if i == 0:  # first head is for label prediction
                labels = y
            else:
                idx = i - 1  # adjust index for sensitive attributes
                labels = s[:, idx]
            head_logits = logits[i]  # get logits for head i

            loss += self.head_weights[i] * self.criterion(
                head_logits, labels
            )  # assuming labels is the sensitive attribute label for head i
            # head_losses.append(loss)
            preds = torch.argmax(head_logits, dim=1)
            self.log_metrics(preds, labels, loss, stage, head_idx=i)

        # loss = torch.mean(w * hl for w, hl in zip(self.head_weights, head_losses))

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def _get_logits(self, x):
        emb, logits = self(x)
        return logits[0]  # return only label head logits

    def log_metrics(self, preds, labels, loss, stage, head_idx=None):
        log_label = "_label" if head_idx == 0 else f"_head_{head_idx - 1}"
        if stage == "train" and head_idx is not None:
            train_acc = self.per_head_train_acc[head_idx].to(preds.device)(preds, labels)
            self.log(f"{stage}{log_label}_acc", train_acc, on_step=False, on_epoch=True)
            self.log(f"{stage}{log_label}_loss", loss, on_step=False, on_epoch=True)
        elif stage == "val" and head_idx is not None:
            val_acc = self.per_head_val_acc[head_idx].to(preds.device)(preds, labels)
            self.log(
                f"{stage}{log_label}_acc",
                val_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )
            self.log(
                f"{stage}{log_label}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        elif stage == "test" and head_idx is not None:
            log_label = "" if head_idx == 0 else f"_head_{head_idx - 1}"
            test_acc = self.per_head_test_acc[head_idx].to(preds.device)(preds, labels)
            test_roc = self.per_head_roc[head_idx].to(preds.device)(preds, labels)

            self.log(
                f"{stage}{log_label}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}{log_label}_acc",
                test_acc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}_group_{head_idx}_auroc",
                test_roc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return
