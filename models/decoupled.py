import torch
import torch.nn as nn
from models.mlp import MLP


class DecoupledModel(MLP):
    """A model that maintains separate MLPs for each sensitive attribute group.

    Each group has its own dedicated MLP, allowing for independent learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list,
        output_dim: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        activation="relu",
        *xargs,
        **kwargs,
    ):
        super(DecoupledModel, self).__init__(
            input_dim,
            hidden_layers,
            output_dim,
            dropout,
            learning_rate,
            weight_decay,
            activation,
            *xargs,
            **kwargs,
        )
        """ self.models_per_group = nn.ModuleList(
            [copy.deepcopy(self.model) for _ in range(self.nb_groups)]
        ) """
        self.models_per_group = nn.ModuleList(
            [
                MLP(
                    input_dim,
                    hidden_layers,
                    output_dim,
                    dropout,
                    learning_rate,
                    weight_decay,
                    activation,
                )
                for _ in range(self.nb_group_labels)
            ]
        )
        self.model = None  # Disable the base model
        self.current_group = None

    def _get_log_model_info(self):
        return (
            super()._get_log_model_info()
            + f"For DecoupledModel with {self.nb_groups} separate models."
        )

    def forward(self, x, model_idx=None):
        if model_idx is None:
            raise ValueError("model_idx must be provided for DecoupledModel.")

        assert (
            0 <= model_idx < self.nb_group_labels
        ), f"model_idx must be between 0 and {self.nb_group_labels - 1}"

        # print(f"Forward pass for group index: {group_idx}")
        return self.models_per_group[model_idx](x)

    def training_step(self, batch, batch_idx):
        if self.current_group is None:
            raise ValueError("current_group must be set before training_step.")

        x, y, s, _ = batch
        logits = self(x, model_idx=self.current_group)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log_metrics(preds, y, s, loss, "train")
        return loss

    def training_step(self, batch, batch_idx):
        if self.current_group is None:
            raise ValueError("current_group must be set before training_step.")

        x, y, s, _ = batch
        logits = self(x, model_idx=self.current_group)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log_metrics(preds, y, s, loss, "val")
        return loss

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def set_current_group(self, group_idx):
        assert (
            0 <= group_idx < self.nb_group_labels
        ), f"group_idx must be between 0 and {self.nb_group_labels - 1}"

        self.current_group = group_idx

    def _step(self, batch, stage):
        x, y, s, group_idx = batch
        _preds = []
        _s = []
        _y = []
        _loss = []
        for g in range(self.nb_group_labels):
            # perform per_dataset training.
            idx = group_idx == g
            labels = y[idx]
            logits = self(x[idx], model_idx=g)
            # logits = self(x, group_idx=group_idx)
            loss = torch.tensor(
                self.criterion(logits, labels).detach().cpu().numpy()
            ).unsqueeze(0)
            preds = torch.argmax(logits, dim=1)
            # loss = loss.mean()
            if stage == "test":
                self.log_group_metrics(logits, labels, g, stage)
            _preds.append(preds)
            _y.append(labels)
            _s.append(s[idx])
            _loss.append(loss)
        self.log_metrics(
            torch.cat(_preds, dim=0),
            torch.cat(_y, dim=0),
            torch.cat(_s, dim=0),
            torch.cat(_loss, dim=0).mean(),
            stage,
        )
        # return loss

    def log_metrics(self, preds, y, s, loss, stage):
        super().log_metrics(preds, y, s, loss, stage)
