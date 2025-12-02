from torchmetrics import Metric
import torch


class DemParityMetric(Metric):
    """Calculate the demographic parity difference.

    The demographic parity difference is defined as the difference
    between the largest and the smallest group-level selection rate,
    :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature(s).
    The demographic parity difference of 0 means that all groups have the same selection rate.

    Parameters
    ----------

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed

    Returns
    -------
    float
        The demographic parity difference
    """

    is_differentiable: False
    higher_is_better: False
    full_state_update: bool = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.add_state("pos_num_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pos_den_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pos_num_1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pos_den_1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(self, preds: torch.Tensor, sensitive_features: torch.Tensor):
        assert preds.shape[0] == sensitive_features.shape[0]

        y_pred = (preds > self.threshold).long()
        s = sensitive_features.long()

        # group 0
        self.pos_num_0 += torch.sum((y_pred == 1) & (s == 0))
        self.pos_den_0 += torch.sum(s == 0)

        # group 1
        self.pos_num_1 += torch.sum((y_pred == 1) & (s == 1))
        self.pos_den_1 += torch.sum(s == 1)

    def compute(self):
        p1 = _safe_divide(self.pos_num_1, self.pos_den_1)
        p0 = _safe_divide(self.pos_num_0, self.pos_den_0)
        return torch.abs(p1 - p0)


class EqualizedOdds(Metric):
    """Calculate the equalized odds difference.

    The sum of two metrics: `true_positive_rate_difference` and
    `false_positive_rate_difference`. The former is the difference between the
    largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
    of the sensitive feature(s). The latter is defined similarly, but for
    :math:`P[h(X)=1 | A=a, Y=0]`.
    The equalized odds difference of 0 means that all groups have the same
    true positive, true negative, false positive, and false negative rates.

    Parameters
    ----------
    preds : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    y_true : array-like
        Ground truth (correct) labels.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed

    Returns
    -------
    float
        The equalized odds difference
    """

    is_differentiable: False
    higher_is_better: False
    full_state_update: bool = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        # True positives
        self.add_state("tp_num_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_den_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_num_1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_den_1", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # False positives
        self.add_state("fp_num_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp_den_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp_num_1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp_den_1", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.threshold = threshold

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sensitive_features: torch.Tensor,
    ):
        assert preds.shape[0] == sensitive_features.shape[0]

        y_pred = (preds > self.threshold).long()
        y_true = target.long()
        s = sensitive_features.long()

        # True positives
        self.tp_num_0 += torch.sum((y_pred == 1) & (y_true == 1) & (s == 0))
        self.tp_den_0 += torch.sum((y_true == 1) & (s == 0))

        self.tp_num_1 += torch.sum((y_pred == 1) & (y_true == 1) & (s == 1))
        self.tp_den_1 += torch.sum((y_true == 1) & (s == 1))

        # False positives
        self.fp_num_0 += torch.sum((y_pred == 1) & (y_true == 0) & (s == 0))
        self.fp_den_0 += torch.sum((y_true == 0) & (s == 0))

        self.fp_num_1 += torch.sum((y_pred == 1) & (y_true == 0) & (s == 1))
        self.fp_den_1 += torch.sum((y_true == 0) & (s == 1))

    def compute(self):
        tpr_0 = _safe_divide(self.tp_num_0, self.tp_den_0)
        tpr_1 = _safe_divide(self.tp_num_1, self.tp_den_1)
        fpr_0 = _safe_divide(self.fp_num_0, self.fp_den_0)
        fpr_1 = _safe_divide(self.fp_num_1, self.fp_den_1)

        return torch.max(torch.abs(tpr_1 - tpr_0), torch.abs(fpr_1 - fpr_0))


class EqualOpportunity(Metric):
    """Equal Opportunity Difference.

    Computes the absolute difference in true positive rates (TPR)
    between two sensitive groups (0 and 1).

    Equal Opportunity = |TPR_s=1 - TPR_s=0|
    """

    is_differentiable: False
    higher_is_better: False
    full_state_update: bool = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.add_state("tp_num_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_den_0", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_num_1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("tp_den_1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.threshold = threshold

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sensitive_features: torch.Tensor,
    ):
        assert preds.shape[0] == sensitive_features.shape[0]
        y_pred = (preds > self.threshold).long()
        y_true = target.long()
        s = sensitive_features.long()

        self.tp_num_0 += torch.sum((y_pred == 1) & (y_true == 1) & (s == 0))
        self.tp_den_0 += torch.sum((y_true == 1) & (s == 0))

        self.tp_num_1 += torch.sum((y_pred == 1) & (y_true == 1) & (s == 1))
        self.tp_den_1 += torch.sum((y_true == 1) & (s == 1))

    def compute(self):
        tpr_0 = _safe_divide(self.tp_num_0, self.tp_den_0)
        tpr_1 = _safe_divide(self.tp_num_1, self.tp_den_1)
        return torch.abs(tpr_1 - tpr_0)


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Performs safe division of two tensors."""
    return torch.where(
        denominator != 0, numerator / denominator, torch.zeros_like(numerator)
    )
