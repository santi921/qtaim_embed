import torch
from torch import Tensor

import torch.nn.functional as F
import torchmetrics

from typing import Optional

BaseMetric = torchmetrics.Metric


class LinkPredMetric(BaseMetric):
    r"""An abstract class for computing link prediction retrieval metrics.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """
    is_differentiable: bool = False
    full_state_update: bool = False
    higher_is_better: Optional[bool] = None

    def __init__(self) -> None:
        super().__init__()

        self.accum: Tensor
        self.total: Tensor

        self.add_state("accum", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        pos_score: Tensor,
        neg_score: Tensor,
    ) -> None:
        r"""Updates the state variables based on the current mini-batch
        prediction.

        :meth:`update` can be repeated multiple times to accumulate the results
        of successive predictions, *e.g.*, inside a mini-batch training or
        evaluation loop.

        Args:
            pos_score (torch.Tensor): The predicted scores for the positive
                examples in the mini-batch.
            neg_score (torch.Tensor): The predicted scores for the negative
                examples in the mini-batch.

        """

        metric = self._compute(pos_score, neg_score)
        self.accum += metric.sum()  # sum of the metric
        self.total += pos_score.shape[
            0
        ]  # counts the number of positive examples(links)

    def compute(self) -> Tensor:
        r"""Computes the final metric value."""
        # if self.total == 0:
        #    return torch.zeros_like(self.accum)

        return self.accum / self.total

    def reset(self) -> None:
        r"""Reset metric state variables to their default value."""
        super().reset()

    def _compute(self, pos_score, neg_scorer) -> Tensor:
        r"""Compute the specific metric.
        To be implemented separately for each metric class.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


class AUCMetric(LinkPredMetric):
    r"""Computes the AUCROC loss for link prediction."""

    def __init__(self) -> None:
        super().__init__()

    def _compute(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        val = compute_auc(pos_score, neg_score)
        # return value as tensor copied over pos_score number of times
        return torch.tensor([val for _ in range(pos_score.shape[0])])


class AccuracyMetric(LinkPredMetric):
    r"""Computes the Accuracy for link prediction."""

    def __init__(self) -> None:
        super().__init__()

    def _compute(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        val = compute_accuracy(pos_score, neg_score)
        # return value as tensor copied over pos_score number of times
        return torch.tensor([val for _ in range(pos_score.shape[0])])


class HingeMetric(LinkPredMetric):
    r"""Computes the Hinge loss for link prediction."""

    def __init__(self) -> None:
        super().__init__()

    def _compute(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:

        return compute_loss_hinge(pos_score, neg_score)


class F1Metric(LinkPredMetric):
    r"""Computes the F1 score for link prediction."""

    def __init__(self) -> None:
        super().__init__()

    def _compute(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        n_edges = pos_score.shape[0]
        n_edges_neg = neg_score.shape[0]
        y_true = torch.cat(
            [
                torch.ones(n_edges, device=pos_score.device),
                torch.zeros(n_edges_neg, device=pos_score.device),
            ]
        )
        y_pred = torch.cat([pos_score, neg_score])
        return torchmetrics.functional.f1_score(
            y_pred, y_true, task="binary", average="macro"
        )

    def update(
        self,
        pos_score: Tensor,
        neg_score: Tensor,
    ) -> None:
        r"""Updates the state variables based on the current mini-batch
        prediction.
        Args:
            pos_score (torch.Tensor): The predicted scores for the positive
                examples in the mini-batch.
            neg_score (torch.Tensor): The predicted scores for the negative
                examples in the mini-batch.

        """

        metric = self._compute(pos_score, neg_score)
        self.accum += metric.sum()
        self.total += 1


class MarginMetric(LinkPredMetric):
    r"""Computes the Margin loss for link prediction."""

    def __init__(self) -> None:
        super().__init__()

    def _compute(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        return compute_loss_margin(pos_score, neg_score)


class CrossEntropyMetric(LinkPredMetric):
    r"""Computes the Cross Entropy loss for link prediction.

    Args:
        k (int): The number of top-:math:`k` predictions to evaluate against.
    """

    def __init__(self) -> None:
        super().__init__()

    def update(
        self,
        pos_score: Tensor,
        neg_score: Tensor,
    ) -> None:
        r"""Updates the state variables based on the current mini-batch
        prediction.
        Args:
            pos_score (torch.Tensor): The predicted scores for the positive
                examples in the mini-batch.
            neg_score (torch.Tensor): The predicted scores for the negative
                examples in the mini-batch.

        """

        metric = self._compute(pos_score, neg_score)
        self.accum += metric.sum()  # sum of the metric
        self.total += (
            2 * pos_score.shape[0]
        )  # counts the number of positive examples(links)

    def _compute(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        return compute_loss_cross_entropy(pos_score, neg_score)


def compute_loss_hinge(pos_score, neg_score):
    """
    Hinge loss.
    Args:
        pos_score: Tensor of shape (n_edges, 1)
        neg_score: Tensor of shape (n_edges, 1)
    Return:
        loss: Tensor of shape (1,)
    """
    n = pos_score.shape[0]
    return (
        (neg_score.view(n, -1) - pos_score.view(n, -1) + 1).clamp(min=0).reshape(-1)
    )  # .sum()


def compute_loss_margin(pos_score, neg_score):
    """
    Margin loss.
    Args:
        pos_score: Tensor of shape (n_edges, 1)
        neg_score: Tensor of shape (n_edges, 1)
    Return:
        loss: Tensor of shape (1,)
    """

    return (1 - pos_score + neg_score).clamp(min=0)  # .sum()


def compute_loss_cross_entropy(pos_score, neg_score):
    """
    Cross entropy loss.
    Args:
        pos_score: Tensor of shape (n_edges, 1)
        neg_score: Tensor of shape (n_edges, 1)
    Return:
        loss: Tensor of shape (1,)
    """

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [
            torch.ones(pos_score.shape[0], device=scores.device),
            torch.zeros(neg_score.shape[0], device=scores.device),
        ]
    )
    # print("device", scores.device)
    # print("device", labels.device)
    return F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    # return F.binary_cross_entropy_with_logits(scores, labels, reduction="sum")


def compute_auc(pos_score, neg_score):
    """
    AUCROC loss.
    Args:
        pos_score: Tensor of shape (n_edges, 1)
        neg_score: Tensor of shape (n_edges, 1)
    Return:
        loss: Tensor of shape (1,)
    """

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [
            torch.ones(pos_score.shape[0], device=scores.device),
            torch.zeros(neg_score.shape[0], device=scores.device),
        ]
    )
    labels = labels.int()
    return torchmetrics.functional.auroc(
        preds=scores, target=labels, average=None, task="binary"
    )


def compute_accuracy(pos_score, neg_score):
    """
    Accuracy.
    Args:
        pos_score: Tensor of shape (n_edges, 1)
        neg_score: Tensor of shape (n_edges, 1)
    Return:
        loss: Tensor of shape (1,)
    """

    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [
            torch.ones(pos_score.shape[0], device=scores.device),
            torch.zeros(neg_score.shape[0], device=scores.device),
        ],
    )
    return torchmetrics.functional.accuracy(scores, labels, average=None, task="binary")
