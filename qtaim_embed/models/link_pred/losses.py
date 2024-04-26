from typing import Optional, Tuple, Union

import torch 
from torch import Tensor

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import torchmetrics
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

        self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')


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
        self.accum += metric.sum() # sum of the metric
        self.total += pos_score.shape[0]# counts the number of positive examples(links)


    def compute(self) -> Tensor:
        r"""Computes the final metric value."""
        #if self.total == 0:
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
        return f'{self.__class__.__name__}(k={self.k})'


class HingeMetric(LinkPredMetric):
    r"""Computes the Hinge loss for link prediction.

    """
    def __init__(self) -> None:
        super().__init__()

    def _compute(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        
        return compute_loss_hinge(pos_score, neg_score)


class MarginMetric(LinkPredMetric):
    r"""Computes the Margin loss for link prediction.

    """
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
        self.accum += metric.sum() # sum of the metric
        self.total += 2 * pos_score.shape[0]# counts the number of positive examples(links)


    def _compute(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        return compute_loss_cross_entropy(pos_score, neg_score)


class AUCMetric(LinkPredMetric):
    r"""Computes the AUCROC loss for link prediction.

    """
    def __init__(self) -> None:
        super().__init__()

    def _compute(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        return compute_auc(pos_score, neg_score)


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
    return (neg_score.view(n, -1) - pos_score.view(n, -1) + 1).clamp(min=0).reshape(-1)#.sum()


def compute_loss_margin(pos_score, neg_score):
    """
    Margin loss.
    Args:
        pos_score: Tensor of shape (n_edges, 1)
        neg_score: Tensor of shape (n_edges, 1)
    Return:
        loss: Tensor of shape (1,)
    """
    n_edges = pos_score.shape[0]

    return (1 - pos_score + neg_score).clamp(min=0)#.sum()


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
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
    #return F.binary_cross_entropy_with_logits(scores, labels, reduction="sum")


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
        [torch.ones(pos_score.shape[0], dtype=torch.int), torch.zeros(neg_score.shape[0], dtype=torch.int)])
    #return roc_auc_score(labels, scores)
    # use torchmetrics instead
    return torchmetrics.functional.auroc(scores, labels, task="binary", average="Sum")


def compute_accuracy(pos_score, neg_score):
    """
    Accuracy.
    Args:
        pos_score: Tensor of shape (n_edges, 1)
        neg_score: Tensor of shape (n_edges, 1)
    Return:
        loss: Tensor of shape (1,)
    """
    n_edges = pos_score.shape[0]
    return ((pos_score > neg_score).float().sum() / n_edges).reshape(-1)