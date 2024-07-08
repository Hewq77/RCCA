import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio
import functools

import matplotlib.pyplot as plt

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def _expand_onehot_labels(labels, label_weights, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

def binary_cross_entropy(pred,
                         label,
                         use_sigmoid=False,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    if use_sigmoid:
        loss = F.binary_cross_entropy_with_logits(
            pred, label.float(), weight=class_weight, reduction='none')
    else:
        loss = F.binary_cross_entropy(
            pred, label.float(), weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss

class AffinityLoss(nn.Module):
    def __init__(self, num_classes, down_sample_size, reduction='mean', lambda_b=1.0, lambda_g=1.0,
                 align_corners=False):
        super(AffinityLoss, self).__init__()
        self.num_classes = num_classes
        self.down_sample_size = down_sample_size
        if isinstance(down_sample_size, int):
            self.down_sample_size = [down_sample_size] * 2
        self.reduction = reduction
        self.lambda_b = lambda_b
        self.lambda_g = lambda_g
        self.align_corners = align_corners

    def forward(self, context_prior_map, label):
        # unary loss
        A = self._construct_ideal_affinity_matrix(label, self.down_sample_size)
        unary_loss = binary_cross_entropy(context_prior_map, A)

        diagonal_matrix = (1 - torch.eye(A.shape[1])).to(A.get_device())
        vtarget = diagonal_matrix * A

        # true intra-class rate (recall)
        recall_part = torch.sum(context_prior_map * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)

        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)
        recall_loss = binary_cross_entropy(recall_part, recall_label, reduction=self.reduction)

        # true inter-class rate (specificity)
        spec_part = torch.sum((1 - context_prior_map) * (1 - A), dim=2)
        denominator = torch.sum(1 - A, dim=2)

        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)
        spec_loss = binary_cross_entropy(spec_part, spec_label, reduction=self.reduction)

        # intra-class predictive value (precision)
        precision_part = torch.sum(context_prior_map * vtarget, dim=2)
        denominator = torch.sum(context_prior_map, dim=2)
        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        precision_loss = binary_cross_entropy(precision_part, precision_label, reduction=self.reduction)

        # global_loss
        global_loss = recall_loss + spec_loss + precision_loss

        return self.lambda_b * unary_loss + self.lambda_g * global_loss

    def _construct_ideal_affinity_matrix(self, label, label_size):
        # down sample
        label = torch.unsqueeze(label, dim=1)
        # scaled_labels = label
        scaled_labels = F.interpolate(label.float(), size=label_size, mode="nearest")
        scaled_labels = torch.squeeze(scaled_labels,dim=1).long()

        scaled_labels[scaled_labels == 255] = self.num_classes
        # to one-hot
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
        one_hot_labels = one_hot_labels.view(
            one_hot_labels.size(0), -1, self.num_classes + 1).float()
        # ideal affinity map
        ideal_affinity_matrix = torch.bmm(one_hot_labels,
                                          one_hot_labels.permute(0, 2, 1))

        return ideal_affinity_matrix

