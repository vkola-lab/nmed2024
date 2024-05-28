import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class SigmoidFocalLoss(nn.Module):
    ''' ... '''
    def __init__(
        self,
        alpha: float = -1,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        ''' ... '''
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ''' ... '''
        p = torch.sigmoid(input)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class SigmoidFocalLossBeta(nn.Module):
    ''' ... '''
    def __init__(
        self,
        beta: float = 0.9999,
        gamma: float = 2.0,
        num_per_cls = (1, 1),
        reduction: str = 'mean',
    ):
        ''' ... '''
        super().__init__()
        eps = sys.float_info.epsilon
        self.gamma = gamma
        self.reduction = reduction

        # weights to balance loss
        self.weight_neg = ((1 - beta) / (1 - beta ** num_per_cls[0] + eps))
        self.weight_pos = ((1 - beta) / (1 - beta ** num_per_cls[1] + eps))
        # weight_neg = (1 - beta) / (1 - beta ** num_per_cls[0])
        # weight_pos = (1 - beta) / (1 - beta ** num_per_cls[1])
        # self.weight_neg = weight_neg / (weight_neg + weight_pos)
        # self.weight_pos = weight_pos / (weight_neg + weight_pos)

    def forward(self, input, target):
        ''' ... '''
        p = torch.sigmoid(input)
        p_t = p * target + (1 - p) * (1 - target)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        alpha_t = self.weight_pos * target + self.weight_neg * (1 - target)
        loss = alpha_t * loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, alpha=0.5, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.alpha = alpha
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps


    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = self.alpha*los_pos + (1-self.alpha)*los_neg
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss#.sum()
