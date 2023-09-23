import torch
from torch.nn.modules.loss import _WeightedLoss
import math


class FH1Loss(object):
    def __init__(self, res=256, modes=32):
        super().__init__()
        self.res = res
        self.eps = 1e-10
        k_x = torch.cat((torch.arange(start=0, end=res // 2, step=1), torch.arange(start=-res // 2, end=0, step=1)),
                        0).reshape(res, 1).repeat(1, res)
        k_y = torch.cat((torch.arange(start=0, end=res // 2, step=1), torch.arange(start=-res // 2, end=0, step=1)),
                        0).reshape(1, res).repeat(res, 1)

        self.k_x = (torch.abs(k_x) * (torch.abs(k_x) < modes)).reshape(1, res, res, 1)
        self.k_y = (torch.abs(k_y) * (torch.abs(k_y) < modes)).reshape(1, res, res, 1)

    def cuda(self, device):
        self.k_x = self.k_x.to(device)
        self.k_y = self.k_y.to(device)

    def cpu(self):
        self.k_x = self.k_x.cpu()
        self.k_y = self.k_y.cpu()

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)
        return torch.sum(diff_norms / y_norms)

    def __call__(self, x, y, a=None, return_l2=True):
        x = torch.squeeze(x)
        y = torch.squeeze(y)

        l2loss = self.rel(x, y)
        x = torch.fft.fftn(x, dim=[1, 2], norm='ortho')
        y = torch.fft.fftn(y, dim=[1, 2], norm='ortho')

        x = x.view(x.shape[0], self.res, self.res, -1)
        y = y.view(y.shape[0], self.res, self.res, -1)

        weight = 1 + (2*math.pi) ** 2 * (self.k_x ** 2 + self.k_y ** 2)
        weight = torch.sqrt(weight).to(x.device)
        hloss = self.rel(x * weight, y * weight)
        return l2loss, hloss


class DH1Loss(_WeightedLoss):
    def __init__(self,
                 dim=2,
                 dilation=2,  # central diff
                 h=1 / 512,  # mesh size
                 eps=1e-10,
                 ):
        super(DH1Loss, self).__init__()
        assert dilation % 2 == 0
        self.dilation = dilation
        self.dim = dim
        self.h = h
        self.eps = eps

    def central_diff(self, u: torch.Tensor, h=None):
        '''
        u: function defined on a grid (bsz, n, n)
        out: gradient (N, n-2, n-2, 2)
        '''
        d = self.dilation  # central diff dilation
        s = d // 2  # central diff stride

        grad_x = (u[:, d:, s:-s] - u[:, :-d, s:-s]) / d
        grad_y = (u[:, s:-s, d:] - u[:, s:-s, :-d]) / d
        grad = torch.stack([grad_x, grad_y], dim=-1)
        return grad / h

    def forward(self, preds, targets, K=None):
        r'''
        preds: (N, n, n, 1)
        targets: (N, n, n, 1)
        K: (N, n, n, 1)
        '''
        if len(preds.shape) == 3:
            preds = preds.unsqueeze(-1)
        h = self.h
        d = self.dim

        target_norm = targets.pow(2).mean(dim=(1, 2)) + self.eps
        loss = ((preds - targets).pow(2)).mean(dim=(1, 2)) / target_norm
        l2loss = loss.sqrt().sum()

        preds_diff = self.central_diff(preds, h=h)
        targets_prime = self.central_diff(targets, h=h)

        targets_prime_norm = d * (targets_prime.pow(2)).mean(dim=(1, 2, 3)) + self.eps

        regularizer = (((preds - targets).pow(2)).mean(dim=(1, 2)) + d * ((targets_prime - preds_diff).pow(2)).mean(
            dim=(1, 2, 3))) / (targets_prime_norm + target_norm)

        h1loss = regularizer.sqrt().sum()
        return l2loss, h1loss


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
