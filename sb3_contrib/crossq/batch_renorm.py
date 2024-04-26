# from: https://github.com/ludvb/batchrenorm/blob/master/batchrenorm/batchrenorm.py

import torch


__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]


import torch
import torch.nn as nn

class CrossNorm(nn.Module):
    def __init__(self, num_features, alpha=.5, beta=.5, scaling=True, eps: float = 1e-3, momentum=0.01):
        super(CrossNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.mode = 'batchnorm'

        self.eps = eps

        self.alpha = alpha
        self.beta = beta

        self.scaling = scaling

        self.r_max = 3
        self.d_max = 5
        self.correction = 1
        
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.data.zero_()
        self.bias.data.zero_()
        self.running_var.data.fill_(1.)
        self.weight.data.fill_(1.)
        # self.weight.data.uniform_(0, 1.) # as pytorch does it

    def set_mode(self, mode, r_max=1, d_max=0):
        self.mode = mode
        self.r_max = r_max
        self.d_max = d_max

    def forward(self, inp):
        if self.training:
            bs = int(inp.size(0)/2)
            phi = inp[:bs]
            phi_ = inp[bs:]

            
            avg = (torch.mean(phi,0) * self.alpha + torch.mean(phi_, 0) * self.beta)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * avg.data
            if self.scaling:
                self.correction = 1
                var = torch.var(inp, 0)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
                std = (var + self.eps).sqrt()
                running_std = (self.running_var * self.correction + self.eps).sqrt()

            r = 1
            d = 0
            # if self.mode == 'renorm':
            if self.num_batches_tracked > 100_000:
                if self.scaling:
                    r = (std / running_std).detach()
                    r = torch.clamp(r, 1/self.r_max, self.r_max)
                    d = ((avg - self.running_mean) / running_std ).detach()
                else:
                    d = (avg - self.running_mean).detach()

                d = torch.clamp(d, -self.d_max, self.d_max)

            if self.scaling:
                output = ((inp - avg) / std) * r + d
            else:
                output = inp - avg + d

            self.num_batches_tracked += 1

        else:
            avg = self.running_mean
            if self.scaling:
                var = self.running_var
                output = (inp - avg) / (var * self.correction + self.eps).sqrt()
            else:
                output = inp - avg


        if self.scaling:
            output = output * self.weight + self.bias
        else:
            output = output + self.bias

        return output

    def switch_to_renorm(self, momentum=0.01):
        self.momentum = momentum
        self.set_mode('renorm', 3, 5)



class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_std", torch.ones(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self):
        return 3.0

    @property
    def dmax(self):
        return 5.0

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        '''
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        '''
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            if mask is not None:
                z = x[~mask]
                batch_mean = z.mean(0) 
                batch_std = z.std(0, unbiased=False) + self.eps
            else:
                batch_mean = x.mean(dims)
                batch_std = x.std(dims, unbiased=False) + self.eps

            r = (
                batch_std.detach() / self.running_std.view_as(batch_std)
            ).clamp_(1 / self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean))
                / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax, self.dmax)

            # x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (
                batch_mean.detach() - self.running_mean
            )
            self.running_std += self.momentum * (
                batch_std.detach() - self.running_std
            )
            self.num_batches_tracked += 1

            if self.num_batches_tracked > 100_000:
                s = batch_std / r
                m = batch_mean - d * batch_std / r
                x = (x - m) / (s + self.eps)
            else:
                x = (x - batch_mean) / batch_std
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")