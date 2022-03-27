""" Activations
A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.
Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


def mish(x, inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    NOTE: I don't have a working inplace variant
    """
    return x.mul(F.softplus(x).tanh())


class Mish(nn.Module):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    def __init__(self, inplace: bool = False):
        super(Mish, self).__init__()

    def forward(self, x):
        return mish(x)


def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()


# PyTorch has this, but not with a consistent inplace argmument interface
class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


def tanh(x, inplace: bool = False):
    return x.tanh_() if inplace else x.tanh()


# PyTorch has this, but not with a consistent inplace argmument interface
class Tanh(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x, inplace: bool = False):
    inner = F.relu6(x + 3.).div_(6.)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, self.inplace)


def hard_mish(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    if inplace:
        return x.mul_(0.5 * (x + 2).clamp(min=0, max=2))
    else:
        return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardMish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_mish(x, self.inplace)


class PReLU(nn.PReLU):
    """Applies PReLU (w/ dummy inplace arg)
    """
    def __init__(self, num_parameters: int = 1, init: float = 0.25, inplace: bool = False) -> None:
        super(PReLU, self).__init__(num_parameters=num_parameters, init=init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.prelu(input, self.weight)


def gelu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return F.gelu(x)


class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function (w/ dummy inplace arg)
    """
    def __init__(self, inplace: bool = False):
        super(GELU, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input)


""" Activations
A collection of jit-scripted activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.
All jit scripted activations are lacking in-place variations on purpose, scripted kernel fusion does not
currently work across in-place op boundaries, thus performance is equal to or less than the non-scripted
versions if they contain in-place ops.
Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


@torch.jit.script
def swish_jit(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul(x.sigmoid())


@torch.jit.script
def mish_jit(x, _inplace: bool = False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())


class SwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishJit, self).__init__()

    def forward(self, x):
        return swish_jit(x)


class MishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(MishJit, self).__init__()

    def forward(self, x):
        return mish_jit(x)


@torch.jit.script
def hard_sigmoid_jit(x, inplace: bool = False):
    # return F.relu6(x + 3.) / 6.
    return (x + 3).clamp(min=0, max=6).div(6.)  # clamp seems ever so slightly faster?


class HardSigmoidJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoidJit, self).__init__()

    def forward(self, x):
        return hard_sigmoid_jit(x)


@torch.jit.script
def hard_swish_jit(x, inplace: bool = False):
    # return x * (F.relu6(x + 3.) / 6)
    return x * (x + 3).clamp(min=0, max=6).div(6.)  # clamp seems ever so slightly faster?


class HardSwishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwishJit, self).__init__()

    def forward(self, x):
        return hard_swish_jit(x)


@torch.jit.script
def hard_mish_jit(x, inplace: bool = False):
    """ Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


class HardMishJit(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardMishJit, self).__init__()

    def forward(self, x):
        return hard_mish_jit(x)

""" Activations (memory-efficient w/ custom autograd)
A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.
These activations are not compatible with jit scripting or ONNX export of the model, please use either
the JIT or basic versions of the activations.
Hacked together by / Copyright 2020 Ross Wightman
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish w/ memory-efficient checkpoint
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """
    @staticmethod
    def symbolic(g, x):
        return g.op("Mul", x, g.op("Sigmoid", x))

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


def swish_me(x, inplace=False):
    return SwishJitAutoFn.apply(x)


class SwishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(SwishMe, self).__init__()

    def forward(self, x):
        return SwishJitAutoFn.apply(x)


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp * x_tanh_sp))


class MishJitAutoFn(torch.autograd.Function):
    """ Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    A memory efficient, jit scripted variant of Mish
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


def mish_me(x, inplace=False):
    return MishJitAutoFn.apply(x)


class MishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(MishMe, self).__init__()

    def forward(self, x):
        return MishJitAutoFn.apply(x)


@torch.jit.script
def hard_sigmoid_jit_fwd(x, inplace: bool = False):
    return (x + 3).clamp(min=0, max=6).div(6.)


@torch.jit.script
def hard_sigmoid_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * ((x >= -3.) & (x <= 3.)) / 6.
    return grad_output * m


class HardSigmoidJitAutoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_sigmoid_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_sigmoid_jit_bwd(x, grad_output)


def hard_sigmoid_me(x, inplace: bool = False):
    return HardSigmoidJitAutoFn.apply(x)


class HardSigmoidMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSigmoidMe, self).__init__()

    def forward(self, x):
        return HardSigmoidJitAutoFn.apply(x)


@torch.jit.script
def hard_swish_jit_fwd(x):
    return x * (x + 3).clamp(min=0, max=6).div(6.)


@torch.jit.script
def hard_swish_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= 3.)
    m = torch.where((x >= -3.) & (x <= 3.),  x / 3. + .5, m)
    return grad_output * m


class HardSwishJitAutoFn(torch.autograd.Function):
    """A memory efficient, jit-scripted HardSwish activation"""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_swish_jit_bwd(x, grad_output)

    @staticmethod
    def symbolic(g, self):
        input = g.op("Add", self, g.op('Constant', value_t=torch.tensor(3, dtype=torch.float)))
        hardtanh_ = g.op("Clip", input, g.op('Constant', value_t=torch.tensor(0, dtype=torch.float)), g.op('Constant', value_t=torch.tensor(6, dtype=torch.float)))
        hardtanh_ = g.op("Div", hardtanh_, g.op('Constant', value_t=torch.tensor(6, dtype=torch.float)))
        return g.op("Mul", self, hardtanh_)


def hard_swish_me(x, inplace=False):
    return HardSwishJitAutoFn.apply(x)


class HardSwishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwishMe, self).__init__()

    def forward(self, x):
        return HardSwishJitAutoFn.apply(x)


@torch.jit.script
def hard_mish_jit_fwd(x):
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


@torch.jit.script
def hard_mish_jit_bwd(x, grad_output):
    m = torch.ones_like(x) * (x >= -2.)
    m = torch.where((x >= -2.) & (x <= 0.), x + 1., m)
    return grad_output * m


class HardMishJitAutoFn(torch.autograd.Function):
    """ A memory efficient, jit scripted variant of Hard Mish
    Experimental, based on notes by Mish author Diganta Misra at
      https://github.com/digantamisra98/H-Mish/blob/0da20d4bc58e696b6803f2523c58d3c8a82782d0/README.md
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return hard_mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return hard_mish_jit_bwd(x, grad_output)


def hard_mish_me(x, inplace: bool = False):
    return HardMishJitAutoFn.apply(x)


class HardMishMe(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardMishMe, self).__init__()

    def forward(self, x):
        return HardMishJitAutoFn.apply(x)
