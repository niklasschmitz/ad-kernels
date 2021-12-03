"""
This module contains dataclass abstractions that are useful for dispatching to more
efficient implementations of kernelmatrix accumulation (see solve.py). Their use is
not strictly needed, but recommended for performance.
"""
from typing import Callable, Sequence
from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class DescriptorKernel:
    descriptor: Callable
    kappa: Callable
    def __call__(self, x1, x2, *, descriptor_kwargs={}, **kwargs):
        d1 = self.descriptor(x1, **descriptor_kwargs)
        d2 = self.descriptor(x2, **descriptor_kwargs)
        return self.kappa(d1, d2, **kwargs)

@dataclass(eq=True, frozen=True)
class KernelSum:
    kernels: Sequence[Callable]
    def __call__(self, x1, x2, **kwargs):
        return sum(kernel(x1, x2, **kwargs) for kernel in self.kernels)

