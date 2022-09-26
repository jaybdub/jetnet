from pydantic import BaseModel
import torch.optim
from typing import Tuple



class TorchOptimizer(BaseModel):

    def build(self, params):
        raise NotImplementedError

        
class TorchOptimizerSGD(TorchOptimizer):
    lr: float
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    def build(self, params):
        return torch.optim.SGD(
            params=params,
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov
        )


class TorchOptimizerAdadelta(TorchOptimizer):
    lr: float = 1.0
    rho: float = 0.9
    eps: float = 1e-6
    weight_decay: float = 0.0

    def build(self, params):
        return torch.optim.Adadelta(
            params,
            lr=self.lr,
            rho=self.rho,
            eps=self.eps,
            weight_decay=self.weight_decay
        )


class TorchOptimizerAdagrad(TorchOptimizer):
    lr: float = 0.01
    lr_decay: float = 0.0
    weight_decay: float = 0.0
    initial_accumulator_value: float = 0.0
    eps: float = 1e-10

    def build(self, params):
        return torch.optim.Adagrad(
            params,
            lr=self.lr,
            lr_decay=self.lr_decay,
            weight_decay=self.weight_decay,
            initial_accumulator_value=self.initial_accumulator_value,
            eps=self.eps
        )


class TorchOptimizerAdam(TorchOptimizer):
    lr: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False

    def build(self, params):
        return torch.optim.Adam(
            params=params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad
        )
