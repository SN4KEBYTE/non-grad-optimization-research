import random
from itertools import product
from typing import (
    Generator,
    Type,
)

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


def init_weights(
    module: nn.Module,
) -> None:
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(
            mean=0.0,
            std=0.02,
        )

        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(
            mean=0.0,
            std=0.5,
        )
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def make_optimizers(
    optim: Type[Optimizer],
    **kwargs,
) -> Generator:
    for k, v in kwargs.items():
        if not isinstance(v, list):
            kwargs[k] = [v]

    keys, values = zip(*kwargs.items())

    for comb in [dict(zip(keys, v)) for v in product(*values)]:
        yield (
            optim,
            comb,
            f'{optim.__name__} {", ".join(f"{k}={v}" for k, v in comb.items())}'
        )


def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
