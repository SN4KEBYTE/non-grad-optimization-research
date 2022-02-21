from typing import Type

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch.optim import Optimizer

from src.utils import init_weights


class Block(nn.Module):
    def __init__(
        self,
        in_shape: int,
        out_shape: int,
        dropout_proba: float = 0.1,
        act_fn: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(
            in_shape,
            in_shape,
        )
        self.bn = nn.LayerNorm(in_shape)
        self.act = act_fn()
        self.drop = nn.Dropout(dropout_proba)
        self.fc2 = nn.Linear(
            in_shape,
            out_shape,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        return x


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        in_shape: int,
        enc_shape: int,
        optim: Type[Optimizer],
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_shape, 256),
            Block(256, 128),
            Block(128, 64),
            Block(64, enc_shape),
        )
        self.decoder = nn.Sequential(
            Block(enc_shape, 64),
            Block(64, 128),
            Block(128, 256),
            nn.Linear(256, in_shape)
        )

        self._optim = optim
        self._args = args
        self._kwargs = kwargs

        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(x)

    def configure_optimizers(
        self,
    ) -> Optimizer:
        return self._optim(
            self.parameters(),
            *self._args,
            **self._kwargs,
        )

    def cosine_embedding_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        return F.cosine_embedding_loss(
            y_pred,
            y_true,
            torch.ones(y_pred.size(0)),
            reduction='mean',
        )

    def mse_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ):
        return F.mse_loss(
            y_pred,
            y_true,
            reduction='mean',
        )

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        out = self.decoder(self.encoder(batch))
        loss = self.mse_loss(
            out,
            batch,
        )

        self.log(
            'train_loss',
            loss.item(),
        )

        return loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        out = self.decoder(self.encoder(batch))
        loss = self.mse_loss(
            out,
            batch,
        )

        self.log(
            'val_loss',
            loss.item(),
        )
