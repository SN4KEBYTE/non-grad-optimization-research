from typing import Tuple

from torch import nn


def make_autoencoder(
    in_shape: int = 300,
    enc_shape: int = 2,
    blocks: Tuple[int] = (256, 128, 64),
) -> nn.Module:
    blocks = list(blocks) + [enc_shape]

    encoder = nn.Sequential(
        nn.Linear(
            in_shape,
            blocks[0],
        ),
        *[
            nn.Sequential(
                nn.Linear(
                    blocks[i],
                    blocks[i],
                ),
                nn.BatchNorm1d(blocks[i]),
                nn.Tanh(),
                nn.Dropout(p=0.1),
                nn.Linear(
                    blocks[i],
                    blocks[i + 1],
                )
            )
            for i in range(len(blocks) - 1)
        ]
    )
    decoder = nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(
                    blocks[i],
                    blocks[i],
                ),
                nn.BatchNorm1d(blocks[i]),
                nn.Tanh(),
                nn.Dropout(p=0.1),
                nn.Linear(
                    blocks[i],
                    blocks[i - 1],
                )
            )
            for i in range(len(blocks) - 1, 0, -1)
        ],
        nn.Linear(
            blocks[0],
            in_shape,
        )
    )

    autoencoder = nn.Sequential(
        encoder,
        decoder,
    )

    return autoencoder
