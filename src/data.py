from typing import (
    List,
    Tuple,
)

import torch
from torch.utils.data import (
    DataLoader,
    random_split,
    TensorDataset,
)


def train_test_split(
    data: torch.Tensor,
    test_size: float,
    seed: int,
) -> Tuple[TensorDataset, TensorDataset]:
    dataset_size = len(data)
    train_samples = int((1 - test_size) * dataset_size)
    test_samples = dataset_size - train_samples

    train_indices, test_indices = random_split(
        list(range(dataset_size)),
        [train_samples, test_samples],
        generator=torch.Generator().manual_seed(seed)
    )

    return (
        data[train_indices],
        data[test_indices],
    )


def create_dataloaders(
    data: torch.Tensor,
    test_size: float,
    batch_size: int,
    seed: int,
) -> List[DataLoader]:
    data_loaders = []
    splitted = train_test_split(
        data,
        test_size,
        seed,
    )

    for i in range(len(splitted)):
        cur = DataLoader(
            dataset=splitted[i],
            batch_size=batch_size,
            num_workers=8,
            shuffle=i == 0,
            drop_last=True,
        )

        data_loaders.append(cur)

    return data_loaders
