import pickle
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import (
    List,
    Tuple,
)

import matplotlib.pyplot as plt
import torch
from backpack import (
    backpack,
    extend,
    extensions,
)
from torch import nn
from torch.optim import (
    SGD,
    AdamW,
)
from torch.utils.data import DataLoader
from torchmetrics import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
)
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.models.cnn import make_image_classifier
from src.optimizers.cgn import CGNOptimizer
from src.utils import (
    make_optimizers,
    seed_everything,
)


def _train(
    _model: nn.Module,
    _optimizer,
    _criterion,
    _train_data_loader: DataLoader,
    _test_data_loader: DataLoader,
    _num_epochs: int,
    _device: torch.device,
    _out_path: Path,
) -> Tuple[List[int], List[float], float, List[float], List[float], List[float], List[float]]:
    _out_path.mkdir(
        exist_ok=True,
        parents=True,
    )

    _train_loss = []
    _validation_loss = []
    _train_time = time.time()
    _accuracy = []
    _f1_score = []
    _precision = []
    _recall = []

    _model.train()

    for _epoch in tqdm(range(_num_epochs)):
        _epoch_loss = []

        for i, (images, labels) in enumerate(_train_data_loader):
            images = images.to(_device)
            labels = labels.to(_device)

            _optimizer.zero_grad()

            output = _model(images)
            _loss = _criterion(
                output,
                labels,
            )

            if isinstance(_optimizer, CGNOptimizer):
                with backpack(_optimizer.bp_extension):
                    _loss.backward()
            else:
                _loss.backward()

            _optimizer.step()
            _epoch_loss.append(_loss.item())

        _val_loss, _acc, _f1, _prec, _rec = _eval(
            _model,
            _criterion,
            _test_data_loader,
        )
        _accuracy.append(_acc)
        _f1_score.append(_f1)
        _precision.append(_prec)
        _recall.append(_rec)
        _epoch_loss = sum(_epoch_loss) / len(_epoch_loss)

        print()
        print(
            f'epoch:      {_epoch + 1} / {_num_epochs}\n'
            f'train loss: {_epoch_loss:.2f}\n'
            f'val loss:   {_val_loss:.2f}\n'
            f'accuracy:   {_acc:.2f}\n'
            f'f1:         {_f1:.2f}\n'
            f'precision:  {_prec:.2f}\n'
            f'recall:     {_rec:.2f}\n'
        )

        _validation_loss.append(_val_loss)
        _train_loss.append(_epoch_loss)

    _train_time = time.time() - _train_time

    torch.save(
        _model.state_dict(),
        _out_path / 'model.pt',
    )
    torch.save(
        _optimizer.state_dict(),
        _out_path / 'optim.pt',
    )

    plt.plot(
        _train_loss,
        color='blue',
        label='train loss',
    )
    plt.plot(
        _validation_loss,
        color='green',
        label='val loss',
    )
    plt.legend(loc='upper right')
    plt.savefig(
        _out_path / 'loss.png',
        dpi=400,
        bbox_inches='tight',
    )
    plt.close()

    return (
        _train_loss,
        _validation_loss,
        _train_time,
        _accuracy,
        _f1_score,
        _precision,
        _recall,
    )


def _eval(
    _model: nn.Module,
    _criterion,
    _data_loader: DataLoader,
) -> Tuple[float, float, float, float, float]:
    _model.eval()

    _acc = Accuracy()
    _f1 = F1Score()
    _prec = Precision()
    _rec = Recall()
    _loss = []

    with torch.no_grad():
        for _images, _labels in _data_loader:
            _images = _images.to(device)
            _labels = _labels.to(device)

            _out = _model(_images)
            _loss.append(criterion(_out, _labels).item())
            _pred = torch.max(_out, 1)[1].data.squeeze()

            _acc(
                _pred,
                _labels,
            )
            _f1(
                _pred,
                _labels,
            )
            _prec(
                _pred,
                _labels,
            )
            _rec(
                _pred,
                _labels,
            )

    _loss = sum(_loss) / len(_loss)
    _model.train()

    return (
        _loss,
        _acc.compute().item(),
        _f1.compute().item(),
        _prec.compute().item(),
        _rec.compute().item(),
    )


if __name__ == '__main__':
    seed_everything(1337)

    parser = ArgumentParser()
    parser.add_argument(
        '-m',
        '--mnist',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-o',
        '--out',
        type=Path,
        required=True,
    )

    args = parser.parse_args()
    mnist_root = args.mnist

    train_data = datasets.MNIST(
        root=mnist_root,
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_data = datasets.MNIST(
        root=mnist_root,
        train=False,
        transform=ToTensor()
    )

    loaders = {
        'train': DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            num_workers=8,
        ),
        'test': DataLoader(
            test_data,
            batch_size=32,
            shuffle=False,
            num_workers=8,
        ),
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device {device}')

    optimizers = {
        'sgd': {
            'cls': SGD,
            'params': {
                'lr': [0.001, 0.01, 0.1],
            }
        },
        'adamw': {
            'cls': AdamW,
            'params': {
                'amsgrad': [True],
                'lr': [0.001, 0.01, 0.1],
            }
        },
        'cgn': {
            'cls': CGNOptimizer,
            'params': {
                'bp_extension': [
                    extensions.GGNMP(),
                    extensions.HMP(),
                    extensions.PCHMP(modify="abs"),
                    extensions.PCHMP(modify="clip"),
                ],
                'maxiter': 1000,
                'lr': [0.001, 0.01, 0.1],
            }
        },
    }

    base_dir = args.out
    base_dir.mkdir(
        exist_ok=True,
        parents=True,
    )
    train_results = []

    for setup in optimizers.values():
        model = make_image_classifier(10).to(device)
        criterion = nn.CrossEntropyLoss()

        for optim, name in make_optimizers(setup['cls'], list(model.parameters()), **setup['params']):
            cur_path = base_dir / name

            if cur_path.exists() and len(list(cur_path.iterdir())) > 0:
                print(f'optimizer {name} already exists, skipping...')
                continue

            if setup['cls'] == CGNOptimizer:
                model = extend(model)
                criterion = extend(criterion)

            cur_res = _train(
                model,
                optim,
                criterion,
                loaders['train'],
                loaders['test'],
                25,
                device,
                cur_path,
            )
            cur_res = [name] + list(cur_res)
            train_results.append(cur_res)

            with open(cur_path / 'res.pkl', 'wb') as f:
                pickle.dump(
                    cur_res,
                    f,
                )

    with open(base_dir / 'image_classifier.pkl', 'wb') as f:
        pickle.dump(
            train_results,
            f,
        )
