import pickle
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from backpack import (
    backpack,
    extend,
    extensions,
)
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import (
    SGD,
    AdamW,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.optimizers.cgn import CGNOptimizer
from src.data import create_dataloaders
from src.models.linear import make_autoencoder
from src.utils import seed_everything, make_optimizers


def _train(
    _model: nn.Module,
    _optimizer,
    _criterion,
    _train_data_loader: DataLoader,
    _test_data_loader: DataLoader,
    _num_epochs: int,
    _device: torch.device,
    _out_path: Path,
):
    _out_path.mkdir(
        exist_ok=True,
        parents=True,
    )

    _train_loss = []
    _validation_loss = []
    _train_time = time.time()

    _model.train()

    for _epoch in tqdm(range(_num_epochs)):
        _epoch_loss = []

        for i, batch in enumerate(_train_data_loader):
            _optimizer.zero_grad()

            _out = model(batch)
            _loss = criterion(
                _out,
                batch,
            )

            if isinstance(_optimizer, CGNOptimizer):
                with backpack(_optimizer.bp_extension):
                    _loss.backward()
            else:
                _loss.backward()

            _optimizer.step()

            _epoch_loss.append(_loss.item())

        _epoch_loss = sum(_epoch_loss) / len(_epoch_loss)
        _val_loss = _eval(
            _model,
            _criterion,
            _test_data_loader,
        )

        print()
        print(
            f'epoch:      {_epoch + 1} / {_num_epochs}\n'
            f'train loss: {_epoch_loss:.2f}\n'
            f'val loss:   {_val_loss:.2f}\n',
        )

        _train_loss.append(_epoch_loss)
        _validation_loss.append(_val_loss)

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
    )


def _eval(
    _model: nn.Module,
    _criterion,
    _data_loader: DataLoader,
) -> float:
    _model.eval()
    _losses = []

    with torch.no_grad():
        for i, batch in enumerate(_data_loader):
            _out = model(batch)
            _loss = criterion(
                _out,
                batch,
            )
            _losses.append(_loss.item())

    _losses = sum(_losses) / len(_losses)
    _model.train()

    return _losses


if __name__ == '__main__':
    seed_everything(1337)

    parser = ArgumentParser()
    parser.add_argument(
        '-d',
        '--data',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '-o',
        '--out',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        required=True,
    )
    args = parser.parse_args()

    data = torch.load(args.data)
    train_dataloader, val_dataloader = create_dataloaders(
        data,
        test_size=0.1,
        batch_size=args.batch_size,
        seed=1337,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                ],
                'maxiter': [100, 500],
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
        for optim_type, params, name in make_optimizers(setup['cls'], **setup['params']):
            if setup['cls'] == CGNOptimizer:
                left = name.index(' object')
                right = name.index('>')
                real_name = name[:left] + name[right:]
            else:
                real_name = name

            cur_path = base_dir / real_name

            if cur_path.exists() and len(list(cur_path.iterdir())) > 0:
                print(f'optimizer {real_name} already exists, skipping...')
                continue

            model = make_autoencoder().to(device)
            optim = optim_type(model.parameters(), **params)
            criterion = nn.MSELoss()

            if setup['cls'] == CGNOptimizer:
                model = extend(model)
                criterion = extend(criterion)

            print(f'running optimizer {real_name}')

            cur_res = _train(
                model,
                optim,
                criterion,
                train_dataloader,
                val_dataloader,
                50,
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
