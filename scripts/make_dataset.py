from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from navec import Navec
from tqdm import tqdm


if __name__ == '__main__':
    cwd = Path(__file__).parents[1]
    models_dir = cwd / 'models'
    models_dir.mkdir(exist_ok=True)
    data_dir = cwd / 'data'
    data_dir.mkdir(exist_ok=True)
    navec_path = models_dir / 'navec.tar'

    if not navec_path.exists():
        urlretrieve(
            'https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar',
            navec_path,
        )

    nvc = Navec.load(navec_path)
    vectors = []

    for word in tqdm(sorted(nvc.vocab.words)):
        if word not in {'<unk>', '<pad>'}:
            vectors.append(nvc[word])

    vectors = np.asarray(
        vectors,
        dtype=vectors[0].dtype,
    )
    vectors = torch.as_tensor(
        vectors,
        dtype=torch.float32,
        device=torch.device('cpu'),
    )

    torch.save(
        vectors,
        data_dir / 'dataset.pt',
    )
