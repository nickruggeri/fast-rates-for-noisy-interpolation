import math
import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import neural_tangents as nt
import numpy as np
import torch
import torchvision
from neural_tangents import stax
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, Normalize, ToTensor

from src.data_utils.preprocessing import FixedBinaryLabelNoiseDataset
from src.env_vars import DATA_DIR


def get_data(args):
    # Convert to binary target.
    target_transform = lambda y: -1 if y in [0, 2, 4] else 1

    transform = Compose(
        [
            ToTensor(),
            Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    train_set = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    train_set = Subset(
        train_set,
        np.arange(len(train_set))[
            (train_set.targets == 0)
            | (train_set.targets == 1)
            | (train_set.targets == 2)
            | (train_set.targets == 3)
            | (train_set.targets == 4)
            | (train_set.targets == 5)
        ],
    )

    test_set = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=False,
        transform=transform,
        target_transform=target_transform,
        download=True,
    )
    test_set = Subset(
        test_set,
        np.arange(len(test_set))[
            (test_set.targets == 0)
            | (test_set.targets == 1)
            | (test_set.targets == 2)
            | (test_set.targets == 3)
            | (test_set.targets == 4)
            | (test_set.targets == 5)
        ],
    )

    # Subsample and add label noise.
    if args.training_samples > 0:
        all_data = torch.utils.data.ConcatDataset([train_set, test_set])

        train_idx = np.random.choice(
            len(all_data),
            size=args.training_samples,
            replace=False,
        )
        test_idx = list(set(range(len(all_data))) - set(train_idx))

        train_set = torch.utils.data.Subset(all_data, train_idx)
        test_set = torch.utils.data.Subset(all_data, test_idx)

    if args.label_noise:
        n_flips = math.ceil(len(train_set) * args.label_noise)
        swap_idx = np.random.choice(len(train_set), n_flips, replace=False)
        swap_mask = np.zeros(len(train_set))
        swap_mask[swap_idx] = 1
        train_set = FixedBinaryLabelNoiseDataset(train_set, swap_mask, mode="+-1")

    train_data = [train_set[i] for i in range(len(train_set))]
    train_x, train_y = np.array([el[0].numpy() for el in train_data]), np.array(
        [el[1] for el in train_data]
    )
    train_y = train_y.reshape(-1, 1)

    test_data = [test_set[i] for i in range(len(test_set))]
    test_x, test_y = np.array([el[0].numpy() for el in test_data]), np.array(
        [el[1] for el in test_data]
    )
    test_y = test_y.reshape(-1, 1)

    return train_x, train_y, test_x, test_y


def create_model(depth, k, width=2048):
    layers = []
    for i in range(depth):
        layers.append(stax.Conv(width, (k, k), (1, 1), "SAME"))
        layers.append(stax.Relu())
    layers.append(stax.Flatten())
    layers.append(stax.Dense(1))

    return stax.serial(*layers)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=Path, default=Path(".") / "out" / "cntk")
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--label_noise", type=float, default=0.0)
    parser.add_argument("--training_samples", type=int, default=500)
    args = parser.parse_args()

    train_x, train_y, test_x, test_y = get_data(args)
    init_fn, apply_fn, kernel_fn = create_model(args.depth, args.kernel_size)

    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_x, train_y)

    y_train_pred = predict_fn(x_test=train_x, get="ntk")
    train_acc = np.mean(train_y == np.sign(y_train_pred))
    assert train_acc == 1

    y_test_pred = predict_fn(x_test=test_x, get="ntk")
    test_acc = np.mean(test_y == np.sign(y_test_pred))

    args.save_dir.mkdir(exist_ok=True, parents=True)
    save_file = args.save_dir / "res.pkl"
    with open(save_file, "wb") as file:
        pkl.dump(test_acc, file)
