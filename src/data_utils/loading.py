from typing import Tuple, Union

import numpy as np
import torchvision
from scipy import io
from sklearn import preprocessing
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, Normalize, ToTensor

from src.env_vars import DATA_DIR


################################################################################
################################################################################
# Synthetic datasets.
def generate_synthetic_classification_data(
    n: int, d: int, random_state: Union[None, int]
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=random_state)

    x = rng.standard_normal((n, d))

    y = np.sign(x[:, 0])
    # Arbitrarily assign zeros to one.
    y[y == 0] = 1

    return x, y


def generate_synthetic_regression_data(
    n: int, d: int, random_state: Union[None, int]
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=random_state)
    x = rng.standard_normal((n, d))
    y = x[:, 0]
    return x, y


################################################################################
################################################################################
# Image datasets.
def load_image_dataset(dataset_name: str) -> Tuple[Dataset, Dataset, int, int]:
    if dataset_name == "RestrictedMNIST":
        target_transform = lambda y: 0 if y == 2 else 1  # Convert to binary target.
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )
        train_dataset = torchvision.datasets.MNIST(
            root=DATA_DIR,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=True,
        )
        train_dataset = Subset(
            train_dataset,
            np.arange(len(train_dataset))[
                (train_dataset.targets == 2) | (train_dataset.targets == 3)
            ],
        )
        test_dataset = torchvision.datasets.MNIST(
            root=DATA_DIR,
            train=False,
            transform=transform,
            target_transform=target_transform,
            download=True,
        )
        test_dataset = Subset(
            test_dataset,
            np.arange(len(test_dataset))[
                (test_dataset.targets == 2) | (test_dataset.targets == 3)
            ],
        )
        d = test_dataset[0][0].shape[-1]
        num_classes = 2
    else:
        raise ValueError("dataset unknown.")

    return train_dataset, test_dataset, d, num_classes


def load_tabular_dataset(
    dataset: str,
    normalize_data: bool = True,
    normalize_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if dataset == "leukemia":
        data = io.loadmat(DATA_DIR / "leukemia.mat")
        x, y = data["X"], data["Y"].astype(int)
    else:
        raise ValueError(f"Dataset {dataset} unknown.")

    if normalize_data:
        # Normalize each feature across the dataset.
        x = preprocessing.normalize(x, axis=0)
    if normalize_labels:
        y = y - y.mean()
    return x, y
