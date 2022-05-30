import math
import re
from typing import Iterator, Tuple, Union

import numpy as np
from sklearn.model_selection import KFold, ShuffleSplit
from torch.utils.data import Dataset

from src.data_utils.loading import (
    generate_synthetic_classification_data,
    generate_synthetic_regression_data,
    load_tabular_dataset,
)


################################################################################
################################################################################
# Data modifiers.
def add_regression_noise(
    y: np.ndarray, noise: float, random_state: Union[None, int]
) -> np.ndarray:
    if noise == 0:
        return y

    rng = np.random.default_rng(seed=random_state)
    return y + rng.normal(loc=0.0, scale=noise, size=len(y))


def add_binary_label_noise(
    y: np.ndarray, label_noise: float, random_state: Union[None, int], mode="exact"
) -> np.ndarray:
    if label_noise != 0:
        y = y.copy()
        rng = np.random.default_rng(seed=random_state)
        if mode == "sample":
            # Choose a number of flips drawn from a Bernoulli with probability label_noise.
            mask = rng.choice(
                a=[False, True], size=len(y), p=[1 - label_noise, label_noise]
            )
            y[mask] = -y[mask]
        elif mode == "exact":
            # Flip exactly a proportion of labels given by label_noise.
            n_flips = math.ceil(len(y) * label_noise)
            flip_idx = rng.choice(len(y), n_flips, replace=False)
            y[flip_idx] = -y[flip_idx]
        else:
            raise ValueError("mode unknown.")
    return y


class FixedBinaryLabelNoiseDataset(Dataset):
    """Wrapper of a dataset with binary 0-1 labels.
    It behaves like the input dataset, expect that it returns swapped labels for specified indices."""

    def __init__(self, dataset: Dataset, swap_mask: np.ndarray, mode: str = "01"):
        super().__init__()

        self.original_dataset = dataset
        self.swap_mask = swap_mask

        assert mode in ["01", "+-1"], "mode unknown"
        self.mode = mode

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, index: int):
        x, y = self.original_dataset[index]

        if self.swap_mask[index]:
            if self.mode == "01":
                # swap 1 and 0
                y = int(not y)
            elif self.mode == "+-1":
                # swap +1 and -1
                y = -y

        return x, y


################################################################################
################################################################################
# Data splitting.
def train_val_splits_classification_tabular(
    dataset: str,
    label_noise: float,
    normalize_data: bool,
    n_splits: int,
    random_split_size: float,
    random_state: Union[None, int],
) -> Iterator[Tuple[np.ndarray]]:
    """Yields train-validation splits of the required dataset. Only works on the tabular datasets for classification.
    If random_split_size is different from zero, then random splits are performed where random_split_size specifies the
    relative size in (0, 1) of the validation set and n_splits the number of repeated random splits to be performed.
    Otherwise, n_splits specifies the value k for k-fold splitting.
    For synthetic data, n_splits and random_split_size are not utilized.
    Random state is passed for determining the train/test splits and random label flips for all data, as well as
    generating synthetic data.
    """
    if not dataset.startswith("synthetic"):
        assert (
            n_splits or random_split_size
        ), "Specify a way to perform dataset splitting, n_splits and random_splits can't both be 0."

    if dataset.startswith("synthetic"):
        # Generate a training set with the given n and d, generate a big validation set with same d.
        parser = re.compile(r"synthetic_n=(?P<n>\d+)_d=(?P<d>\d+)")
        parsed = parser.match(dataset)
        n, d = int(parsed["n"]), int(parsed["d"])

        train_x, train_y = generate_synthetic_classification_data(
            n, d, random_state + 12 if random_state is not None else None
        )
        train_y = add_binary_label_noise(
            train_y,
            label_noise,
            random_state + 34 if random_state is not None else None,
            mode="exact",
        )

        val_x, val_y = generate_synthetic_classification_data(
            1000 * n, d, random_state + 56 if random_state is not None else None
        )
        yield train_x, train_y, val_x, val_y

    else:
        # Load real data and iterate over train-validation splits.
        x, y = load_tabular_dataset(dataset, normalize_data, normalize_labels=False)

        splitter = (
            ShuffleSplit(
                n_splits, test_size=random_split_size, random_state=random_state
            )
            if random_split_size
            else KFold(n_splits, random_state=random_state)
        )
        for i, (train_idx, val_idx) in enumerate(splitter.split(x, y)):
            x_train, y_train = x[train_idx, :], y[train_idx]
            x_val, y_val = x[val_idx, :], y[val_idx]

            y_train = add_binary_label_noise(
                y_train,
                label_noise,
                random_state + i if random_state is not None else None,
                mode="exact",
            )

            yield x_train, y_train, x_val, y_val


def train_val_splits_regression_tabular(
    dataset: str,
    noise: float,
    random_state: Union[None, int],
) -> Iterator[Tuple[np.ndarray]]:
    """Yields train-validation splits of the required dataset. Only works on the tabular datasets for regression.
    Random state is passed for generating the synthetic data.
    """

    if dataset.startswith("synthetic"):
        # Generate a training set with the given n and d, generate a big validation set with same d.
        parser = re.compile(r"synthetic_n=(?P<n>\d+)_d=(?P<d>\d+)")
        parsed = parser.match(dataset)
        assert (
            parsed is not None
        ), f"the dataset {dataset} is not recognised. Use something of form synthetic_n=x_d=x for some x values."
        n, d = int(parsed["n"]), int(parsed["d"])

        train_x, train_y = generate_synthetic_regression_data(
            n, d, random_state + 12 if random_state is not None else None
        )
        train_y = add_regression_noise(
            train_y, noise, random_state + 34 if random_state is not None else None
        )

        val_x, val_y = generate_synthetic_regression_data(
            1000 * n, d, random_state + 56 if random_state is not None else None
        )
        yield train_x, train_y, val_x, val_y
    else:
        raise ValueError(f"Dataset {dataset} unknown.")
