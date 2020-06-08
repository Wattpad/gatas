import os
from typing import Tuple, Generator, Optional, Union

import numba
import numpy as np
from sklearn.model_selection import ShuffleSplit

from framework.dataset import io


BatchType = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class LinkPredictionDataset:
    def __init__(
            self,
            triplets: np.ndarray,
            targets: np.ndarray,
            num_classes: int,
            random_state: int,
            indices: Optional[np.ndarray] = None) -> None:

        self.triplets = triplets
        self.targets = targets
        self.num_classes = num_classes
        self.random_state = random_state

        self.indices = indices if indices is not None else np.arange(targets.size, dtype=np.int32)
        self.size = self.indices.size

    @classmethod
    def from_path(cls, path: str, num_classes: int, random_state: int) -> 'LinkPredictionDataset':
        triplets = io.load_npy(os.path.join(path, 'triplets.npy'), mmap_mode='r')
        targets = io.load_npy(os.path.join(path, 'targets.npy'), mmap_mode='r')

        return cls(triplets, targets, num_classes, random_state)

    def get_batches(self, size: int) -> Generator[BatchType, None, None]:
        np.random.shuffle(self.indices)

        for start_index in range(0, self.size, size):
            end_index = min(start_index + size, self.size)

            batch_indices = self.indices[start_index:end_index]

            yield self.get_batch(self.triplets[batch_indices], self.targets[batch_indices], self.num_classes)

    @staticmethod
    @numba.njit
    def get_batch(
            triplets: np.ndarray,
            targets: np.ndarray,
            num_classes: int) -> BatchType:

        size = targets.size

        node_ids = np.empty((size * 2,), dtype=np.int32)
        pair_indices = np.empty((size * 2, 2), dtype=np.int32)

        edge_targets = np.zeros((size, num_classes), dtype=np.float32)
        edge_targets_mask = np.zeros((size, num_classes), dtype=np.float32)

        for index in range(size):
            node_ids[index * 2] = triplets[index, 0]
            pair_indices[index * 2, :] = [index, 0]

            node_ids[index * 2 + 1] = triplets[index, 1]
            pair_indices[index * 2 + 1, :] = [index, 1]

            edge_targets[index, triplets[index, 2]] = targets[index]
            edge_targets_mask[index, triplets[index, 2]] = 1.

        return node_ids, pair_indices, edge_targets, edge_targets_mask

    def subset(self, indices: np.ndarray) -> 'LinkPredictionDataset':
        dataset = LinkPredictionDataset(
            triplets=self.triplets,
            targets=self.targets,
            num_classes=self.num_classes,
            random_state=self.random_state,
            indices=self.indices[indices],
        )

        return dataset


def get_splitted_dataset(
        path: str,
        num_classes: int,
        random_state: int) -> Tuple[LinkPredictionDataset, LinkPredictionDataset, LinkPredictionDataset]:

    train_dataset = LinkPredictionDataset(
        triplets=io.load_npy(os.path.join(path, 'triplets_train.npy')),
        targets=io.load_npy(os.path.join(path, 'targets_train.npy')),
        num_classes=num_classes,
        random_state=random_state,
    )

    validation_dataset = LinkPredictionDataset(
        triplets=io.load_npy(os.path.join(path, 'triplets_validation.npy')),
        targets=io.load_npy(os.path.join(path, 'targets_validation.npy')),
        num_classes=num_classes,
        random_state=random_state,
    )

    test_dataset = LinkPredictionDataset(
        triplets=io.load_npy(os.path.join(path, 'triplets_test.npy')),
        targets=io.load_npy(os.path.join(path, 'targets_test.npy')),
        num_classes=num_classes,
        random_state=random_state,
    )

    return train_dataset, validation_dataset, test_dataset


def get_dataset_splits(
        dataset: LinkPredictionDataset,
        train_size: Union[int, float],
        validation_size: Union[int, float],
        test_size: Union[int, float],
        number_splits: int,
        random_state: int) -> Generator[Tuple[LinkPredictionDataset, LinkPredictionDataset, LinkPredictionDataset], None, None]:

    if isinstance(train_size, float):
        train_size = round(dataset.size * train_size)

    if isinstance(validation_size, float):
        validation_size = round(dataset.size * validation_size)

    if isinstance(test_size, float):
        test_size = round(dataset.size * test_size)

    train_test_splitter = ShuffleSplit(
        n_splits=number_splits,
        train_size=train_size,
        test_size=validation_size + test_size,
        random_state=random_state,
    )

    validation_test_splitter = ShuffleSplit(
        n_splits=1,
        train_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )

    for train_indices, validation_test_indices in train_test_splitter.split(np.empty(dataset.size, dtype=np.bool)):
        train_dataset = dataset.subset(train_indices)

        validation_indices, test_indices = \
            next(validation_test_splitter.split(np.empty(validation_test_indices.size, dtype=np.bool)))

        validation_indices = validation_test_indices[validation_indices]
        test_indices = validation_test_indices[test_indices]

        validation_dataset = dataset.subset(validation_indices)
        test_dataset = dataset.subset(test_indices)

        yield train_dataset, validation_dataset, test_dataset
