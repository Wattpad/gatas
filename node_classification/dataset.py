import math
import os
from typing import Tuple, Generator, Optional

import numba
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from framework.dataset import io


@numba.experimental.jitclass([
    ('num_nodes', numba.int32),
    ('node_indices', numba.int32[:]),
    ('class_indices', numba.int32[:, :]),
    # ('class_indices', numba.int32[:]),
    ('random_state', numba.int32),
])
class NodeClassifierDataset:
    def __init__(self, node_indices: np.ndarray, class_indices: np.ndarray, random_state: int) -> None:
        self.num_nodes = node_indices.size

        self.node_indices = node_indices
        self.class_indices = class_indices

        self.random_state = random_state

    def get_batches(self, size: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        indices = np.arange(self.num_nodes)

        np.random.seed(self.random_state)

        np.random.shuffle(indices)

        for start_index in range(0, self.num_nodes, size):
            end_index = min(start_index + size, self.num_nodes)

            batch_indices = indices[start_index:end_index]

            yield self.node_indices[batch_indices], self.class_indices[batch_indices]

    def get_subset(self, indices: np.ndarray) -> 'NodeClassifierDataset':
        dataset = NodeClassifierDataset(self.node_indices[indices], self.class_indices[indices], self.random_state)

        return dataset


def get_dataset_splits(
        path: str,
        train_ratio: float,
        validation_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        random_state: int = 110069) -> Tuple[NodeClassifierDataset, NodeClassifierDataset, NodeClassifierDataset]:

    class_ids = io.load_npy(os.path.join(path, 'class_ids.npy'), mmap_mode='r')
    num_nodes = np.size(class_ids)

    indices = np.arange(num_nodes, dtype=np.int32)
    np.random.RandomState(random_state).shuffle(indices)

    train_split = math.ceil(num_nodes * train_ratio)
    train_indices = indices[:train_split]
    train_dataset = NodeClassifierDataset(train_indices, class_ids[train_indices], random_state)

    validation_split = train_split + math.ceil(num_nodes * validation_ratio) if validation_ratio else num_nodes
    validation_indices = indices[train_split:validation_split]
    validation_dataset = NodeClassifierDataset(validation_indices, class_ids[validation_indices], random_state)

    if test_ratio is None:
        return train_dataset, validation_dataset, validation_dataset

    test_split = validation_split + math.ceil(num_nodes * test_ratio)
    test_indices = indices[validation_split:test_split]
    test_dataset = NodeClassifierDataset(test_indices, class_ids[test_indices], random_state)

    return train_dataset, validation_dataset, test_dataset


def get_stratified_dataset_splits(
        path: str,
        train_size: int,
        validation_size: int,
        test_size: Optional[int] = None,
        number_splits: int = 10,
        random_state: int = 110069) -> Generator[Tuple[NodeClassifierDataset, NodeClassifierDataset, NodeClassifierDataset], None, None]:

    train_test_splitter = StratifiedShuffleSplit(
        n_splits=number_splits,
        train_size=train_size,
        test_size=validation_size + (test_size if test_size else 0),
        random_state=random_state)

    validation_test_splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=validation_size,
        test_size=(test_size if test_size else 0),
        random_state=random_state)

    class_ids = io.load_npy(os.path.join(path, 'class_ids.npy'), mmap_mode='r')

    for train_indices, validation_test_indices in train_test_splitter.split(np.zeros_like(class_ids), class_ids):
        train_dataset = NodeClassifierDataset(train_indices.astype(np.int32), class_ids[train_indices], random_state)

        if test_size is None:
            validation_dataset = NodeClassifierDataset(validation_test_indices.astype(np.int32), class_ids[validation_test_indices], random_state)

            yield train_dataset, validation_dataset, validation_dataset

        validation_test_class_ids = class_ids[validation_test_indices]
        validation_indices, test_indices = next(validation_test_splitter.split(np.zeros_like(validation_test_class_ids), validation_test_class_ids))
        validation_indices = validation_test_indices[validation_indices]
        test_indices = validation_test_indices[test_indices]

        validation_dataset = NodeClassifierDataset(validation_indices.astype(np.int32), class_ids[validation_indices], random_state)
        test_dataset = NodeClassifierDataset(test_indices.astype(np.int32), class_ids[test_indices], random_state)

        yield train_dataset, validation_dataset, test_dataset


def get_splitted_dataset(path: str, random_state: int = 110069) -> Tuple[NodeClassifierDataset, NodeClassifierDataset, NodeClassifierDataset]:
    class_ids = io.load_npy(os.path.join(path, 'class_ids.npy'), mmap_mode='r')

    train_indices = io.load_npy(os.path.join(path, 'train_indices.npy'))
    train_dataset = NodeClassifierDataset(train_indices, class_ids[train_indices], random_state)

    validation_indices = io.load_npy(os.path.join(path, 'validation_indices.npy'))
    validation_dataset = NodeClassifierDataset(validation_indices, class_ids[validation_indices], random_state)

    test_indices = io.load_npy(os.path.join(path, 'test_indices.npy'))
    test_dataset = NodeClassifierDataset(test_indices, class_ids[test_indices], random_state)

    return train_dataset, validation_dataset, test_dataset
