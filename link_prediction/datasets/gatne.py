import io
import os
from typing import Generator, Tuple, Mapping, Callable, Iterator, Set

import defopt
import numpy as np

from link_prediction.datasets import twitter


def read_triplet_file(path: str) -> Generator[Tuple[int, int, int], None, None]:
    with io.open(path, mode='r') as text_file:
        for line in text_file:
            columns = line.strip().split()

            yield int(columns[1]), int(columns[2]), int(columns[0]) - 1


def read_triplet_file_with_targets(path: str) -> Generator[Tuple[int, int, int, int], None, None]:
    with io.open(path, mode='r') as text_file:
        for line in text_file:
            columns = line.strip().split()

            yield int(columns[1]), int(columns[2]), int(columns[0]) - 1, int(columns[3])


def get_node_map(path: str) -> Mapping[int, int]:
    node_ids = set()

    for head, tail, _ in read_triplet_file(os.path.join(path, 'train.txt')):
        node_ids.add(head)
        node_ids.add(tail)

    for dataset_type in ['validation', 'test']:
        for head, tail, _, _ in read_triplet_file_with_targets(os.path.join(path, f'{dataset_type}.txt')):
            node_ids.add(head)
            node_ids.add(tail)

    return twitter.compute_node_maps(node_ids)[0]


def read_mapped_triplets(node_map: Mapping[int, int], path: str) -> Generator[Tuple[int, int, int], None, None]:
    for head, tail, edge_type in read_triplet_file(path):
        yield node_map[head], node_map[tail], edge_type


def read_mapped_triplets_with_targets(node_map: Mapping[int, int], path: str) -> Generator[Tuple[int, int, int, int], None, None]:
    for head, tail, edge_type, target in read_triplet_file_with_targets(path):
        yield node_map[head], node_map[tail], edge_type, target


def get_all_samples(node_map: Mapping[int, int], path: str) -> Set[Tuple[int, int, int]]:
    train = set(read_mapped_triplets(node_map, os.path.join(path, 'train.txt')))

    validation = set((
        (head, tail, edge_type)
        for head, tail, edge_type, target
        in read_mapped_triplets_with_targets(node_map, os.path.join(path, 'validation.txt'))
    ))

    test = set((
        (head, tail, edge_type)
        for head, tail, edge_type, target
        in read_mapped_triplets_with_targets(node_map, os.path.join(path, 'test.txt'))
    ))

    return train | validation | test


def generate_negative_dataset(
        node_map: Mapping[int, int],
        num_edge_types: int,
        path: str) -> Generator[Tuple[int, int, int], None, None]:

    all_samples = get_all_samples(node_map, path)
    negative_samples = set()

    while True:
        triplet = twitter.generate_sample(len(node_map), num_edge_types)

        if triplet not in all_samples and triplet not in negative_samples:
            negative_samples.add(triplet)

            yield triplet


def generate_triplets(
        positive_samples: Set[Tuple[int, int, int]],
        node_map: Mapping[int, int],
        num_edge_types: int,
        path: str) -> Generator[Tuple[int, int, int, int], None, None]:

    negative_samples = generate_negative_dataset(node_map, num_edge_types, path)

    for index, (positive_triplet, negative_triplet) in enumerate(zip(positive_samples, negative_samples)):
        yield positive_triplet[0], positive_triplet[1], positive_triplet[2], 1
        yield negative_triplet[0], negative_triplet[1], negative_triplet[2], 0


def convert_triplets(generator: Callable[[], Iterator[Tuple[int, int, int]]]) -> np.ndarray:
    size = sum(1 for _ in generator())

    triplets = np.empty((size, 3), dtype=np.int32)

    indices = np.arange(size)
    np.random.shuffle(indices)

    for index, (head, tail, edge_type) in zip(indices, generator()):
        triplets[index] = [head, tail, edge_type]

    return triplets


def convert_triplets_with_targets(generator: Callable[[], Iterator[Tuple[int, int, int, int]]]) -> Tuple[np.ndarray, np.ndarray]:
    size = sum(1 for _ in generator())

    triplets = np.empty((size, 3), dtype=np.int32)
    targets = np.empty((size,), dtype=np.int32)

    indices = np.arange(size)
    np.random.shuffle(indices)

    for index, (head, tail, edge_type, target) in zip(indices, generator()):
        triplets[index] = [head, tail, edge_type]
        targets[index] = target

    return triplets, targets


def save_triplets(node_map: Mapping[int, int], num_edge_types: int, path: str) -> None:
    triplets, targets = convert_triplets_with_targets(lambda: generate_triplets(
        positive_samples=set(read_mapped_triplets(node_map, os.path.join(path, 'train.txt'))),
        node_map=node_map,
        num_edge_types=num_edge_types,
        path=path,
    ))
    np.save(os.path.join(path, 'triplets_train.npy'), triplets)
    np.save(os.path.join(path, 'targets_train.npy'), targets)

    triplets, targets = convert_triplets_with_targets(lambda: read_mapped_triplets_with_targets(node_map, os.path.join(path, 'validation.txt')))
    np.save(os.path.join(path, 'triplets_validation.npy'), triplets)
    np.save(os.path.join(path, 'targets_validation.npy'), targets)

    triplets, targets = convert_triplets_with_targets(lambda: read_mapped_triplets_with_targets(node_map, os.path.join(path, 'test.txt')))
    np.save(os.path.join(path, 'triplets_test.npy'), triplets)
    np.save(os.path.join(path, 'targets_test.npy'), targets)


def get_triplets_train_positive(node_map: Mapping[int, int], path: str) -> np.ndarray:
    return convert_triplets(lambda: read_mapped_triplets(node_map, os.path.join(path, 'train.txt')))


def main(*, path: str, num_edge_types: int) -> None:
    node_map = get_node_map(path)
    save_triplets(node_map, num_edge_types, path)
    triplets_train_positive = get_triplets_train_positive(node_map, path)
    graph = twitter.get_graph_by_triplets(triplets_train_positive, len(node_map), num_edge_types)
    twitter.create_graph(graph, path)


if __name__ == '__main__':
    defopt.run(main)
