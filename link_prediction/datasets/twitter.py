import io
import os
import random
from functools import reduce
from typing import Generator, Tuple, Mapping, Set, Optional, Iterable, List, MutableMapping

import defopt
import numpy as np


FILES = [
    'higgs-mention_network.edgelist',
    'higgs-reply_network.edgelist',
    'higgs-retweet_network.edgelist',
    'higgs-social_network.edgelist',
]


def read_triplet_file(path: str) -> Generator[Tuple[int, int], None, None]:
    with io.open(path, mode='r') as text_file:
        for line in text_file:
            columns = line.split()

            yield int(columns[0]), int(columns[1])


def get_triplets(path: str) -> Generator[Tuple[int, int, int], None, None]:
    for edge_type, file in enumerate(FILES):
        for head, tail in read_triplet_file(os.path.join(path, file)):
            yield head, tail, edge_type


def get_mapped_triplets(path: str, node_map: Mapping[int, int]) -> Generator[Tuple[int, int, int], None, None]:
    for head, tail, edge_type in get_triplets(path):
        if head in node_map and tail in node_map:
            yield node_map[head], node_map[tail], edge_type


def get_graph_by_node_map(path: str, node_map: Mapping[int, int]) -> List[MutableMapping[int, Set[int]]]:
    num_nodes = len(node_map)

    graph: List[MutableMapping[int, Set[int]]] = [{} for _ in range(num_nodes)]

    for head, tail, edge_type in get_mapped_triplets(path, node_map):
        graph[head].setdefault(tail, set()).add(edge_type)

    return graph


def get_graph_by_triplets(triplets: np.ndarray, num_nodes: int, num_edge_types: int) -> List[MutableMapping[int, Set[int]]]:
    graph: List[MutableMapping[int, Set[int]]] = [{} for _ in range(num_nodes)]

    for head, tail, edge_type in triplets:
        graph[head].setdefault(tail, set()).add(edge_type)
        graph[tail].setdefault(head, set()).add(edge_type + num_edge_types)

    return graph


def get_largest_subgraph(graph: List[MutableMapping[int, Set[int]]]) -> Set[int]:
    num_nodes = len(graph)
    visited: Set[int] = set()
    subgraphs = []

    while len(visited) < num_nodes:
        subgraph = set()

        stack = [next(
            node_index
            for node_index in range(num_nodes)
            if node_index not in visited
        )]

        while stack:
            node = stack.pop()

            if node not in subgraph:
                subgraph.add(node)
                visited.add(node)

                for neighbour in graph[node].keys():
                    stack.append(neighbour)

        subgraphs.append(subgraph)

    sorted_subgraphs = sorted(
        [(subgraph, len(subgraph)) for subgraph in subgraphs],
        key=lambda graph_tuple: graph_tuple[1],
        reverse=True,
    )

    return sorted_subgraphs[0][0]


def compute_node_maps(node_ids: Iterable[int]) -> Tuple[Mapping[int, int], Mapping[int, int]]:
    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    index_to_id = {index: node_id for index, node_id in enumerate(node_ids)}

    return id_to_index, index_to_id


def get_biggest_nodes(graph: List[Mapping[int, Set[int]]], subgraph: Set[int], num_nodes: int) -> List[int]:
    node_lengths = [
        (node_index, sum(len(edge_types) for edge_types in graph[node_index].values()))
        for node_index in subgraph
    ]

    node_lengths_sorted = sorted(node_lengths, key=lambda length: length[1], reverse=True)

    node_indices = [node_index for node_index, _ in node_lengths_sorted[:num_nodes]]

    return node_indices


def create_node_ids(path: str, num_nodes: Optional[int] = None) -> Mapping[int, int]:
    node_ids = set()

    for head, tail, _ in get_triplets(path):
        node_ids.add(head)
        node_ids.add(tail)

    node_id_to_index, node_index_to_id = compute_node_maps(node_ids)

    if num_nodes is not None:
        graph = get_graph_by_node_map(path, node_id_to_index)

        largest_subgraph = get_largest_subgraph(graph)

        # node_indices = get_biggest_nodes(graph, largest_subgraph, num_nodes)
        node_indices = random.sample(largest_subgraph, num_nodes)

        node_ids = {node_index_to_id[node_index] for node_index in node_indices}

    np.save(os.path.join(path, 'node_ids.npy'), np.array(node_ids, dtype=np.int32))

    return compute_node_maps(node_ids)[0]


def generate_sample(num_nodes: int, num_edge_types: int) -> Tuple[int, int, int]:
    head = random.randint(0, num_nodes - 1)
    tail = random.randint(0, num_nodes - 1)
    edge_type = random.randint(0, num_edge_types - 1)

    return head, tail, edge_type


def generate_negative_dataset(path: str, node_map: Mapping[int, int], num_edge_types: int) -> Generator[Tuple[int, int, int], None, None]:
    num_nodes = len(node_map)

    positive_samples, negative_samples = set(get_mapped_triplets(path, node_map)), set()

    while True:
        triplet = generate_sample(num_nodes, num_edge_types)

        if triplet not in positive_samples and triplet not in negative_samples:
            negative_samples.add(triplet)

            yield triplet


def create_triplets(path: str, node_map: Mapping[int, int], num_edge_types: int) -> Tuple[np.ndarray, np.ndarray]:
    positive_samples = set(get_mapped_triplets(path, node_map))
    negative_samples = generate_negative_dataset(path, node_map, num_edge_types)

    dataset_size = len(positive_samples) * 2

    triplets = np.empty((dataset_size, 3), dtype=np.int32)
    targets = np.empty((dataset_size,), dtype=np.int32)

    for index, (positive_triplet, negative_triplet) in enumerate(zip(positive_samples, negative_samples)):
        triplets[index * 2, :] = positive_triplet
        targets[index * 2] = 1

        triplets[index * 2 + 1, :] = negative_triplet
        targets[index * 2 + 1] = 0

    return triplets, targets


def split_triplets(
        triplets: np.ndarray,
        targets: np.ndarray,
        train_size: float,
        validation_size: float,
        test_size: float,
        path: str) -> np.ndarray:

    size = triplets.shape[0]

    indices = np.arange(size)
    np.random.shuffle(indices)

    train_size = round(train_size * size)
    validation_size = round(validation_size * size)
    test_size = round(test_size * size)

    train_indices = indices[:train_size]
    np.save(os.path.join(path, 'triplets_train.npy'), triplets[train_indices])
    np.save(os.path.join(path, 'targets_train.npy'), targets[train_indices])

    validation_indices = indices[train_size:train_size + validation_size]
    np.save(os.path.join(path, 'triplets_validation.npy'), triplets[validation_indices])
    np.save(os.path.join(path, 'targets_validation.npy'), targets[validation_indices])

    test_indices = indices[train_size + validation_size:train_size + validation_size + test_size]
    np.save(os.path.join(path, 'triplets_test.npy'), triplets[test_indices])
    np.save(os.path.join(path, 'targets_test.npy'), targets[test_indices])

    triplets_train_positive = triplets[train_indices][targets[train_indices].astype(np.bool)]

    return triplets_train_positive


def split_triplets_variation(
        triplets: np.ndarray,
        targets: np.ndarray,
        base_size: float,
        train_size: float,
        validation_size: float,
        test_size: float,
        path: str) -> np.ndarray:

    size = triplets.shape[0]

    indices = np.arange(size)
    np.random.shuffle(indices)

    base_size = round(base_size * size)
    train_size = round(train_size * size)
    validation_size = round(validation_size * size)
    test_size = round(test_size * size)

    base_indices = indices[:base_size]

    train_indices = indices[base_size:base_size + train_size]
    np.save(os.path.join(path, 'triplets_train.npy'), triplets[train_indices])
    np.save(os.path.join(path, 'targets_train.npy'), targets[train_indices])

    validation_indices = indices[base_size + train_size:base_size + train_size + validation_size]
    np.save(os.path.join(path, 'triplets_validation.npy'), triplets[validation_indices])
    np.save(os.path.join(path, 'targets_validation.npy'), targets[validation_indices])

    test_indices = indices[base_size + train_size + validation_size:base_size + train_size + validation_size + test_size]
    np.save(os.path.join(path, 'triplets_test.npy'), triplets[test_indices])
    np.save(os.path.join(path, 'targets_test.npy'), targets[test_indices])

    triplets_base_positive = triplets[base_indices][targets[base_indices].astype(np.bool)]

    return triplets_base_positive


def create_graph(graph: List[MutableMapping[int, Set[int]]], path: str) -> None:
    accumulated_num_edges, adjacencies, edge_types = [0], [], []

    for tails in graph:
        edges = sorted(
            (tail, edge_type)
            for tail, pair_edge_types in tails.items()
            for edge_type in pair_edge_types
        )

        for tail, edge_type in edges:
            adjacencies.append(tail)
            edge_types.append(edge_type)

        accumulated_num_edges.append(accumulated_num_edges[-1] + len(edges))

    np.array(accumulated_num_edges, dtype=np.int32).tofile(os.path.join(path, 'accumulated_num_edges.bin'))
    np.array(adjacencies, dtype=np.int32).tofile(os.path.join(path, 'adjacencies.bin'))
    np.array(edge_types, dtype=np.int32).tofile(os.path.join(path, 'edge_types.bin'))


def count_edge_types(
        graph: List[Mapping[int, Set[int]]],
        node_ids: Iterable[int],
        num_edge_types: int) -> List[List[int]]:

    node_counts = []

    for head in node_ids:
        edge_type_count = [0] * num_edge_types

        for edge_types in graph[head].values():
            for edge_type in edge_types:
                edge_type_count[edge_type] += 1

        node_counts.append(edge_type_count)

    return node_counts


def select_nodes(
        graph: List[Mapping[int, Set[int]]],
        node_ids: List[int],
        num_nodes: int,
        num_edge_types: int) -> List[int]:

    edge_types_counts = count_edge_types(graph, node_ids, num_edge_types)

    num_nodes_per_edge_type = round(num_nodes / num_edge_types)

    node_indices_subset: Set[int] = set()

    for edge_type in range(num_edge_types):
        edge_type_counts = sorted(
            (
                (node_index, counts[edge_type])
                for node_index, counts
                in enumerate(edge_types_counts)
                if node_index not in node_indices_subset
            ),
            key=lambda node_counts: node_counts[1],
            reverse=True,
        )

        node_indices_subset.update(node_index for node_index, _ in edge_type_counts[:num_nodes_per_edge_type])

    node_ids_subset = [node_ids[node_index] for node_index in node_indices_subset]

    return node_ids_subset


def compute_overlap_node_ids(path: str, num_edge_types: int, num_nodes: Optional[int] = None) -> Mapping[int, int]:
    sets: MutableMapping[int, Set[int]] = {}

    for head, tail, edge_type in get_triplets(path):
        if edge_type not in sets:
            sets[edge_type] = set()

        sets[edge_type].add(head)
        sets[edge_type].add(tail)

    node_ids = reduce(lambda x, y: x & y, sets.values())

    node_id_to_index, node_index_to_id = compute_node_maps(node_ids)

    if num_nodes is not None:
        graph = get_graph_by_node_map(path, node_id_to_index)

        largest_subgraph = get_largest_subgraph(graph)

        node_indices = random.sample(largest_subgraph, num_nodes)
        # node_indices = get_biggest_nodes(graph, largest_subgraph, num_nodes)
        # node_indices = select_nodes(graph, list(largest_subgraph), num_nodes, num_edge_types)

        node_ids = {node_index_to_id[node_index] for node_index in node_indices}

    np.save(os.path.join(path, 'node_ids.npy'), np.array(node_ids, dtype=np.int32))

    return compute_node_maps(node_ids)[0]


def main(
        *,
        path: str,
        train_size: float = .85,
        validation_size: float = .05,
        test_size: float = .1,
        num_nodes: Optional[int] = None) -> None:

    num_edge_types = len(FILES)

    node_map = compute_overlap_node_ids(path, num_edge_types, num_nodes)
    triplets, targets = create_triplets(path, node_map, num_edge_types)
    triplets_train_positive = split_triplets(triplets, targets, train_size, validation_size, test_size, path)
    graph = get_graph_by_triplets(triplets_train_positive, len(node_map), num_edge_types)
    create_graph(graph, path)


if __name__ == '__main__':
    defopt.run(main)
