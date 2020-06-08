import os
from typing import List, Tuple, MutableMapping

import defopt
import numba
import numpy as np
from numba import types as nb_types

from framework.dataset import io


TransitionType = Tuple[int, int, int]


@numba.experimental.jitclass([
    ('stack_primary', nb_types.List(nb_types.Tuple([numba.int64, numba.int64, numba.int64]))),
    ('stack_secondary', nb_types.List(nb_types.Tuple([numba.int64, numba.int64, numba.int64]))),
])
class Queue:
    def __init__(self, element: TransitionType) -> None:
        self.stack_primary = [element]
        self.stack_secondary: List[TransitionType] = [self.stack_primary.pop()]

    def size(self) -> int:
        return len(self.stack_primary) + len(self.stack_secondary)

    def enqueue(self, element: TransitionType) -> None:
        self.stack_primary.append(element)

    def dequeue(self) -> TransitionType:
        if len(self.stack_secondary) == 0:
            while len(self.stack_primary) > 0:
                self.stack_secondary.append(self.stack_primary.pop())

        return self.stack_secondary.pop()


@numba.njit
def extend_path(path_index: int, edge_type: int, num_edge_types: int) -> int:
    if path_index == 0:
        return edge_type + 1

    return (path_index + 1) * (num_edge_types + 1) + edge_type + 1


@numba.njit
def traverse(
        start_index: int,
        accumulated_num_edges: np.ndarray,
        adjacencies: np.ndarray,
        edge_types: np.ndarray,
        num_steps: int,
        num_edge_types: int) -> Tuple[MutableMapping[Tuple[int, int, int], int], List[int]]:

    neighbour_path_counts: MutableMapping[Tuple[int, int, int], int] = {}
    step_counts: List[int] = [0] * (num_steps + 1)

    visited: MutableMapping[int, int] = {}
    queue = Queue((start_index, 0, 0))

    while queue.size() > 0:
        key = queue.dequeue()
        node_index, path_index, step = key

        if node_index in visited and visited[node_index] < step:
            continue

        visited[node_index] = step

        if key in neighbour_path_counts:
            neighbour_path_counts[key] += 1
        else:
            neighbour_path_counts[key] = 1

        step_counts[step] += 1

        if step >= num_steps:
            continue

        for sparse_index in range(accumulated_num_edges[node_index], accumulated_num_edges[node_index + 1]):
            neighbour_id, edge_type = adjacencies[sparse_index], edge_types[sparse_index]
            extended_path_id = extend_path(path_index, edge_type, num_edge_types)

            queue.enqueue((neighbour_id, extended_path_id, step + 1))

    return neighbour_path_counts, step_counts


@numba.njit
def create_transition_tensors(
        accumulated_num_edges: np.ndarray,
        adjacencies: np.ndarray,
        edge_types: np.ndarray,
        num_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    num_edge_types = np.max(edge_types) + 1

    accumulated_transition_lengths = [0]
    neighbours = []
    path_indices = []
    probabilities = []

    for start_index in range(accumulated_num_edges.size - 1):
        neighbour_path_counts, step_counts = traverse(
            start_index=start_index,
            accumulated_num_edges=accumulated_num_edges,
            adjacencies=adjacencies,
            edge_types=edge_types,
            num_steps=num_steps,
            num_edge_types=num_edge_types,
        )

        step_probability = [1 / num_steps if num_steps != 0 else 0. for num_steps in step_counts]
        accumulated_transition_lengths.append(accumulated_transition_lengths[-1] + len(neighbour_path_counts))

        for (end_index, path_index, step), count in neighbour_path_counts.items():
            neighbours.append(end_index)
            path_indices.append(path_index)
            probabilities.append(count * step_probability[step])

    accumulated_transition_lengths = np.array(accumulated_transition_lengths, dtype=np.int32)
    neighbours = np.array(neighbours, dtype=np.int32)
    path_indices = np.array(path_indices, dtype=np.int32)
    probabilities = np.array(probabilities, dtype=np.float32)

    return accumulated_transition_lengths, neighbours, path_indices, probabilities


def store_transition_tensor(*, path: str, num_steps: int, suffix: str = '') -> None:
    """
    Computes and stores a transition tensor.

    :param path: The path to the CSR sparse representation vectors.
    :param num_steps: Number of steps to compute.
    :param suffix: Suffix of the input files.
    """
    accumulated_num_edges = io.load_bin(os.path.join(path, f'accumulated_num_edges{suffix}.bin'), dtype=np.int32)
    adjacencies = io.load_bin(os.path.join(path, f'adjacencies{suffix}.bin'), dtype=np.int32)
    edge_types = io.load_bin(os.path.join(path, f'edge_types{suffix}.bin'), dtype=np.int32)

    accumulated_transition_lengths, neighbours, path_indices, probabilities = create_transition_tensors(
        accumulated_num_edges=accumulated_num_edges,
        adjacencies=adjacencies,
        edge_types=edge_types,
        num_steps=num_steps,
    )

    np.save(os.path.join(path, f'accumulated_transition_lengths{suffix}.npy'), accumulated_transition_lengths)
    np.save(os.path.join(path, f'neighbours{suffix}.npy'), neighbours)
    np.save(os.path.join(path, f'path_indices{suffix}.npy'), path_indices)
    np.save(os.path.join(path, f'probabilities{suffix}.npy'), probabilities)


if __name__ == '__main__':
    defopt.run(store_transition_tensor)
