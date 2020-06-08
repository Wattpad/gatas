import math
import os
from typing import NamedTuple, Tuple

import numba
import numpy as np
import tensorflow as tf

from framework.dataset import io


class NeighbourSample(NamedTuple):
    indices: tf.Tensor
    segments: tf.Tensor
    path_indices: tf.Tensor
    weights: tf.Tensor


class NeighbourSampler(tf.Module):
    def __init__(
            self,
            accumulated_transition_lengths: np.ndarray,
            neighbours: np.ndarray,
            path_indices: np.ndarray,
            probabilities: np.ndarray,
            num_edge_types: int,
            num_steps: int) -> None:

        super().__init__()

        self.accumulated_transition_lengths = accumulated_transition_lengths
        self.neighbours = neighbours
        self.path_indices = path_indices
        self.probabilities = probabilities

        self.num_edge_types = num_edge_types
        self.num_steps = num_steps

        self.path_depths = compute_path_depths(num_steps, num_edge_types)

        self.coefficients = tf.Variable(
            initial_value=self.compute_initial_coefficients(num_steps),
            shape=(num_steps + 1,),
            dtype=tf.float32,
        )

    @classmethod
    def get_num_edge_types(cls, path: str, suffix: str) -> int:
        edge_types = io.load_bin(os.path.join(path, f'edge_types{suffix}.bin'), dtype=np.int32)

        num_edge_types = np.max(edge_types) + 2

        return num_edge_types

    @classmethod
    def from_path(cls, num_steps: int, path: str, suffix: str = '') -> 'NeighbourSampler':
        accumulated_transition_lengths = io.load_npy(
            path=os.path.join(path, f'accumulated_transition_lengths{suffix}.npy'),
            mmap_mode='r',
        )
        neighbours = io.load_npy(os.path.join(path, f'neighbours{suffix}.npy'), mmap_mode='r')
        path_indices = io.load_npy(os.path.join(path, f'path_indices{suffix}.npy'), mmap_mode='r')
        probabilities = io.load_npy(os.path.join(path, f'probabilities{suffix}.npy'), mmap_mode='r')

        instance = cls(
            accumulated_transition_lengths=accumulated_transition_lengths,
            neighbours=neighbours,
            path_indices=path_indices,
            probabilities=probabilities,
            num_edge_types=cls.get_num_edge_types(path, suffix),
            num_steps=num_steps,
        )

        return instance

    @staticmethod
    def compute_initial_coefficients(num_steps: int) -> np.ndarray:
        steps = np.concatenate((np.array([0], dtype=np.float32), np.arange(num_steps, dtype=np.float32)))

        decaying_distribution = -steps / np.log(num_steps + 1, dtype=np.float32)

        return decaying_distribution

    def __call__(self, node_indices: tf.Tensor, sample_size: int, noisify: bool = True) -> NeighbourSample:
        step_probabilities = tf.nn.softmax(self.coefficients)

        indices, segments, path_indices, steps, probabilities = self.generate_sample(
            node_indices=node_indices,
            coefficients=step_probabilities,
            sample_size=sample_size,
            noisify=noisify,
        )

        weights = tf.gather(step_probabilities, steps) * probabilities

        sample = NeighbourSample(indices, segments, path_indices, weights)

        return sample

    def generate_sample(
            self,
            node_indices: tf.Tensor,
            coefficients: tf.Tensor,
            sample_size: int,
            noisify: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

        indices, segments, path_indices, steps, probabilities= tf.numpy_function(
            func=lambda x, y: generate_sample(
                node_indices=x,
                transition_pointers=self.accumulated_transition_lengths,
                neighbours=self.neighbours,
                path_indices=self.path_indices,
                probabilities=self.probabilities,
                coefficients=y,
                path_depths=self.path_depths,
                num_steps=self.num_steps,
                sample_size=sample_size,
                noisify=noisify,
            ),
            inp=(node_indices, coefficients),
            Tout=(tf.int32, tf.int32, tf.int32, tf.int32, tf.float32),
        )

        indices.set_shape((None,))
        segments.set_shape((None,))
        path_indices.set_shape((None,))
        steps.set_shape((None,))
        probabilities.set_shape((None,))

        return indices, segments, path_indices, steps, probabilities


@numba.njit
def generate_sample(
        node_indices: np.ndarray,
        transition_pointers: np.ndarray,
        neighbours: np.ndarray,
        path_indices: np.ndarray,
        probabilities: np.ndarray,
        coefficients: np.ndarray,
        path_depths: np.ndarray,
        num_steps: int,
        sample_size: int,
        noisify: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    indices = []
    segments = []
    path_indices_subset = []
    steps = []
    probabilities_subset = []

    for segment_id, node_index in enumerate(node_indices):
        start_index, end_index = transition_pointers[node_index], transition_pointers[node_index + 1]

        neighbour_sparse_indices, _ = compute_top_k(
            probabilities=probabilities[start_index:end_index],
            path_indices=path_indices[start_index:end_index],
            path_depths=path_depths,
            coefficients=coefficients,
            num_steps=num_steps,
            k=sample_size,
            noisify=noisify,
        )

        for neighbour_sparse_index in neighbour_sparse_indices:
            neighbour_index = neighbour_sparse_index + start_index

            indices.append(neighbours[neighbour_index])
            segments.append(segment_id)
            path_indices_subset.append(path_indices[neighbour_index])
            steps.append(path_depths[path_indices[neighbour_index]])
            probabilities_subset.append(probabilities[neighbour_index])

    indices = np.array(indices, dtype=np.int32)
    segments = np.array(segments, dtype=np.int32)
    path_indices_subset = np.array(path_indices_subset, dtype=np.int32)
    steps = np.array(steps, dtype=np.int32)
    probabilities_subset = np.array(probabilities_subset, dtype=np.float32)

    return indices, segments, path_indices_subset, steps, probabilities_subset


@numba.njit
def get_path_depth(index: int, num_edge_types: int, step: int = 0) -> int:
    if (index == 0 and step == 0) or index < 0:
        return step

    parent_index = math.floor(index / num_edge_types) - 1

    return get_path_depth(parent_index, num_edge_types, step + 1)


@numba.njit
def compute_path_depths(num_steps: int, num_edge_types: int) -> np.ndarray:
    num_paths = 0

    for step in range(num_steps):
        num_paths += num_edge_types ** (step + 1)

    path_depths = np.empty((num_paths,), dtype=np.int32)

    for path_index in range(num_paths):
        path_depths[path_index] = get_path_depth(path_index, num_edge_types)

    return path_depths


@numba.njit
def compute_top_k(
        probabilities: np.ndarray,
        path_indices: np.ndarray,
        path_depths: np.ndarray,
        coefficients: np.ndarray,
        num_steps: int,
        k: int,
        noisify: bool) -> Tuple[np.ndarray, np.ndarray]:

    top_indices = np.full(fill_value=-1, shape=(k,), dtype=np.int32)
    top_values = np.full(fill_value=-np.inf, shape=(k,), dtype=np.float32)

    if k == 0:
        return top_indices, top_values

    last_index = k - 1

    for index, probability in enumerate(probabilities):
        step = path_depths[path_indices[index]]

        if step > num_steps:
            continue

        value = calculate_transition_logits(probability, coefficients[step], noisify=noisify)

        if value <= top_values[0]:
            continue

        # heap bubble-down operation
        node_index = 0

        top_indices[node_index], top_values[node_index] = index, value

        while True:
            child_index = 2 * node_index + 1

            swap_index = node_index

            if child_index <= last_index and top_values[node_index] > top_values[child_index]:
                swap_index = child_index

            if child_index + 1 <= last_index and top_values[swap_index] > top_values[child_index + 1]:
                swap_index = child_index + 1

            if swap_index == node_index:
                break

            temp_index, temp_value = top_indices[swap_index], top_values[swap_index]
            top_indices[swap_index] = top_indices[node_index]
            top_values[swap_index] = top_values[node_index]
            top_indices[node_index], top_values[node_index] = temp_index, temp_value

            node_index = swap_index

    # extract indices from fixed-length heaps
    indices = np.where(top_indices >= 0)[0]
    top_indices, top_values = top_indices[indices], top_values[indices]

    return top_indices, top_values


@numba.njit
def calculate_transition_logits(
        probabilities: np.ndarray,
        coefficients: np.ndarray,
        noisify: bool,
        eps: float = 1e-20) -> float:

    logits = np.log(probabilities * coefficients)

    if noisify:
        gumbel_sample = -np.log(-np.log(np.random.random(np.shape(probabilities)) + eps) + eps)
        logits += gumbel_sample

    return logits
