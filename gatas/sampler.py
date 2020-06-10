import math
import os
from typing import NamedTuple, Tuple, Union

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
            accumulated_transition_lengths: Union[np.ndarray, tf.Tensor],
            neighbours: Union[np.ndarray, tf.Tensor],
            path_indices: Union[np.ndarray, tf.Tensor],
            probabilities: Union[np.ndarray, tf.Tensor],
            num_edge_types: int,
            num_steps: int) -> None:

        super().__init__()

        self.accumulated_transition_lengths = tf.convert_to_tensor(accumulated_transition_lengths)
        self.neighbours = tf.convert_to_tensor(neighbours)
        self.path_indices = tf.convert_to_tensor(path_indices)
        self.probabilities = tf.convert_to_tensor(probabilities)

        self.num_edge_types = num_edge_types
        self.num_steps = num_steps

        self.path_depths = tf.convert_to_tensor(compute_path_depths(num_steps, num_edge_types))

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
        accumulated_transition_lengths = io.load_npy(os.path.join(path, f'accumulated_transition_lengths{suffix}.npy'))
        neighbours = io.load_npy(os.path.join(path, f'neighbours{suffix}.npy'))
        path_indices = io.load_npy(os.path.join(path, f'path_indices{suffix}.npy'))
        probabilities = io.load_npy(os.path.join(path, f'probabilities{suffix}.npy'))

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

    @staticmethod
    def get_sample(
            logits: tf.Tensor,
            vector_indices: tf.Tensor,
            matrix_indices: tf.Tensor,
            lengths: tf.Tensor,
            sample_size: int) -> Tuple[tf.Tensor, tf.Tensor]:

        batch_size = tf.size(lengths)
        max_length = tf.reduce_max(lengths)
        k = tf.math.minimum(max_length, sample_size)

        mask = tf.sequence_mask(tf.math.minimum(lengths, k))

        logits_scattered = tf.fill((batch_size, max_length), tf.math.log(0.))
        logits_scattered = tf.tensor_scatter_nd_update(logits_scattered, matrix_indices, logits)

        top_k_indices = tf.math.top_k(logits_scattered, k)[1]
        top_k_indices += tf.cumsum(lengths, exclusive=True)[:, tf.newaxis]
        top_k_indices = tf.boolean_mask(top_k_indices, mask)

        sample_indices = tf.gather(vector_indices, top_k_indices)

        return sample_indices, top_k_indices

    def __call__(self, node_indices: tf.Tensor, sample_size: int, noisify: bool = True) -> NeighbourSample:
        coefficients = tf.nn.softmax(self.coefficients)

        vector_indices, matrix_indices, lengths = self.filter_transitions(*self.get_segments(node_indices))

        logits = self.calculate_transition_logits(coefficients, vector_indices, noisify=noisify)
        sample_indices, vector_indices_subset = self.get_sample(
            logits=logits,
            vector_indices=vector_indices,
            matrix_indices=matrix_indices,
            lengths=lengths,
            sample_size=sample_size,
        )

        segments = tf.gather(matrix_indices[:, 0], vector_indices_subset)
        weights = tf.gather(self.probabilities, sample_indices) * self.gather_with_steps(coefficients, sample_indices)

        sample = NeighbourSample(
            indices=tf.gather(self.neighbours, sample_indices),
            segments=segments,
            path_indices=tf.gather(self.path_indices, sample_indices),
            weights=weights,
        )

        return sample

    def get_segments(self, indices: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        batch_size = tf.size(indices)

        lengths = self.accumulated_transition_lengths[1:] - self.accumulated_transition_lengths[:-1]
        lengths = tf.gather(lengths, indices)

        max_length = tf.reduce_max(lengths)

        offsets = tf.gather(self.accumulated_transition_lengths, indices)
        mask = tf.sequence_mask(lengths)

        outer_indices = tf.broadcast_to(tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis], (batch_size, max_length))
        inner_indices = tf.broadcast_to(tf.range(max_length, dtype=tf.int32), (batch_size, max_length))

        vector_indices = offsets[:, tf.newaxis] + inner_indices
        vector_indices = tf.boolean_mask(vector_indices, mask)

        matrix_indices = tf.stack((tf.boolean_mask(outer_indices, mask), tf.boolean_mask(inner_indices, mask)), axis=1)

        return vector_indices, matrix_indices, lengths

    def filter_transitions(
            self,
            vector_indices: tf.Tensor,
            matrix_indices: tf.Tensor,
            lengths: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        batch_size, max_path_index = tf.size(lengths), tf.size(self.path_depths)

        subset_mask = tf.gather(self.path_indices, vector_indices) < max_path_index

        vector_indices_subset = tf.boolean_mask(vector_indices, subset_mask)

        lengths_subset = tf.math.unsorted_segment_sum(
            data=tf.ones_like(vector_indices_subset, dtype=lengths.dtype),
            segment_ids=tf.boolean_mask(matrix_indices[:, 0], subset_mask),
            num_segments=batch_size,
        )

        max_length, sequence_mask = tf.reduce_max(lengths_subset), tf.sequence_mask(lengths_subset)

        outer_indices = tf.broadcast_to(tf.range(batch_size, dtype=tf.int32)[:, tf.newaxis], (batch_size, max_length))
        inner_indices = tf.broadcast_to(tf.range(max_length, dtype=tf.int32), (batch_size, max_length))
        matrix_indices_subset = tf.stack(
            values=(tf.boolean_mask(outer_indices, sequence_mask), tf.boolean_mask(inner_indices, sequence_mask)),
            axis=1,
        )

        return vector_indices_subset, matrix_indices_subset, lengths_subset

    def gather_with_steps(self, coefficients: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
        path_indices = tf.gather(self.path_indices, indices)
        path_depths = tf.gather(self.path_depths, path_indices)
        coefficients = tf.gather(coefficients, path_depths)

        return coefficients

    def calculate_transition_logits(
            self,
            coefficients: tf.Tensor,
            indices: tf.Tensor,
            noisify: bool,
            eps: float = 1e-20) -> tf.Tensor:

        probabilities = tf.gather(self.probabilities, indices)
        coefficients = self.gather_with_steps(coefficients, indices)

        joint_probabilities = tf.math.log(probabilities * coefficients)

        if noisify:
            joint_probabilities -= tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(probabilities)) + eps) + eps)

        return joint_probabilities


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
