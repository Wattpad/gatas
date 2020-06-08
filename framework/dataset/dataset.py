import numba
import numpy as np


@numba.njit()
def convert_incremental_lengths_to_indices(accumulated_lengths: np.ndarray) -> np.ndarray:
    """
    Convert a vector of accumulated item lengths into a matrix of indices.
    For instance, given [0, 2, 5], we obtain [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]].
    :param accumulated_lengths: A vector of accumulated lengths of size N + 1.
    :return: A matrix of size (total length, 2) representing the assignment of each index to an item in N.
    """
    num_indices = accumulated_lengths[-1] - accumulated_lengths[0]

    indices = np.empty((num_indices, 2), dtype=np.int32)

    index = 0

    for outer_index in range(accumulated_lengths.size - 1):
        length = accumulated_lengths[outer_index + 1] - accumulated_lengths[outer_index]

        for inner_index in range(length):
            indices[index, 0] = outer_index
            indices[index, 1] = inner_index

            index += 1

    return indices


@numba.njit()
def convert_lengths_to_indices(lengths: np.ndarray) -> np.ndarray:
    """
    Convert a vector of item lengths into a matrix of indices.
    For instance, given [2, 3], we obtain [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]].
    :param lengths: A vector of lengths of size N.
    :return: A matrix of size (total length, 2) representing the assignment of each index to an item in N.
    """
    num_indices = np.sum(lengths)

    indices = np.empty((num_indices, 2), dtype=np.int32)

    index = 0

    for outer_index, length in enumerate(lengths):
        for inner_index in range(length):
            indices[index, 0] = outer_index
            indices[index, 1] = inner_index

            index += 1

    return indices


@numba.njit()
def convert_incremental_lengths_to_segment_indices(accumulated_lengths: np.ndarray) -> np.ndarray:
    """
    Convert a vector of accumulated item lengths into a segment of indices.
    For instance, given [0, 2, 5], we obtain [0, 0, 1, 1, 1].
    :param accumulated_lengths: A vector of accumulated lengths of size N + 1.
    :return: A vector of size (total length) representing the assignment of each item index in N.
    """
    num_indices = accumulated_lengths[-1] - accumulated_lengths[0]

    segment_indices = np.empty(num_indices, dtype=np.int32)

    index = 0

    for outer_index in range(accumulated_lengths.size - 1):
        length = accumulated_lengths[outer_index + 1] - accumulated_lengths[outer_index]

        for _ in range(length):
            segment_indices[index] = outer_index

            index += 1

    return segment_indices


@numba.njit()
def convert_incremental_lengths_to_lengths(accumulated_lengths: np.ndarray) -> np.ndarray:
    """
    Convert a vector of accumulated item lengths into lengths.
    For instance, given [0, 2, 5], we obtain [2, 3].
    :param accumulated_lengths: A vector of accumulated lengths of size N + 1.
    :return: A vector of size N representing the length of each item.
    """
    num_items = accumulated_lengths.size - 1

    lengths = np.empty(num_items, dtype=np.int32)

    for index in range(num_items):
        lengths[index] = accumulated_lengths[index + 1] - accumulated_lengths[index]

    return lengths


@numba.njit()
def accumulate(vector: np.ndarray) -> np.ndarray:
    """
    Accumulates the values of a vector such that accumulated[i] = accumulated[i - 1] + vector[i]
    :param vector: A vector of size N.
    :return: A vector of size N + 1.
    """
    accumulated = np.zeros(vector.size + 1, dtype=np.int32)

    for index, value in enumerate(vector):
        accumulated[index + 1] = accumulated[index] + value

    return accumulated
