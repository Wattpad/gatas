import numpy as np
import tensorflow as tf

from gatas import transitions


class TransitionTensorsTest(tf.test.TestCase):
    def test_transition_tensors(self):
        accumulated_num_edges = np.array([0, 3, 3, 7, 7, 7, 7], dtype=np.int32)
        adjacencies = np.array([0, 1, 2, 2, 3, 4, 5], dtype=np.int32)
        edge_types = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int32)

        accumulated_transition_lengths, neighbours, path_indices, probabilities = transitions.create_transition_tensors(
            accumulated_num_edges=accumulated_num_edges,
            adjacencies=adjacencies,
            edge_types=edge_types,
            num_steps=2,
        )

        expected_accumulated_transition_lengths = [
            0, 6, 7, 11, 12, 13, 14
        ]

        expected_neighbours = [
            0,
            1, 2,
            3, 4, 5,
            1,
            2,
            3, 4, 5,
            3,
            4,
            5,
        ]

        expected_path_indices = [
            0,
            2, 1,
            7, 8, 7,
            0,
            0,
            1, 2, 1,
            0,
            0,
            0,
        ]

        expected_probabilities = [
            1.0,
            0.5, 0.5,
            0.3333, 0.3333, 0.3333,
            1.0,
            1.0,
            0.3333, 0.3333, 0.3333,
            1.0,
            1.0,
            1.0
        ]

        self.assertAllEqual(expected_accumulated_transition_lengths, accumulated_transition_lengths)
        self.assertAllEqual(expected_neighbours, neighbours)
        self.assertAllEqual(expected_path_indices, path_indices)
        self.assertAllClose(expected_probabilities, probabilities, rtol=1e-4, atol=1e-4)
