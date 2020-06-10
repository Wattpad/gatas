import numpy as np
import tensorflow as tf

from gatas import sampler
from gatas.sampler import NeighbourSampler


class NeighbourhoodSamplerTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.enable_eager_execution()

    def test_compute_path_depths(self):
        path_depths = sampler.compute_path_depths(3, 3)
        expected_path_depths = np.array([
            0,
            1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        ], dtype=np.int32)

        self.assertAllEqual(path_depths, expected_path_depths)

    def test_neighbour_sampler(self):
        accumulated_transition_lengths = np.array([0, 3, 3, 7, 7, 7, 7], np.int32)

        neighbours = np.array([
            1, 1, 2,
            2, 3, 4, 5,
        ], dtype=np.int32)

        path_indices = np.array([
            1, 2, 2,
            3, 4, 5, 6,
        ], dtype=np.int32)

        probabilities = np.array([
            0.2, 0.1, 0.8,
            0.8, 0.1, 0.7, 0.2,
        ], dtype=np.float32)

        neighbour_sampler = NeighbourSampler(
            accumulated_transition_lengths=accumulated_transition_lengths,
            neighbours=neighbours,
            path_indices=path_indices,
            probabilities=probabilities,
            num_edge_types=3,
            num_steps=2,
        )

        sample = neighbour_sampler(
            node_indices=tf.convert_to_tensor([0, 1, 2], dtype=tf.int32),
            sample_size=100,
            noisify=False,
        )

        path_depths = neighbour_sampler.path_depths.numpy()

        coefficients = neighbour_sampler.coefficients.numpy()
        coefficients = np.exp(coefficients)
        coefficients /= np.sum(coefficients)

        expected_order = np.array([2, 0, 1, 3, 5, 6, 4])
        expected_indices = neighbours[expected_order]
        expected_path_indices = path_indices[expected_order]
        expected_segments = [0, 0, 0, 2, 2, 2, 2]
        expected_weights = (probabilities * coefficients[path_depths[path_indices]])[expected_order]

        self.assertAllEqual(sample.indices, expected_indices)
        self.assertAllEqual(sample.path_indices, expected_path_indices)
        self.assertAllEqual(sample.segments, expected_segments)
        self.assertAllClose(sample.weights, expected_weights)
