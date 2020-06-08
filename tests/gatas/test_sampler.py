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

    def test_compute_top_k(self):
        probabilities = np.array(
            [0.16288152, 0.04801334, 0.7079845, 0.01123138, 0.06209923, 0.00779003],
            dtype=np.float32,
        )

        path_indices = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        path_depths = np.array([0], dtype=np.int32)
        coefficients = np.array([1.], dtype=np.int32)

        max_indices, max_values = sampler.compute_top_k(
            probabilities=probabilities,
            path_indices=path_indices,
            path_depths=path_depths,
            coefficients=coefficients,
            num_steps=100,
            k=2,
            noisify=False,
        )

        self.assertAllEqual(np.sort(max_indices), np.sort(np.argsort(probabilities)[-2:]))
        self.assertAllClose(np.sort(max_values), np.sort(np.sort(np.log(probabilities))[-2:]))

    def test_compute_top_large(self):
        probabilities = np.array(
            [0.16288152, 0.04801334, 0.7079845, 0.01123138, 0.06209923, 0.00779003],
            dtype=np.float32,
        )

        path_indices = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        path_depths = np.array([0], dtype=np.int32)
        coefficients = np.array([1.], dtype=np.int32)

        max_indices, max_values = sampler.compute_top_k(
            probabilities=probabilities,
            path_indices=path_indices,
            path_depths=path_depths,
            coefficients=coefficients,
            num_steps=100,
            k=100,
            noisify=False,
        )

        self.assertAllEqual(np.sort(max_indices), np.arange(probabilities.size))
        self.assertAllClose(np.sort(max_values), np.sort(np.log(probabilities)))

    def test_compute_top_k_small(self):
        probabilities = np.array(
            [0.9654813, 0.01837953, 0.00135655, 0.01478269],
            dtype=np.float32,
        )

        path_indices = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        path_depths = np.array([0], dtype=np.int32)
        coefficients = np.array([1.], dtype=np.int32)

        max_indices, max_values = sampler.compute_top_k(
            probabilities=probabilities,
            path_indices=path_indices,
            path_depths=path_depths,
            coefficients=coefficients,
            num_steps=100,
            k=1,
            noisify=False,
        )

        self.assertAllEqual(max_indices, [np.argmax(probabilities)])
        self.assertAllClose(max_values, [np.max(np.log(probabilities))])

    def test_compute_top_k_zero(self):
        probabilities = np.array(
            [
                0.16288152, 0.04801334, 0.7079845, 0.01123138, 0.06209923, 0.00779003,
            ],
            dtype=np.float32)

        path_indices = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        path_depths = np.array([0], dtype=np.int32)
        coefficients = np.array([1.], dtype=np.int32)

        max_indices, max_values = sampler.compute_top_k(
            probabilities=probabilities,
            path_indices=path_indices,
            path_depths=path_depths,
            coefficients=coefficients,
            num_steps=10,
            k=0,
            noisify=False,
        )

        self.assertAllEqual(max_indices, [])
        self.assertAllEqual(max_values, [])

    def test_calculate_transition_logits(self):
        probabilities = np.array(
            [
                0.3333, 0.1110, 0.0370,
                0.3333, 0.1110, 0.0370,
                0.3333, 0.1944, 0.0856,
                0.0833, 0.0486,
                0.0833, 0.0486,
                0.0833, 0.0486,
                0.25, 0.0625, 0.0156,
                0.25, 0.0625, 0.0156,
                0.25, 0.0625, 0.0156,
                0.25, 0.0625, 0.0156,
            ],
            dtype=np.float32,
        )

        path_indices = np.array(
            [
                0, 1, 3,
                0, 2, 4,
                0, 1, 5,
                2, 6,
                1, 7,
                2, 8,
                0, 1, 9,
                0, 2, 10,
                0, 1, 11,
                0, 2, 3,
            ],
            dtype=np.int32,
        )

        path_depths = sampler.compute_path_depths(2, 3)

        coefficients = 1 - np.arange(3, dtype=np.float32) / 3
        coefficients = np.exp(coefficients)
        coefficients /= np.sum(coefficients)

        logits = sampler.calculate_transition_logits(
            probabilities=probabilities,
            coefficients=coefficients[path_depths[path_indices]],
            noisify=False,
        )

        expected_probabilities = [
            0.1495, 0.0357, 0.0085,
            0.1495, 0.0357, 0.0085,
            0.1495, 0.0625, 0.0197,
            0.0268, 0.0112,
            0.0268, 0.0112,
            0.0268, 0.0112,
            0.1121, 0.0201, 0.0036,
            0.1121, 0.0201, 0.0036,
            0.1121, 0.0201, 0.0036,
            0.1121, 0.0201, 0.0036,
        ]

        self.assertAllClose(np.exp(logits), expected_probabilities, rtol=1e-4, atol=1e-4)

    def test_generate_sample(self):
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
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6, 0.7,
        ], dtype=np.float32)

        coefficients = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        path_depths = sampler.compute_path_depths(2, 3)

        indices, segments, path_indices_subset, steps, probabilities_subset = sampler.generate_sample(
            node_indices=np.array([0, 1, 2], np.int32),
            transition_pointers=accumulated_transition_lengths,
            neighbours=neighbours,
            path_indices=path_indices,
            probabilities=probabilities,
            coefficients=coefficients,
            path_depths=path_depths,
            num_steps=2,
            sample_size=100,
            noisify=False,
        )

        expected_order = np.array([0, 2, 1, 3, 5, 4, 6])
        expected_indices = neighbours[expected_order]
        expected_path_indices = path_indices[expected_order]
        expected_segments = [0, 0, 0, 2, 2, 2, 2]
        expected_steps = path_depths[expected_path_indices]
        expected_probabilities = probabilities[expected_order]

        self.assertAllEqual(indices, expected_indices)
        self.assertAllEqual(path_indices_subset, expected_path_indices)
        self.assertAllEqual(segments, expected_segments)
        self.assertAllEqual(steps, expected_steps)
        self.assertAllClose(probabilities_subset, expected_probabilities)

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
            0.1, 0.2, 0.3,
            0.4, 0.5, 0.6, 0.7,
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

        coefficients = neighbour_sampler.coefficients.numpy()
        coefficients = np.exp(coefficients)
        coefficients /= np.sum(coefficients)

        expected_order = np.array([0, 2, 1, 3, 5, 4, 6])
        expected_indices = neighbours[expected_order]
        expected_path_indices = path_indices[expected_order]
        expected_segments = [0, 0, 0, 2, 2, 2, 2]
        expected_weights = (probabilities * coefficients[neighbour_sampler.path_depths[path_indices]])[expected_order]

        self.assertAllEqual(sample.indices, expected_indices)
        self.assertAllEqual(sample.path_indices, expected_path_indices)
        self.assertAllEqual(sample.segments, expected_segments)
        self.assertAllClose(sample.weights, expected_weights)
