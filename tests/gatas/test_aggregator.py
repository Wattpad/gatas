import tensorflow as tf

from gatas.aggregator import NeighbourAggregator


class NeighbourAggregatorTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.enable_eager_execution()

    def test_node_representations(self):
        anchor_indices = [1, 2, 3]

        neighbour_indices = [0, 4, 5]
        neighbour_path_indices = [0, 1, 4]
        neighbour_assignments = [0, 0, 2]
        neighbour_weights = [1., 1., 1.]

        node_features = [
            [0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.],
            [10., 11., 12., 13., 14.],
            [8., 11., 10., 14., 20.],
            [0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.],
        ]

        layer_size = 10
        num_attention_heads = 4

        model = NeighbourAggregator(
            input_noise_rate=0.0,
            dropout_rate=0.0,
            num_nodes=6,
            num_edge_types=2,
            num_steps=3,
            edge_type_embedding_size=6,
            node_embedding_size=6,
            layer_size=layer_size,
            num_attention_heads=num_attention_heads,
            node_features=tf.convert_to_tensor(node_features),
        )

        neighbour_probabilities = model(
            anchor_indices=tf.convert_to_tensor(anchor_indices),
            neighbour_indices=tf.convert_to_tensor(neighbour_indices),
            neighbour_assignments=tf.convert_to_tensor(neighbour_assignments),
            neighbour_weights=tf.convert_to_tensor(neighbour_weights),
            neighbour_path_indices=tf.convert_to_tensor(neighbour_path_indices),
        )

        self.assertAllEqual(neighbour_probabilities.shape, [3, layer_size * num_attention_heads])
