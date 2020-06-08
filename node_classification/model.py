from typing import Tuple, Union, Optional

import numpy as np
import tensorflow as tf

from framework.common.parameters import DataSpecification
from gatas.aggregator import NeighbourAggregator
from gatas.sampler import NeighbourSampler


class NodeClassifier(tf.Module):
    def __init__(
            self,
            path: str,
            num_nodes: int,
            num_classes: int,
            layer_size: int,
            layer_size_classifier: int,
            num_attention_heads: int,
            edge_type_embedding_size: int,
            node_embedding_size: Optional[int],
            num_steps: int,
            sample_size: int,
            input_noise_rate: float,
            dropout_rate: float,
            lambda_coefficient: float,
            node_features: Optional[Union[tf.Tensor, np.ndarray]] = None):

        super().__init__()

        self.sample_size = sample_size
        self.lambda_coefficient = lambda_coefficient

        self.neighbour_sampler = NeighbourSampler.from_path(num_steps, path)

        self.neighbour_aggregator = NeighbourAggregator(
            input_noise_rate=input_noise_rate,
            dropout_rate=dropout_rate,
            num_nodes=num_nodes,
            num_edge_types=self.neighbour_sampler.num_edge_types,
            num_steps=num_steps,
            edge_type_embedding_size=edge_type_embedding_size,
            node_embedding_size=node_embedding_size,
            layer_size=layer_size,
            num_attention_heads=num_attention_heads,
            node_features=node_features,
        )

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(units=layer_size_classifier, activation=tf.nn.elu),
            tf.keras.layers.Dense(units=layer_size_classifier, activation=tf.nn.elu),
            tf.keras.layers.Dense(units=num_classes, use_bias=False),
        ])

    @staticmethod
    def get_schema(num_classes: int) -> DataSpecification:
        schema = DataSpecification([
            tf.TensorSpec(tf.TensorShape((None,)), tf.int32, 'anchor_indices'),
            tf.TensorSpec(tf.TensorShape((None, num_classes)), tf.int32, 'targets'),
            # tf.TensorSpec(tf.TensorShape((None,)), tf.int32, 'targets'),
        ])

        return schema

    def __call__(self, anchor_indices: tf.Tensor, training: Union[tf.Tensor, bool]) -> Tuple[tf.Tensor, tf.Tensor]:
        neighbour_sample = self.neighbour_sampler(anchor_indices, self.sample_size)

        node_representations = self.neighbour_aggregator(
            anchor_indices=anchor_indices,
            neighbour_indices=neighbour_sample.indices,
            neighbour_assignments=neighbour_sample.segments,
            neighbour_weights=neighbour_sample.weights,
            neighbour_path_indices=neighbour_sample.path_indices,
            training=training,
        )

        logits = self.classifier(node_representations)

        predictions = tf.cast(tf.round(tf.nn.sigmoid(logits)), tf.int32)
        # predictions = tf.argmax(tf.nn.softmax(logits), axis=1, output_type=tf.int32)

        return logits, predictions

    def calculate_loss(self, logits: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=targets, logits=logits)
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=targets, logits=logits)

        l2_norm = tf.add_n([
            tf.nn.l2_loss(variable)
            for variable in tf.trainable_variables()
            if variable.name not in {'coefficient_indices:0', 'edge_type_embeddings:0'}
        ])

        loss = tf.reduce_mean(loss) + self.lambda_coefficient * l2_norm

        return loss

    @classmethod
    def get_clipped_gradient_updates(
            cls,
            loss: tf.Tensor,
            optimizer: tf.train.Optimizer,
            max_gradient_norm: float = 5.0) -> tf.Tensor:

        gradients, variables = zip(*optimizer.compute_gradients(loss))

        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        updates = optimizer.apply_gradients(zip(clipped_gradients, variables))

        return updates
