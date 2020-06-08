from typing import Optional, Union

import numpy as np
import tensorflow as tf

from framework.common.parameters import DataSpecification
from gatas.aggregator import NeighbourAggregator
from gatas.sampler import NeighbourSampler


class LinkPredictor(tf.Module):
    def __init__(
            self,
            path: str,
            num_nodes: int,
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
            num_classes: int,
            group_size: int,
            node_features: Optional[Union[tf.Tensor, np.ndarray]] = None):

        super().__init__()

        self.sample_size = sample_size
        self.lambda_coefficient = lambda_coefficient
        self.group_size = group_size

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

        self.node_transformer = tf.keras.layers.Dense(units=layer_size_classifier, activation=tf.nn.elu)

        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(units=layer_size_classifier, activation=tf.nn.elu),
            tf.keras.layers.Dense(units=layer_size_classifier, activation=tf.nn.elu),
            tf.keras.layers.Dense(units=num_classes, use_bias=False),
        ])

    @classmethod
    def get_schema(cls, num_classes: int) -> DataSpecification:
        schema = DataSpecification([
            tf.TensorSpec(tf.TensorShape((None,)), tf.int32, 'anchor_indices'),
            tf.TensorSpec(tf.TensorShape((None, 2)), tf.int32, 'group_indices'),
            tf.TensorSpec(tf.TensorShape((None, num_classes)), tf.int32, 'targets'),
            tf.TensorSpec(tf.TensorShape((None, num_classes)), tf.float32, 'target_weights'),
        ])

        return schema

    def __call__(self, anchor_indices: tf.Tensor, group_indices: tf.Tensor, training: Union[tf.Tensor, bool]):
        neighbour_sample = self.neighbour_sampler(anchor_indices, self.sample_size)

        node_representations = self.neighbour_aggregator(
            anchor_indices=anchor_indices,
            neighbour_indices=neighbour_sample.indices,
            neighbour_assignments=neighbour_sample.segments,
            neighbour_weights=neighbour_sample.weights,
            neighbour_path_indices=neighbour_sample.path_indices,
            training=training,
        )

        node_representations = self.node_transformer(node_representations)

        num_groups = tf.reduce_max(group_indices, axis=0)[0] + 1
        node_representations_size = node_representations.shape[1]

        node_representations_grouped = tf.scatter_nd(
            indices=group_indices,
            updates=node_representations,
            shape=(num_groups, self.group_size, node_representations_size),
        )

        logits = self.classifier(tf.reshape(
            tensor=node_representations_grouped,
            shape=(num_groups, self.group_size * node_representations_size),
        ))

        probabilities = tf.nn.sigmoid(logits)

        return logits, probabilities

    def calculate_loss(self, logits: tf.Tensor, targets: tf.Tensor, target_weights: tf.Tensor) -> tf.Tensor:
        loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=targets,
            logits=logits,
            weights=target_weights,
            reduction=tf.losses.Reduction.NONE,
        )

        l2_norm = tf.add_n([
            tf.nn.l2_loss(variable)
            for variable in tf.trainable_variables()
            if variable.name not in {'coefficient_indices:0', 'edge_type_embeddings:0'}
        ])

        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1)) + self.lambda_coefficient * l2_norm

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
