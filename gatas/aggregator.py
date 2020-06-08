from typing import Tuple, Optional, Union

import numba
import numpy as np
import tensorflow as tf

from gatas import common


class NeighbourAggregator(tf.Module):
    def __init__(
            self,
            input_noise_rate: float,
            dropout_rate: float,
            num_nodes: int,
            num_edge_types: int,
            num_steps: int,
            edge_type_embedding_size: int,
            node_embedding_size: Optional[int],
            layer_size: int,
            num_attention_heads: int,
            node_features: Optional[Union[tf.Tensor, np.ndarray]] = None) -> None:

        super().__init__()

        self.num_edge_types = num_edge_types
        self.layer_size = layer_size
        self.num_attention_heads = num_attention_heads

        if node_features is not None:
            self.node_features = tf.convert_to_tensor(node_features, name='node_embeddings')
            node_size = layer_size + node_embedding_size if node_embedding_size else layer_size

        else:
            self.node_features = None
            node_size = node_embedding_size if node_embedding_size else 0

        initialize = tf.glorot_uniform_initializer()

        self.feature_transformer = tf.keras.Sequential(
            layers=[
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(rate=input_noise_rate),
                tf.keras.layers.Dense(layer_size, use_bias=True, activation=tf.nn.elu),
                tf.keras.layers.Dropout(rate=dropout_rate),
            ],
            name='input_transformation',
        )

        if node_embedding_size:
            self.node_embeddings = tf.Variable(
                initial_value=initialize(shape=(num_nodes, node_embedding_size), dtype=tf.float32),
                name='trainable_node_embeddings',
            )

        else:
            self.node_embeddings = None

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.edge_type_embeddings = tf.Variable(
            initial_value=initialize(shape=(num_edge_types, edge_type_embedding_size), dtype=tf.float32),
            name='edge_type_embeddings',
        )

        self.positional_embeddings = tf.convert_to_tensor(common.compute_positional_embeddings(
            max_length=num_steps,
            num_features=edge_type_embedding_size,
        ))

        self.coefficient_transformer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, use_bias=False),
            tf.keras.layers.Dropout(dropout_rate),
        ])

        self.value_transformer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=layer_size, use_bias=True, activation=tf.nn.elu),
            tf.keras.layers.Dropout(dropout_rate),
        ])

        self.attention_hidden_weights = tf.Variable(
            initial_value=initialize(shape=(node_size + layer_size, layer_size, num_attention_heads), dtype=tf.float32),
            name='attention_hidden_weights',
        )

        self.attention_hidden_biases = tf.Variable(
            initial_value=tf.zeros(shape=(layer_size, num_attention_heads), dtype=tf.float32),
            name='attention_hidden_biases',
        )

        self.attention_weights = tf.Variable(
            initial_value=initialize(shape=(layer_size, num_attention_heads), dtype=tf.float32),
            name='attention_weights',
        )

        self.values_weights = tf.Variable(
            initial_value=initialize(shape=(layer_size, layer_size, num_attention_heads), dtype=tf.float32),
            name='values_weights',
        )

        self.values_biases = tf.Variable(
            initial_value=tf.zeros(shape=(layer_size, num_attention_heads), dtype=tf.float32),
            name='values_biases',
        )

    def __call__(
            self,
            anchor_indices: tf.Tensor,
            neighbour_indices: tf.Tensor,
            neighbour_assignments: tf.Tensor,
            neighbour_weights: tf.Tensor,
            neighbour_path_indices: tf.Tensor,
            training: bool = False,
            concatenate: bool = True) -> tf.Tensor:

        anchor_features, neighbour_features = self.generate_features(
            anchor_indices=anchor_indices,
            neighbour_indices=neighbour_indices,
            training=training,
        )

        if self.node_embeddings is not None:
            anchor_features, neighbour_features = self.build_embeddings(
                anchor_indices=anchor_indices,
                neighbour_indices=neighbour_indices,
                anchor_features=anchor_features,
                neighbour_features=neighbour_features,
                training=training,
            )

        neighbour_embeddings = self.build_embeddings_with_path(
            path_indices=neighbour_path_indices,
            anchor_features=anchor_features,
            neighbour_features=neighbour_features,
            neighbour_assignments=neighbour_assignments,
            training=training,
        )

        node_representations = self.create_attended_representations(
            anchor_embeddings=anchor_features,
            neighbour_embeddings=neighbour_embeddings,
            neighbour_assignments=neighbour_assignments,
            neighbour_weights=neighbour_weights,
            training=training,
            concatenate=concatenate,
        )

        return node_representations

    def generate_features(
            self,
            anchor_indices: tf.Tensor,
            neighbour_indices: tf.Tensor,
            training: bool) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:

        if self.node_features is None:
            return None, None

        anchor_features = self.feature_transformer(
            inputs=tf.nn.embedding_lookup(self.node_features, anchor_indices),
            training=training,
        )

        neighbour_features = self.feature_transformer(
            inputs=tf.nn.embedding_lookup(self.node_features, neighbour_indices),
            training=training,
        )

        return anchor_features, neighbour_features

    def build_embeddings(
            self,
            anchor_indices: tf.Tensor,
            neighbour_indices: tf.Tensor,
            anchor_features: Optional[tf.Tensor],
            neighbour_features: Optional[tf.Tensor],
            training: bool) -> Tuple[tf.Tensor, tf.Tensor]:

        anchor_embeddings = self.dropout(
            inputs=tf.nn.embedding_lookup(self.node_embeddings, anchor_indices),
            training=training,
        )

        neighbour_embeddings = self.dropout(
            inputs=tf.nn.embedding_lookup(self.node_embeddings, neighbour_indices),
            training=training,
        )

        if anchor_features is not None:
            anchor_embeddings = tf.concat((anchor_embeddings, anchor_features), axis=-1)

        if neighbour_features is not None:
            neighbour_embeddings = tf.concat((neighbour_embeddings, neighbour_features), axis=-1)

        return anchor_embeddings, neighbour_embeddings

    def get_relation_path_sequences(self, path_indices: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        segment_indices_op, position_indices_op, edge_type_indices_op = tf.numpy_function(
            func=lambda x: compute_edge_type_path_sequences(x, self.num_edge_types),
            inp=(path_indices,),
            Tout=(tf.int32, tf.int32, tf.int32),
        )

        segment_indices_op.set_shape((None,))
        position_indices_op.set_shape((None,))
        edge_type_indices_op.set_shape((None,))

        return segment_indices_op, position_indices_op, edge_type_indices_op

    def build_embeddings_with_path(
            self,
            path_indices: tf.Tensor,
            anchor_features: tf.Tensor,
            neighbour_features: tf.Tensor,
            neighbour_assignments: tf.Tensor,
            training: bool) -> tf.Tensor:

        segment_indices, position_indices, edge_type_indices = self.get_relation_path_sequences(path_indices)

        path_embeddings = tf.add(
            x=tf.gather(self.edge_type_embeddings, edge_type_indices),
            y=tf.gather(self.positional_embeddings, position_indices),
        )

        coefficients = tf.gather(tf.gather(anchor_features, neighbour_assignments), segment_indices)
        coefficients = tf.concat((coefficients, path_embeddings), axis=-1)
        coefficients = self.coefficient_transformer(coefficients, training=training)
        coefficients = common.segment_softmax(coefficients, segment_indices)

        values = tf.concat((tf.gather(neighbour_features, segment_indices), path_embeddings), axis=-1)
        values = self.value_transformer(values, training=training)

        neighbour_embeddings = tf.math.segment_sum(coefficients * values, segment_indices)

        return neighbour_embeddings

    def create_attended_representations(
            self,
            anchor_embeddings: tf.Tensor,
            neighbour_embeddings: tf.Tensor,
            neighbour_assignments: tf.Tensor,
            neighbour_weights: tf.Tensor,
            training: bool,
            concatenate: bool) -> tf.Tensor:

        embeddings = tf.concat((tf.gather(anchor_embeddings, neighbour_assignments), neighbour_embeddings), axis=1)

        attention_coefficients = tf.einsum('bd,dfa->bfa', embeddings, self.attention_hidden_weights)
        attention_coefficients = tf.nn.elu(attention_coefficients + self.attention_hidden_biases[tf.newaxis, :, :])
        attention_coefficients = tf.einsum('bda,da->ba', attention_coefficients, self.attention_weights)
        attention_coefficients = tf.math.log(neighbour_weights)[:, tf.newaxis] + attention_coefficients
        attention_coefficients = common.segment_softmax(attention_coefficients, neighbour_assignments)
        attention_coefficients = self.dropout(attention_coefficients[:, tf.newaxis, :], training=training)

        neighbour_embeddings = tf.einsum('bd,dfa->bfa', neighbour_embeddings, self.values_weights)
        neighbour_embeddings = tf.nn.elu(neighbour_embeddings + self.values_biases[tf.newaxis, :, :])
        neighbour_embeddings = self.dropout(neighbour_embeddings, training=training)
        neighbour_embeddings = attention_coefficients * neighbour_embeddings

        attention_heads = tf.nn.elu(tf.math.segment_sum(neighbour_embeddings, neighbour_assignments))

        if concatenate:
            anchors_attended = tf.reshape(attention_heads, (-1, self.layer_size * self.num_attention_heads))
        else:
            anchors_attended = tf.reduce_mean(attention_heads, axis=-1)

        return anchors_attended


@numba.njit
def compute_edge_type_path_sequences(
        path_indices: np.ndarray,
        num_edge_types: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    segment_indices, position_indices, edge_type_indices = [], [], []

    for segment_index, edge_type_path_index in enumerate(path_indices):
        if edge_type_path_index == 0:
            segment_indices.append(segment_index)
            position_indices.append(0)
            edge_type_indices.append(0)

        else:
            position_index = 1

            while edge_type_path_index >= 0:
                segment_indices.append(segment_index)
                position_indices.append(position_index)
                edge_type_indices.append(edge_type_path_index % num_edge_types)

                edge_type_path_index = (edge_type_path_index / num_edge_types) - 1
                position_index += 1

    segment_indices = np.array(segment_indices, dtype=np.int32)
    position_indices = np.array(position_indices, dtype=np.int32)
    edge_type_indices = np.array(edge_type_indices, dtype=np.int32)

    return segment_indices, position_indices, edge_type_indices
