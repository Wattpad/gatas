import numpy as np
import tensorflow as tf


def segment_softmax(logits: tf.Tensor, segment_ids: tf.Tensor) -> tf.Tensor:
    logits_max = tf.math.segment_max(logits, segment_ids)

    logits_exp = tf.math.exp(logits - tf.gather(logits_max, segment_ids))

    partitions = tf.gather(tf.math.segment_sum(logits_exp, segment_ids), segment_ids)

    softmax = logits_exp / partitions

    return softmax


def segment_normalize(logits: tf.Tensor, segment_ids: tf.Tensor) -> tf.Tensor:
    partitions = tf.gather(tf.math.segment_sum(logits, segment_ids), segment_ids)

    probabilities = logits / partitions

    return probabilities


def compute_positional_embeddings(max_length: int, num_features: int) -> np.ndarray:
    feature_indices, positions = np.arange(num_features, dtype=np.float32), np.arange(max_length + 1, dtype=np.float32)

    angle_rates = 1 / np.power(10000, 2 * (feature_indices // 2) / num_features)
    positional_encodings = positions[:, np.newaxis] * angle_rates[np.newaxis, :]

    positional_encodings[:, 0::2] = np.sin(positional_encodings[:, 0::2])
    positional_encodings[:, 1::2] = np.cos(positional_encodings[:, 1::2])

    return positional_encodings
