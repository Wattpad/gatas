import numpy as np
import sklearn.metrics as sk_metrics

from framework.trackers import Numeric


def f1_score_micro(targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> Numeric:
    predictions = np.round(probabilities)

    mask = weights.astype(np.bool)

    score = sk_metrics.f1_score(targets[mask], predictions[mask])

    return score


def f1_score_macro(targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> Numeric:
    predictions = np.round(probabilities)

    scores = []

    class_data = zip(np.transpose(targets), np.transpose(predictions), np.transpose(weights))

    for class_targets, class_predictions, class_weights in class_data:
        if np.sum(class_weights) == 0:
            continue

        score = sk_metrics.f1_score(class_targets, class_predictions, sample_weight=class_weights)

        scores.append(score)

    score = sum(scores) / len(scores)

    return score


def roc_auc(targets: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> Numeric:
    scores = []

    class_data = zip(np.transpose(targets), np.transpose(probabilities), np.transpose(weights))

    for class_targets, class_probabilities, class_weights in class_data:
        if np.sum(class_weights) == 0:
            continue

        score = sk_metrics.roc_auc_score(class_targets, class_probabilities, sample_weight=class_weights)

        scores.append(score)

    score = sum(scores) / len(scores)

    return score
