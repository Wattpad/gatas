from enum import Enum
from typing import Optional

import numpy as np
import sklearn.metrics as sk_metrics

from framework.trackers import Numeric, Aggregator


class Metric(Enum):
    TRAINING_ACCURACY = 'Training Accuracy'
    VALIDATION_ACCURACY = 'Validation Accuracy'
    TESTING_ACCURACY = 'Testing Accuracy'

    VALIDATION_WEIGHTED_F1 = 'Validation Weighted F1 Score'
    TESTING_WEIGHTED_F1 = 'Testing Weighted F1 Score'
    VALIDATION_WEIGHTED_PRECISION = 'Validation Weighted Precision'
    TESTING_WEIGHTED_PRECISION = 'Testing Weighted Precision'
    VALIDATION_WEIGHTED_RECALL = 'Validation Weighted Recall'
    TESTING_WEIGHTED_RECALL = 'Testing Weighted Recall'

    TRAINING_MICRO_F1 = 'Training Micro F1 Score'
    VALIDATION_MICRO_F1 = 'Validation Micro F1 Score'
    TESTING_MICRO_F1 = 'Testing Micro F1 Score'
    VALIDATION_MICRO_PRECISION = 'Validation Micro Precision'
    TESTING_MICRO_PRECISION = 'Testing Micro Precision'
    VALIDATION_MICRO_RECALL = 'Validation Micro Recall'
    TESTING_MICRO_RECALL = 'Testing Micro Recall'

    TRAINING_MACRO_F1 = 'Training Macro F1 Score'
    VALIDATION_MACRO_F1 = 'Validation Macro F1 Score'
    TESTING_MACRO_F1 = 'Testing Macro F1 Score'
    VALIDATION_MACRO_PRECISION = 'Validation Macro Precision'
    TESTING_MACRO_PRECISION = 'Testing Macro Precision'
    VALIDATION_MACRO_RECALL = 'Validation Macro Recall'
    TESTING_MACRO_RECALL = 'Testing Macro Recall'

    TRAINING_MEAN_COST = 'Training Mean Cost'
    VALIDATION_MEAN_COST = 'Validation Mean Cost'
    TESTING_MEAN_COST = 'Testing Mean Cost'

    VALIDATION_JACCARD_SCORE = 'Validation Jaccard Score'

    MEAN_BETA = 'Mean Beta'


class MetricFunctions:
    @staticmethod
    def accuracy(targets: np.ndarray, predictions: np.ndarray) -> Numeric:
        return sk_metrics.accuracy_score(targets, predictions)

    @staticmethod
    def precision(targets: np.ndarray, predictions: np.ndarray, average: Optional[str] = 'weighted') -> Numeric:
        return sk_metrics.precision_score(targets, predictions, average=average, zero_division=0)

    @staticmethod
    def recall(targets: np.ndarray, predictions: np.ndarray, average: Optional[str] = 'weighted') -> Numeric:
        return sk_metrics.recall_score(targets, predictions, average=average, zero_division=0)

    @staticmethod
    def f1_score(targets: np.ndarray, predictions: np.ndarray, average: Optional[str] = 'weighted') -> Numeric:
        return sk_metrics.f1_score(targets, predictions, average=average, zero_division=0)

    @staticmethod
    def jaccard_score(targets: np.ndarray, predictions: np.ndarray, average: Optional[str] = 'weighted') -> Numeric:
        return sk_metrics.jaccard_score(targets, predictions, average=average)

    @staticmethod
    def jaccard_score_multiclass(targets: np.ndarray, predictions: np.ndarray) -> Numeric:
        overlap = np.sum(targets * predictions, axis=1)

        union = np.sum(targets + predictions > 0, axis=1)

        score = float(np.mean(np.divide(overlap, union, out=np.zeros_like(union, dtype=np.float32), where=union != 0)))

        return score

    @staticmethod
    def mean(array: np.ndarray) -> Numeric:
        return float(np.mean(array))

    @staticmethod
    def calculate_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> np.array:
        num_classes = targets.shape[1]

        best_thresholds = np.empty(shape=num_classes, dtype=np.float32)

        for index in range(num_classes):
            precisions, recalls, thresholds = sk_metrics.precision_recall_curve(
                y_true=targets[:, index],
                probas_pred=probabilities[:, index],
            )

            f1_scores_denominator = precisions + recalls
            f1_scores = np.divide(
                2 * precisions * recalls,
                f1_scores_denominator,
                out=np.zeros_like(f1_scores_denominator),
                where=f1_scores_denominator != 0,
            )

            best_thresholds[index] = thresholds[np.nanargmax(f1_scores) - 1]

        return best_thresholds

    @classmethod
    def calculate_with_optimized_predictions(cls, targets: np.ndarray, probabilities: np.ndarray, function: Aggregator) -> Numeric:
        thresholds = cls.calculate_thresholds(targets, probabilities)

        predictions = probabilities >= thresholds

        score = function(targets, predictions.astype(np.float32))

        return score
