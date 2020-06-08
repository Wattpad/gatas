import os
import traceback
from typing import Tuple, Callable, NamedTuple, List, Optional

import defopt
import numpy as np
import tensorflow as tf

from framework.common import parameters
from framework.trackers.aggregator import Aggregation, Statistic
from framework.trackers.metrics import MetricFunctions, Metric
from framework.trackers.tracker_mlflow import MLFlowTracker
from link_prediction import dataset
from link_prediction import metrics as metric_functions
from link_prediction.model import LinkPredictor


TRAINING_TARGET_WEIGHTS = 'Training Target Weights'
VALIDATION_TARGET_WEIGHTS = 'Validation Target Weights'
TESTING_TARGET_WEIGHTS = 'Testing Target Weights'

TRAINING_ROC_AUC = 'Training ROC AUC'
VALIDATION_ROC_AUC = 'Validation ROC AUC'
TESTING_ROC_AUC = 'Testing ROC AUC'


class ModelMetrics(NamedTuple):
    test_roc_auc: float
    test_macro_f1: float
    test_micro_f1: float


class NodeClassifierTrainer:
    def __init__(
            self,
            data_path: str,
            model_path: str,
            num_nodes: int,
            num_classes: int,
            max_steps: int,
            sample_size: int,
            layer_size: int,
            layer_size_classifier: int,
            num_attention_heads: int,
            edge_type_embedding_size: int,
            node_embedding_size: Optional[int],
            input_noise_rate: float,
            dropout_rate: float,
            lambda_coefficient: float,
            learning_rate: float) -> None:

        self.data_path = data_path

        self.checkpoint_path = os.path.join(model_path, 'model.ckpt')

        self.iterator = self.create_iterator(num_classes)

        self.model = LinkPredictor(
            path=data_path,
            num_nodes=num_nodes,
            layer_size=layer_size,
            layer_size_classifier=layer_size_classifier,
            num_attention_heads=num_attention_heads,
            edge_type_embedding_size=edge_type_embedding_size,
            node_embedding_size=node_embedding_size,
            num_steps=max_steps,
            sample_size=sample_size,
            input_noise_rate=input_noise_rate,
            dropout_rate=dropout_rate,
            lambda_coefficient=lambda_coefficient,
            num_classes=num_classes,
            group_size=2,
        )

        self.anchor_indices, self.group_indices, self.targets, self.target_weights = self.iterator.get_next()
        self.training = tf.placeholder_with_default(False, shape=(), name='training')
        self.logits, self.probabilities = self.model(self.anchor_indices, self.group_indices, self.training)
        self.loss = self.model.calculate_loss(self.logits, self.targets, self.target_weights)
        self.updates = self.model.get_clipped_gradient_updates(self.loss, optimizer=tf.contrib.opt.NadamOptimizer(learning_rate))

        self.tracker = MLFlowTracker('link-prediction', aggregators=[
            Aggregation(
                Metric.TRAINING_MEAN_COST,
                [Statistic.TRAINING_COST],
                MetricFunctions.mean),
            Aggregation(
                Metric.VALIDATION_MEAN_COST,
                [Statistic.VALIDATION_COST],
                MetricFunctions.mean),
            Aggregation(
                Metric.TESTING_MEAN_COST,
                [Statistic.TESTING_COST],
                MetricFunctions.mean),
            Aggregation(
                Metric.TRAINING_MICRO_F1,
                [Statistic.TRAINING_TARGET, Statistic.TRAINING_PROBABILITY, TRAINING_TARGET_WEIGHTS],
                lambda x, y, z: metric_functions.f1_score_micro(x, y, z)),
            Aggregation(
                Metric.VALIDATION_MICRO_F1,
                [Statistic.VALIDATION_TARGET, Statistic.VALIDATION_PROBABILITY, VALIDATION_TARGET_WEIGHTS],
                lambda x, y, z: metric_functions.f1_score_micro(x, y, z)),
            Aggregation(
                Metric.TESTING_MICRO_F1,
                [Statistic.TESTING_TARGET, Statistic.TESTING_PROBABILITY, TESTING_TARGET_WEIGHTS],
                lambda x, y, z: metric_functions.f1_score_micro(x, y, z)),
            Aggregation(
                Metric.TRAINING_MACRO_F1,
                [Statistic.TRAINING_TARGET, Statistic.TRAINING_PROBABILITY, TRAINING_TARGET_WEIGHTS],
                lambda x, y, z: metric_functions.f1_score_macro(x, y, z)),
            Aggregation(
                Metric.VALIDATION_MACRO_F1,
                [Statistic.VALIDATION_TARGET, Statistic.VALIDATION_PROBABILITY, VALIDATION_TARGET_WEIGHTS],
                lambda x, y, z: metric_functions.f1_score_macro(x, y, z)),
            Aggregation(
                Metric.TESTING_MACRO_F1,
                [Statistic.TESTING_TARGET, Statistic.TESTING_PROBABILITY, TESTING_TARGET_WEIGHTS],
                lambda x, y, z: metric_functions.f1_score_macro(x, y, z)),
            Aggregation(
                TRAINING_ROC_AUC,
                [Statistic.TRAINING_TARGET, Statistic.TRAINING_PROBABILITY, TRAINING_TARGET_WEIGHTS],
                lambda x, y, z: metric_functions.roc_auc(x, y, z)),
            Aggregation(
                VALIDATION_ROC_AUC,
                [Statistic.VALIDATION_TARGET, Statistic.VALIDATION_PROBABILITY, VALIDATION_TARGET_WEIGHTS],
                lambda x, y, z: metric_functions.roc_auc(x, y, z)),
            Aggregation(
                TESTING_ROC_AUC,
                [Statistic.TESTING_TARGET, Statistic.TESTING_PROBABILITY, TESTING_TARGET_WEIGHTS],
                lambda x, y, z: metric_functions.roc_auc(x, y, z)),
        ])

        self.tracker.set_tags({
            'Tier': 'Development',
            'Problem': 'Link Prediction',
        })

    @staticmethod
    def create_iterator(num_classes: int) -> tf.data.Iterator:
        iterator = tf.data.Iterator.from_structure(
            output_types=LinkPredictor.get_schema(num_classes).get_types(),
            output_shapes=LinkPredictor.get_schema(num_classes).get_shapes(),
        )

        return iterator

    @staticmethod
    def initialize_iterator(
            iterator: tf.data.Iterator,
            generator: Callable,
            arguments: Tuple,
            queue_size: int = -1) -> tf.Operation:

        validation_dataset = tf.data.Dataset \
            .from_generator(
                generator=generator,
                output_types=iterator.output_types,
                output_shapes=iterator.output_shapes,
                args=arguments,
            ) \
            .prefetch(queue_size)

        iterator = iterator.make_initializer(validation_dataset)

        return iterator

    def train_early_stopping(
            self,
            session: tf.Session,
            train_iterator: tf.Operation,
            validation_iterator: tf.Operation,
            test_iterator: tf.Operation,
            save_path: str,
            num_epochs: int,
            early_stopping_threshold: int,
            previous_best_metric: float = -np.inf,
            saver_variables: Optional[List[tf.Tensor]] = None) -> ModelMetrics:

        saver = tf.train.Saver(var_list=saver_variables)

        non_improvement_times = 0

        for epoch in range(num_epochs):
            session.run(train_iterator)

            try:
                while True:
                    cost_, probabilities_, targets_, target_weights_, _ = session.run(
                        fetches=(self.loss, self.probabilities, self.targets, self.target_weights, self.updates),
                        feed_dict={self.training: True},
                    )

                    self.tracker.add_statistics({
                        Statistic.TRAINING_COST: cost_,
                        Statistic.TRAINING_PROBABILITY: probabilities_,
                        Statistic.TRAINING_TARGET: targets_,
                        TRAINING_TARGET_WEIGHTS: target_weights_,
                    })

            except tf.errors.OutOfRangeError:
                pass

            session.run(validation_iterator)

            try:
                while True:
                    cost_, probabilities_, targets_, target_weights_ = session.run(
                        fetches=(self.loss, self.probabilities, self.targets, self.target_weights),
                    )

                    self.tracker.add_statistics({
                        Statistic.VALIDATION_COST: cost_,
                        Statistic.VALIDATION_PROBABILITY: probabilities_,
                        Statistic.VALIDATION_TARGET: targets_,
                        VALIDATION_TARGET_WEIGHTS: target_weights_,
                    })

            except tf.errors.OutOfRangeError:
                pass

            training_loss = self.tracker.compute_metric(Metric.TRAINING_MEAN_COST)
            train_metric = self.tracker.compute_metric(TRAINING_ROC_AUC)
            validation_metric = self.tracker.compute_metric(VALIDATION_ROC_AUC)

            print(f'Epoch: {epoch}, training loss: {training_loss}, train metric: {train_metric}, validation metric: {validation_metric}')

            try:
                if validation_metric > previous_best_metric:
                    non_improvement_times, previous_best_metric = 0, validation_metric

                    saver.save(session, save_path)

                elif non_improvement_times < early_stopping_threshold:
                    non_improvement_times += 1

                else:
                    print('Stopping after no improvement.')
                    break

            finally:
                self.tracker.finish_epoch()

        saver.restore(session, save_path)

        session.run(test_iterator)

        try:
            while True:
                cost_, probabilities_, targets_, target_weights_ = session.run(
                    fetches=(self.loss, self.probabilities, self.targets, self.target_weights),
                )

                self.tracker.add_statistics({
                    Statistic.TESTING_COST: cost_,
                    Statistic.TESTING_PROBABILITY: probabilities_,
                    Statistic.TESTING_TARGET: targets_,
                    TESTING_TARGET_WEIGHTS: target_weights_,
                })

        except tf.errors.OutOfRangeError:
            pass

        metrics = ModelMetrics(
            test_roc_auc=self.tracker.compute_metric(TESTING_ROC_AUC),
            test_macro_f1=self.tracker.compute_metric(Metric.TESTING_MACRO_F1),
            test_micro_f1=self.tracker.compute_metric(Metric.TESTING_MICRO_F1),
        )

        self.tracker.clear()
        self.tracker.save_model(save_path)

        return metrics

    def multi_step_train_with_early_stopping(
            self,
            data_path: str,
            num_folds: int,
            batch_size: int,
            max_num_epochs: int,
            maximum_non_improvement_epochs: int,
            num_classes: int,
            random_state: int) -> None:

        initializers = tf.global_variables_initializer()
        fold_metrics = []

        splits = (dataset.get_splitted_dataset(data_path, num_classes, random_state) for _ in range(num_folds))

        for fold_index, (train_dataset, validation_dataset, test_dataset) in enumerate(splits):
            train_iterator = self.initialize_iterator(
                iterator=self.iterator,
                generator=train_dataset.get_batches,
                arguments=(batch_size,))

            validation_iterator = self.initialize_iterator(
                iterator=self.iterator,
                generator=validation_dataset.get_batches,
                arguments=(batch_size,))

            test_iterator = self.initialize_iterator(
                iterator=self.iterator,
                generator=test_dataset.get_batches,
                arguments=(batch_size,))

            with tf.Session() as session:
                session.run(initializers)

                metrics = self.train_early_stopping(
                    session=session,
                    train_iterator=train_iterator,
                    validation_iterator=validation_iterator,
                    test_iterator=test_iterator,
                    save_path=self.checkpoint_path,
                    num_epochs=max_num_epochs,
                    early_stopping_threshold=maximum_non_improvement_epochs,
                )

                print(f'\nFold: {fold_index + 1}, ROC AUC: {metrics.test_roc_auc}, Macro F1 Score: {metrics.test_macro_f1}')
                self.tracker.log_metrics({
                    'Fold ROC AUC': metrics.test_roc_auc,
                    'Fold Macro F1 Score': metrics.test_macro_f1,
                    'Fold Micro F1 Score': metrics.test_micro_f1,
                }, fold_index + 1)

                fold_metrics.append(metrics)

        roc_auc_mean = float(np.mean([metrics.test_roc_auc for metrics in fold_metrics]))
        roc_auc_std = float(np.std([metrics.test_roc_auc for metrics in fold_metrics]))
        macro_f1_mean = float(np.mean([metrics.test_macro_f1 for metrics in fold_metrics]))
        macro_f1_std = float(np.std([metrics.test_macro_f1 for metrics in fold_metrics]))

        print(f'\nROC AUC: {roc_auc_mean} (±{roc_auc_std}), F1 Macro: {macro_f1_mean} (±{macro_f1_std})')
        self.tracker.log_metrics({
            'Test ROC AUC Mean': roc_auc_mean,
            'Test ROC AUC Standard Deviation': roc_auc_std,
            'Test Macro F1 Score Mean': macro_f1_mean,
            'Test Macro F1 Score Standard Deviation': macro_f1_std,
        })


def train(
        *,
        data_path: str,
        model_path: str = '.',
        num_nodes: int = 2000,
        num_classes: int = 5,
        max_steps: int = 2,
        sample_size: int = 100,
        learning_rate: float = .001,
        lambda_coefficient: float = 0,
        batch_size: int = 100,
        input_noise_rate: float = 0.,
        dropout_rate: float = 0.,
        layer_size: int = 50,
        layer_size_classifier: int = 250,
        num_attention_heads: int = 10,
        edge_type_embedding_size: int = 50,
        node_embedding_size: Optional[int] = 50,
        max_num_epochs: int = 1000,
        num_folds: int = 10,
        maximum_non_improvement_epochs: int = 5,
        random_state: int = 110069) -> None:
    """
    Trains a link predictor.

    :param data_path: Path to data.
    :param model_path: Path to model.
    :param num_nodes: Number of nodes in the graph.
    :param num_classes: Number of edge types to classify.
    :param max_steps: Maximum random walk steps.
    :param sample_size: Neighbourhood sample size.
    :param learning_rate: Learning rate for the optimizer.
    :param lambda_coefficient: L2 loss coefficient.
    :param batch_size: Batch size for stochastic gradient descend.
    :param input_noise_rate: Node feature drop rate during training.
    :param dropout_rate: Dropout probability.
    :param layer_size: The size of the output for each layer in the neighbour aggregator.
    :param layer_size_classifier: The size of the output for each layer in the classifier.
    :param num_attention_heads: The number of attention heads for a GATAS node.
    :param edge_type_embedding_size: The size of the trainable edge type embeddings.
    :param node_embedding_size: The size of the trainable node embeddings, if any.
    :param max_num_epochs: Maximum number of epochs to train for.
    :param num_folds: Number of runs.
    :param maximum_non_improvement_epochs: Number of epochs for early stopping (patience).
    :param random_state: Random seed for dataset random processes.
    """
    trainer = NodeClassifierTrainer(
        data_path=data_path,
        model_path=model_path,
        num_nodes=num_nodes,
        num_classes=num_classes,
        max_steps=max_steps,
        sample_size=sample_size,
        layer_size=layer_size,
        layer_size_classifier=layer_size_classifier,
        num_attention_heads=num_attention_heads,
        edge_type_embedding_size=edge_type_embedding_size,
        node_embedding_size=node_embedding_size,
        input_noise_rate=input_noise_rate,
        dropout_rate=dropout_rate,
        lambda_coefficient=lambda_coefficient,
        learning_rate=learning_rate,
    )

    trainer.tracker.register_parameters(parameters.get_script_parameters(train))

    with trainer.tracker:
        try:
            trainer.multi_step_train_with_early_stopping(
                data_path=data_path,
                num_folds=num_folds,
                batch_size=batch_size,
                max_num_epochs=max_num_epochs,
                maximum_non_improvement_epochs=maximum_non_improvement_epochs,
                num_classes=num_classes,
                random_state=random_state,
            )

        except Exception as error:
            trainer.tracker.set_tags({'Error': traceback.format_exc()})
            raise error


if __name__ == '__main__':
    defopt.run(train)
