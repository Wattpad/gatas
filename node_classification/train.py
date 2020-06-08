import os
import traceback
from typing import Tuple, Callable, NamedTuple, List, Optional

import defopt
import numpy as np
import tensorflow as tf

from framework.common import parameters
from framework.dataset import io
from framework.trackers.aggregator import Aggregation, Statistic
from framework.trackers.metrics import MetricFunctions, Metric
from framework.trackers.tracker_mlflow import MLFlowTracker
from node_classification import dataset
from node_classification.model import NodeClassifier


class ModelMetrics(NamedTuple):
    test_accuracy: float
    test_micro_f1: float


class NodeClassifierTrainer:
    def __init__(
            self,
            data_path: str,
            model_path: str,
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

        # num_classes = np.max(io.load_npy(os.path.join(data_path, 'class_ids.npy'))) + 1
        num_classes = io.load_npy(os.path.join(data_path, 'class_ids.npy')).shape[1]
        node_embeddings = io.load_npy(os.path.join(data_path, 'node_embeddings.npy'), mmap_mode='r')

        self.iterator = self.create_iterator(num_classes)
        self.training = tf.placeholder_with_default(False, shape=(), name='training')

        self.model = NodeClassifier(
            path=data_path,
            num_nodes=node_embeddings.shape[0],
            num_classes=num_classes,
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
            node_features=node_embeddings,
        )

        self.anchor_indices, self.targets = self.iterator.get_next()
        self.training = tf.placeholder_with_default(False, shape=(), name='training')
        self.logits, self.predictions = self.model(self.anchor_indices, self.training)
        self.loss = self.model.calculate_loss(self.logits, self.targets)
        self.updates = self.model.get_clipped_gradient_updates(self.loss, optimizer=tf.contrib.opt.NadamOptimizer(learning_rate))

        self.tracker = MLFlowTracker('node-classifier', aggregators=[
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
                [Statistic.TRAINING_TARGET, Statistic.TRAINING_PREDICTION],
                lambda x, y: MetricFunctions.f1_score(x, y, average='micro')),
            Aggregation(
                Metric.VALIDATION_MICRO_F1,
                [Statistic.VALIDATION_TARGET, Statistic.VALIDATION_PREDICTION],
                lambda x, y: MetricFunctions.f1_score(x, y, average='micro')),
            Aggregation(
                Metric.TESTING_MICRO_F1,
                [Statistic.TESTING_TARGET, Statistic.TESTING_PREDICTION],
                lambda x, y: MetricFunctions.f1_score(x, y, average='micro')),
            Aggregation(
                Metric.TRAINING_ACCURACY,
                [Statistic.TRAINING_TARGET, Statistic.TRAINING_PREDICTION],
                lambda x, y: MetricFunctions.accuracy(x, y)),
            Aggregation(
                Metric.VALIDATION_ACCURACY,
                [Statistic.VALIDATION_TARGET, Statistic.VALIDATION_PREDICTION],
                lambda x, y: MetricFunctions.accuracy(x, y)),
            Aggregation(
                Metric.TESTING_ACCURACY,
                [Statistic.TESTING_TARGET, Statistic.TESTING_PREDICTION],
                lambda x, y: MetricFunctions.accuracy(x, y)),
        ])

        self.tracker.set_tags({
            'Tier': 'Development',
            'Problem': 'Node Classification',
        })

    @staticmethod
    def create_iterator(num_classes: int) -> tf.data.Iterator:
        iterator = tf.data.Iterator.from_structure(
            output_types=NodeClassifier.get_schema(num_classes).get_types(),
            output_shapes=NodeClassifier.get_schema(num_classes).get_shapes(),
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
                args=arguments) \
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
                    cost_, predictions_, targets_, _ = session.run(
                        fetches=(self.loss, self.predictions, self.targets, self.updates),
                        feed_dict={self.training: True},
                    )

                    self.tracker.add_statistics({
                        Statistic.TRAINING_COST: cost_,
                        Statistic.TRAINING_PREDICTION: predictions_,
                        Statistic.TRAINING_TARGET: targets_,
                    })

            except tf.errors.OutOfRangeError:
                pass

            session.run(validation_iterator)

            try:
                while True:
                    cost_, predictions_, targets_ = session.run(fetches=(self.loss, self.predictions, self.targets))

                    self.tracker.add_statistics({
                        Statistic.VALIDATION_COST: cost_,
                        Statistic.VALIDATION_PREDICTION: predictions_,
                        Statistic.VALIDATION_TARGET: targets_,
                    })

            except tf.errors.OutOfRangeError:
                pass

            training_loss = self.tracker.compute_metric(Metric.TRAINING_MEAN_COST)
            train_metric = self.tracker.compute_metric(Metric.TRAINING_MICRO_F1)
            validation_metric = self.tracker.compute_metric(Metric.VALIDATION_MICRO_F1)

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
                cost_, predictions_, targets_ = session.run(fetches=(self.loss, self.predictions, self.targets))

                self.tracker.add_statistics({
                    Statistic.TESTING_COST: cost_,
                    Statistic.TESTING_PREDICTION: predictions_,
                    Statistic.TESTING_TARGET: targets_,
                })

        except tf.errors.OutOfRangeError:
            pass

        evaluation_metrics = ModelMetrics(
            test_accuracy=self.tracker.compute_metric(Metric.TESTING_ACCURACY),
            test_micro_f1=self.tracker.compute_metric(Metric.TESTING_MICRO_F1),
        )

        self.tracker.clear()
        self.tracker.save_model(save_path)

        return evaluation_metrics

    def multi_step_train_with_early_stopping(
            self,
            num_folds: int,
            batch_size: int,
            max_num_epochs: int,
            maximum_non_improvement_epochs: int) -> None:

        initializers = tf.global_variables_initializer()
        fold_metrics = []

        train_dataset, validation_dataset, test_dataset = dataset.get_splitted_dataset(self.data_path)

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

        for fold_index in range(num_folds):
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

                print(f'\nFold: {fold_index + 1}, Test Accuracy: {metrics.test_accuracy}, Micro F1 Score: {metrics.test_micro_f1}')
                self.tracker.log_metrics({
                    'Fold Accuracy': metrics.test_accuracy,
                    'Fold Micro F1 Score': metrics.test_micro_f1,
                }, fold_index + 1)

                fold_metrics.append(metrics)

        accuracy_mean = float(np.mean([metrics.test_accuracy for metrics in fold_metrics]))
        accuracy_std = float(np.std([metrics.test_accuracy for metrics in fold_metrics]))
        micro_f1_mean = float(np.mean([metrics.test_micro_f1 for metrics in fold_metrics]))
        micro_f1_std = float(np.std([metrics.test_micro_f1 for metrics in fold_metrics]))

        print(f'\nAccuracy: {accuracy_mean} (±{accuracy_std}), F1 Micro: {micro_f1_mean} (±{micro_f1_std})')
        self.tracker.log_metrics({
            'Test Accuracy Mean': accuracy_mean,
            'Test Accuracy Standard Deviation': accuracy_std,
            'Test Micro F1 Score Mean': micro_f1_mean,
            'Test Micro F1 Score Standard Deviation': micro_f1_std,
        })


def train(
        *,
        data_path: str,
        model_path: str = '.',
        max_steps: int = 3,
        sample_size: int = 500,
        learning_rate: float = 0.001,
        lambda_coefficient: float = 0,
        batch_size: int = 100,
        input_noise_rate: float = 0.0,
        dropout_rate: float = 0.0,
        layer_size: int = 50,
        layer_size_classifier: int = 256,
        num_attention_heads: int = 10,
        edge_type_embedding_size: int = 5,
        node_embedding_size: Optional[int] = None,
        max_num_epochs: int = 1000,
        num_folds: int = 10,
        maximum_non_improvement_epochs: int = 10) -> None:
    """
    Trains a node classifier.

    :param data_path: Path to data.
    :param model_path: Path to model.
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
    """
    trainer = NodeClassifierTrainer(
        data_path=data_path,
        model_path=model_path,
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
                num_folds=num_folds,
                batch_size=batch_size,
                max_num_epochs=max_num_epochs,
                maximum_non_improvement_epochs=maximum_non_improvement_epochs,
            )

        except Exception as error:
            trainer.tracker.set_tags({'Error': traceback.format_exc()})
            raise error


if __name__ == '__main__':
    defopt.run(train)
