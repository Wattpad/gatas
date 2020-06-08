import glob
from typing import Mapping, Collection, Optional

import mlflow

from framework.trackers import Numeric, PrimitiveType
from framework.trackers.aggregator import Aggregation, MetricAggregator
from framework.trackers.tracker import Tracker


class MLFlowTracker(MetricAggregator, Tracker):
    def __init__(
            self,
            experiment_name: str,
            tracking_uri: Optional[str] = None,
            aggregators: Optional[Collection[Aggregation]] = None):

        super().__init__(aggregators)

        self.step = 0

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        mlflow.set_tags(tags)

    def register_parameters(self, parameters: Mapping[str, PrimitiveType]) -> None:
        mlflow.log_params({self.to_str(key): value for key, value in parameters.items()})

    def log_metrics(self, metrics: Mapping[str, Numeric], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step)

    def save_model(self, path: str) -> None:
        for file_path in glob.glob(path + '*'):
            mlflow.log_artifact(file_path)

    def finish_epoch(self) -> None:
        metrics = {self.to_str(key): value for key, value in self.get_metrics().items()}

        self.log_metrics(metrics, self.step)

        self.flush()

        self.step += 1

    def __enter__(self) -> 'MLFlowTracker':
        return self

    def __exit__(self, context_type, value, traceback) -> None:
        mlflow.end_run()
