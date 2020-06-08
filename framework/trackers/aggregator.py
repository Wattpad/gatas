from enum import Enum
from typing import Mapping, Union, MutableMapping, Collection, Tuple, List, Optional, NamedTuple

import numpy as np

from framework.trackers import Numeric, Aggregator
from framework.trackers.metrics import Metric


class Statistic(Enum):
    TRAINING_TARGET = 'Training Target'
    VALIDATION_TARGET = 'Validation Target'
    TESTING_TARGET = 'Testing Target'

    TRAINING_PREDICTION = 'Training Prediction'
    VALIDATION_PREDICTION = 'Validation Prediction'
    TESTING_PREDICTION = 'Testing Prediction'

    TRAINING_PROBABILITY = 'Training Probability'
    VALIDATION_PROBABILITY = 'Validation Probability'
    TESTING_PROBABILITY = 'Testing Probability'

    TRAINING_COST = 'Training Cost'
    VALIDATION_COST = 'Validation Cost'
    TESTING_COST = 'Testing Cost'

    BETA = 'Beta'


MetricKey = Union[str, Metric]
StatisticKey = Union[str, Statistic]
StatisticValue = Union[List[Numeric], Numeric, np.ndarray]


class Aggregation(NamedTuple):
    metric: MetricKey
    statistics: Collection[StatisticKey]
    function: Aggregator


class MetricAggregator:
    def __init__(self, aggregators: Optional[Collection[Aggregation]] = None):
        self.aggregators = {}  # type: MutableMapping[MetricKey, Tuple[Collection[StatisticKey], Aggregator]]
        self.state = {}  # type: MutableMapping[StatisticKey, List[Numeric]]

        if aggregators:
            self.register_aggregators(aggregators)

    @staticmethod
    def to_numeric_list(value: StatisticValue) -> List[Numeric]:
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, List):
            return value
        else:
            return [value]

    @staticmethod
    def to_str(value: MetricKey) -> str:
        return value.value if isinstance(value, Enum) else value

    def register_aggregator(self, metric: MetricKey, statistics: Collection[StatisticKey], function: Aggregator) -> None:
        self.aggregators[metric] = (statistics, function)

        for statistic in statistics:
            if statistic not in self.state:
                self.state[statistic] = []

    def register_aggregators(self, aggregators: Collection[Aggregation]):
        for aggregator in aggregators:
            self.register_aggregator(aggregator.metric, aggregator.statistics, aggregator.function)

    def add_statistic(self, key: StatisticKey, value: StatisticValue) -> None:
        self.state[key].extend(self.to_numeric_list(value))

    def add_statistics(self, statistics: Mapping[StatisticKey, StatisticValue]) -> None:
        for key, value in statistics.items():
            self.add_statistic(key, value)

    def _compute_metric(self, key: MetricKey) -> Optional[Numeric]:
        statistics, function = self.aggregators[key]

        if not any(len(self.state[statistic]) != 0 for statistic in statistics):
            return None

        return function(*(np.array(self.state[statistic]) for statistic in statistics))

    def compute_metric(self, key: MetricKey) -> Numeric:
        value = self._compute_metric(key)

        assert value is not None

        return value

    def compute_metrics(self, keys: Collection[MetricKey]) -> Mapping[MetricKey, Numeric]:
        return {
            key: value
            for key, value
            in ((key, self._compute_metric(key)) for key in keys)
            if value is not None
        }

    def get_metrics(self) -> Mapping[MetricKey, Numeric]:
        return self.compute_metrics(self.aggregators.keys())

    def clear(self) -> None:
        for sequence in self.state.values():
            sequence.clear()

    def flush(self) -> Mapping[MetricKey, Numeric]:
        metrics = self.compute_metrics(self.aggregators.keys())

        self.clear()

        return metrics
