from typing import Mapping, Optional

from framework.trackers import Numeric


class Tracker:
    def set_tags(self, tags: Mapping[str, str]) -> None:
        raise NotImplementedError

    def register_parameters(self, parameters: Mapping[str, Numeric]) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: Mapping[str, Numeric], step: Optional[int] = None) -> None:
        raise NotImplementedError

    def save_model(self, path: str) -> None:
        raise NotImplementedError

    def finish_epoch(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> 'Tracker':
        raise NotImplementedError

    def __exit__(self, context_type, value, traceback) -> None:
        raise NotImplementedError
