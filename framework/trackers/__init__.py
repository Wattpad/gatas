from typing import Union, Callable

Numeric = Union[int, float]
PrimitiveType = Union[int, float, str, bool]
Aggregator = Callable[..., Numeric]
