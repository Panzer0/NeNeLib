from dataclasses import dataclass
from typing import Optional
from numpy import ndarray


@dataclass
class ValueLayerConv:
    values: Optional[ndarray] = None
    delta: Optional[ndarray] = None
