from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass(frozen=True)
class LinearEquation:
    point1: tuple[float, float]
    point2: tuple[float, float] | None = None
    slope: float = dataclasses.field(init=False)
    slope_: dataclasses.InitVar[float | None] = None

    def __post_init__(self, slope_: float) -> None:
        if self.point2 is None and slope_ is None:
            raise ValueError("Either 'point2' or 'slope_' must be set.")
        elif self.point2 is None:
            # Reference: https://stackoverflow.com/a/54119384
            super().__setattr__('slope', slope_)
        elif self.point1 == self.point2:
            raise ValueError("'point1' and 'point2' are the same.")
        elif self.point2[0] == self.point1[0]:
            super().__setattr__('slope', np.inf)
        elif self.point2[1] == self.point1[1]:
            super().__setattr__('slope', 0)
        else:
            super().__setattr__(
                'slope',
                (self.point2[1] - self.point1[1])
                / (self.point2[0] - self.point1[0]),
            )

    @property
    def negative_reciprocal(self) -> float:
        return np.inf if math.isclose(self.slope, 0) else -1 / self.slope

    @property
    def is_line_segment(self) -> bool:
        return self.point2 is not None

    def get_point_of_intersection(
        self, other: LinearEquation
    ) -> tuple[float, float]:
        if math.isclose(self.slope, other.slope):
            raise ValueError("Slopes are the same.")
        x: float
        y: float
        if math.isclose(self.slope, np.inf):
            x = self.point1[0]
            y = other.slope * (x - other.point1[0]) + other.point1[1]
        else:
            if math.isclose(other.slope, np.inf):
                x = other.point1[0]
            else:
                x = (
                    other.point1[1]
                    - self.point1[1]
                    + self.slope * self.point1[0]
                    - other.slope * other.point1[0]
                ) / (self.slope - other.slope)
            y = self.slope * (x - self.point1[0]) + self.point1[1]
        return (x, y)
