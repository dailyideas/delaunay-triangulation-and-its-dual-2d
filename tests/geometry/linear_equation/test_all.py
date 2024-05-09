import math

from delaunay_triangulation_and_its_dual_2d.geometry import linear_equation
import numpy as np
import pytest


def test_slope():
    assert math.isclose(
        linear_equation.LinearEquation(point1=(0, 0), point2=(1, -2)).slope, -2
    )
    assert math.isclose(
        linear_equation.LinearEquation(point1=(0, 0), point2=(2, 1)).slope, 0.5
    )
    assert math.isclose(
        linear_equation.LinearEquation(point1=(0, 0), slope_=1).slope, 1
    )
    assert math.isclose(
        linear_equation.LinearEquation(point1=(0, 0), point2=(6, 0)).slope, 0
    )
    assert math.isclose(
        linear_equation.LinearEquation(point1=(0, 0), point2=(0, 6)).slope,
        np.inf,
    )
    assert math.isclose(
        linear_equation.LinearEquation(point1=(0, 0), slope_=np.inf).slope,
        np.inf,
    )

    with pytest.raises(ValueError):
        linear_equation.LinearEquation(point1=(0, 0))
    with pytest.raises(ValueError):
        linear_equation.LinearEquation(point1=(0, 0), point2=(0, 0))
    with pytest.raises(TypeError):
        linear_equation.LinearEquation(point2=(0, 0))
    with pytest.raises(TypeError):
        linear_equation.LinearEquation(point2=(0, 0), slope_=0)


@pytest.mark.parametrize(
    "point1, point2, target",
    [
        ((0, 0), (1, -2), 0.5),
        ((0, 0), (2, 1), -2),
        ((0, 0), (6, 0), np.inf),
        ((0, 0), (0, 6), 0),
    ],
)
def test_negative_reciprocal(
    point1: tuple[float, float], point2: tuple[float, float], target: float
):
    assert math.isclose(
        linear_equation.LinearEquation(
            point1=point1, point2=point2
        ).negative_reciprocal,
        target,
    )


@pytest.mark.parametrize(
    "line1_point1, line1_point2, line2_point1, line2_point2, target",
    [
        ((0, 0), (1, -2), (0, 0), (2, 1), (0, 0)),
        ((0, 0), (2, 1), (0, 6), (6, 6), (12, 6)),
        ((0, 0), (2, 1), (6, 0), (6, 6), (6, 3)),
        ((0, 6), (6, 6), (0, 0), (2, 1), (12, 6)),
        ((6, 0), (6, 6), (0, 0), (2, 1), (6, 3)),
    ],
)
def test_get_point_of_intersection(
    line1_point1: tuple[float, float],
    line1_point2: tuple[float, float],
    line2_point1: tuple[float, float],
    line2_point2: tuple[float, float],
    target: tuple[float, float],
):
    line1 = linear_equation.LinearEquation(
        point1=line1_point1, point2=line1_point2
    )
    line2 = linear_equation.LinearEquation(
        point1=line2_point1, point2=line2_point2
    )
    intersection = line1.get_point_of_intersection(other=line2)
    assert all(math.isclose(x, y) for x, y in zip(intersection, target))
