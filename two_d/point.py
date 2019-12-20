"""Contains the class for 2D euclidean points that are defined with an X and Y coordinate """
from __future__ import annotations
from typing import Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from core.vector import Scalar, Vector


point_raw = Union[Tuple[Scalar, Scalar], np.ndarray]


class Point(Vector):
    def __init__(self, x_y: point_raw, name=None):
        if len(x_y) != 2:
            raise ValueError(f"Must have 2 coordinates not {len(x_y)}")

        super().__init__(x_y, name=name)

    def vector(self) -> Vector:
        """
        Converts the point to a vector, this just exists if you want to have a base object, because
        Point is mostly a wrapper around Vector that just has `.x` and `.y` parameters.

        :return: The Vector instance that it converted to
        :rtype: Vector
        """
        return Vector(self.coordinates)

    @staticmethod
    def from_vector(vector: Vector) -> Point:
        """
        Converts a Vector object to an Point object, the object must have only two dimensions
        (x, y).

        :param vector: The vector object to convert from
        :type vector: Vector
        :return: The new Point object
        :rtype: Point
        """
        return Point(vector.coordinates)

    def flip_x(self):
        """
        Returns a new Point that is flipped over the x axis, it achieves this by inverting the y
        coordinate.

        :return: The new Point
        :rtype: Point
        """
        return Point((self.x, -self.y))

    def flip_y(self):
        """
        Returns a new Point that is flipped over the y axis, it achieves this by inverting the x
        coordinate.

        :return: The new Point
        :rtype: Point
        """
        return Point((-self.x, self.y))

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    def plot(self, color: str = None, **kwargs):
        """
        This plots `self` on a figure, and if the color is not specified then the `None` will prompt
        a random color (of 18 possibilities). This will use the the global figure if none are
        specified, and it can also plot on a subplot if specified.

        :param color: Color to make the point
        :type color: str
        """
        plt.scatter(*self.coordinates, color=color, **kwargs)

    @staticmethod
    def show_plot():
        plt.show()
