from typing import Tuple, Union
from numpy import np

from vector import Scalar, Vector


point_raw = Union[Tuple[Scalar, Scalar], np.ndarray]


class Point(Vector):

    def __init__(self, x_y: point_raw, name=None):
        if len(x_y) != 2:
            raise ValueError(f"Must have 2 coordinates not {len(x_y)}")

        self.x, self.y = x_y
        super().__init__(x_y, name=name)

    def vector(self):
        return Vector(self.coordinates)

    @staticmethod
    def from_vector(point: Vector):
        return Point(point.coordinates)
