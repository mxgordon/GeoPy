import math
from typing import Union, Tuple

import numpy as np


Scalar = Union[float, int]
vector_raw = Union[Tuple[Scalar, ...], np.ndarray]


class Vector:

    def __init__(self, coordinates: vector_raw):

        self.coordinates = np.array(coordinates, dtype=float)
        self.d = len(self.coordinates)

    def distance(self, origin: vector_raw = None) -> float:
        """
        Compares the distance (computed with the pythagorean theorem) between the origin (if no
        other Point is specified) and the Point that `.distance()` is called on.

        :param origin: The end point for the distance measurement
        :type origin: Union[tuple, list, Point]
        :return: The distance between `self` and `origin`
        :rtype: float
        """
        if origin is None:
            origin = self.__class__([0 for _ in range(self.d)])
        else:
            origin = self.to_class(origin)

        if origin.d != self.d:
            raise ValueError(f"Origin was expected to be {self.d} long, got {origin.d}")

        return math.sqrt(sum(map(lambda x, x2: (x2 - x)**2,  self.coordinates, origin.coordinates)))

    @classmethod
    def to_class(cls, item):
        """
        Attempts to convert `item` to an instance of `cls`, or it ignores it if it is a scalar
        (int or float), if it cannot use `.vector()` and `item` is not a scalar, then it does not
        act upon `item` and it returns it to be dealt with later.

        :param item: the item to convert
        :type item: Any
        :return: the item after the attempted conversion
        :rtype: Any
        """
        if type(item) in (tuple, list, np.ndarray):
            return cls(item)

        if type(item) in (str, dict):
            raise TypeError(f"Expected tuple, list, np.ndarray, Vector2D, Point")

        else:
            try:
                return item.vector()
            except AttributeError:
                return item

    @classmethod
    def check_class(cls, item):
        """
        Makes sure that `item` is an instance of `cls`, and it will throw and error if it is not

        :param item: The object to check
        :type item: Any
        :raises TypeError: If item is not an instance of `cls`
        """
        if not isinstance(item, cls):
            raise TypeError(f"Must be a {cls.__name__} object, got {item.__class__.__name__}")

    def __repr__(self) -> str:
        """
        Returns the representation in a format like "Point(2, 4, ...)"

        :return: the string representation
        :rtype: str
        """
        return f"{self.__class__.__name__}({', '.join(map(lambda x: str(x), self.coordinates))})"

    # ---------- Comparisons -----------

    def __eq__(self, other: vector_raw) -> bool:
        """
        Unlike the other comparison methods, this one checks for absolute equality, so equality
        between all of the coordinates of each vector. Whereas the other methods (<, >, etc.) only
        check the distance to the origin.

        :param other: Other vector object to compare to
        :type other: Point
        :return: If they are equal
        :rtype: bool
        """
        other = self.to_class(other)
        self.check_class(other)
        return (self.coordinates == other.coordinates).all()

    def __gt__(self, other):
        """
        Returns whether it if farther from the origin than `other`.

        :param other: The object it's being compared to
        :type other: Point, Vector2D
        :return: Whether `self` is further from the origin than `other`
        :rtype: bool
        """
        other = self.to_class(other)
        self.check_class(other)
        return self.distance() > other.distance()

    # def

    # -------- Math -------------

    def __add__(self, other: Union[vector_raw, Scalar]):
        """
        Uses the numpy's matrix/scalar support to add matrices of the same size or a matrix and a
        scalar.

        :param other: The other vector/scalar to add
        :type other: Union[vector_raw, Scalar]
        :return: The sum of `other` and `self`
        :rtype: Point
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates + other))

    def __sub__(self, other: Union[vector_raw, Scalar]):
        """
        Uses the numpy's matrix/scalar support to subtract matrices of the same size or a matrix and
        a scalar.

        :param other: The other vector/scalar to subtract
        :type other: Union[vector_raw, Scalar]
        :return: The difference of `other` and `self`
        :rtype: Point
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates - other))

    def __mul__(self, other: Union[vector_raw, Scalar]):
        """
        Uses the numpy's matrix/scalar support to multiply matrices of the same size or a matrix and
        a scalar.

        :param other: The other vector/scalar to multiply
        :type other: Union[vector_raw, Scalar]
        :return: The product of `other` and `self`
        :rtype: Point
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates * other))

    def __truediv__(self, other: Union[vector_raw, Scalar]):
        """
        Uses the numpy's matrix/scalar support to divide matrices of the same size or a matrix and
        a scalar.

        :param other: The other vector/scalar to divide
        :type other: Union[vector_raw, Scalar]
        :return: The quotient of `other` and `self`
        :rtype: Point
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates / other))

    def __floordiv__(self, other: Union[vector_raw, Scalar]):
        """
        Uses the numpy's matrix/scalar support to int divide matrices of the same size or a matrix
        and a scalar.

        :param other: The other vector/scalar to int divide
        :type other: Union[vector_raw, Scalar]
        :return: The int quotient of `other` and `self`
        :rtype: Point
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates // other))

    def __mod__(self, other: Union[vector_raw, Scalar]):
        """
        Uses the numpy's matrix/scalar support to modulus matrices of the same size or a matrix and
        a scalar.

        :param other: The other vector/scalar to mod by
        :type other: Union[vector_raw, Scalar]
        :return: The modulus result of `other` and `self`
        :rtype: Point
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates % other))

    def __pow__(self, power, modulo=None):
        """
        Uses the numpy's matrix/scalar support to modulus matrices of the same size or a matrix and
        a scalar.

        :param power: The other vector/scalar to mod by
        :type power: Union[vector_raw, Scalar]
        :return: The modulus result of `other` and `self`
        :rtype: Point
        """
        power = self.to_class(power)
        power = power.coordinates if isinstance(power, self.__class__) else power
        return self.__class__((self.coordinates.__pow__(power, modulo)))


print(Vector((1, 1)) * (1, 0))
