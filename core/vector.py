"""Contains a generic base class for any kind of point, 2D, 3D, polar, etc."""
from __future__ import annotations
from typing import Union, Tuple
import math

import numpy as np


Scalar = Union[float, int]
Vector_raw = Union[Tuple[Scalar, ...], np.ndarray]


class Vector:
    def __init__(self, coordinates: Vector_raw, name=None):

        self.coordinates = np.array(coordinates, dtype=float)
        self.name = name

    def distance(self, origin: Union[Vector_raw, Vector] = None) -> float:
        """
        Compares the distance (computed with the pythagorean theorem) between the origin (if no
        other Vector is specified) and the Vector that `.distance()` is called on.

        :param origin: The end point for the distance measurement
        :type origin: Union[Vector_raw, Vector]
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

    def copy(self, order="C") -> Vector:
        """
        Copy the vector and returns a fresh one without modifying the origina one, this implements
        numpy's `.copy()` to copy the coordinates, then it instantiates a new object.

        :param order: The order to pass into `.copy()`
        :type order: str
        :return: The new object
        :rtype: Vector
        """
        return self.__class__(self.coordinates.copy(order=order))

    @property
    def d(self):
        return len(self.coordinates)

    @classmethod
    def to_class(cls, item, no_scalar=False):
        """
        Attempts to convert `item` to an instance of `cls`, or it ignores it if it is a scalar
        (int or float), if it cannot use `.vector()` and `item` is not a scalar, then it does not
        act upon `item` and it returns it to be dealt with later.

        :param item: the item to convert
        :type item: Any
        :param no_scalar: True if it should throw and error if a scalar is passed in
        :type no_scalar: bool
        :return: the item after the attempted conversion
        :rtype: Any
        """
        if type(item) in (tuple, list, np.ndarray):
            return cls(item)

        elif type(item) in (str, dict):
            raise TypeError(f"Expected tuple, list, np.ndarray, {cls.__name__} not "
                            f"{item.__class__.__name__}")

        else:
            try:
                return item.vector()
            except AttributeError:
                if no_scalar:
                    raise TypeError(f"Expected type tuple, list, np.ndarray or {cls.__name__} not "
                                    f"{item.__class__.__name__}")
                return item

    @classmethod
    def check_class(cls, item):
        """
        Makes sure that `item` is an instance of `cls`, throws and error if it is not.

        :param item: The object to check
        :type item: Any
        :raises TypeError: If item is not an instance of `cls`
        """
        if not isinstance(item, cls):
            raise TypeError(f"Must be a {cls.__name__} object, got {item.__class__.__name__}")

    def __repr__(self) -> str:
        """
        Returns the representation in a format like 'Vector<"name">(2, 4, ...)'.

        :return: the string representation
        :rtype: str
        """
        name = ("<\"" + self.name + "\">") if self.name is not None else ""
        return f"{self.__class__.__name__}{name}" \
               f"({', '.join(map(lambda x: str(x), self.coordinates))})"

    # ----------- Indexing -----------

    def __getitem__(self, item: Union[slice, int, tuple]):
        """
        Return the coordinate(s) at index 'item'.

        :param item: index
        :type item: Union[slice, int, tuple]
        :return: value(s) at `self.coordinates[item]`
        :rtype: Union[Scalar, np.ndarray]
        """
        return self.coordinates.__getitem__(item)

    def __setitem__(self, key: Union[slice, int, tuple], value: Scalar):
        """
        Sets the values at self.coordinates[key] to value, both key and value must be the same
        length, or key needs to have the same "slice equivalent" length.

        :param key: index to set at
        :type key: Union[slice, int, tuple]
        :param value: value to set to
        :type value: Scalar
        """
        self.coordinates.__setitem__(key, value)

    def __delitem__(self, key):
        """
        Cannot delete an item, the vector must keep the same amount of dimensions, if you want to
        make a smaller vector, than make a new vector, but slice the first on when instantiating it
        """
        raise NotImplementedError(f"Cannot change the size of a {self.__class__.__name__}")

    # ---------- Comparisons -----------

    def __eq__(self, other: Union[Vector_raw, Vector]) -> bool:
        """
        Unlike the other comparison methods, this one checks for absolute equality, so equality
        between all of the coordinates of each vector. Whereas the other methods (<, >, etc.) only
        check the distance to the origin.

        :param other: Other vector object to compare to
        :type other: Union[Vector_raw, Vector]
        :return: If they are equal
        :rtype: bool
        """
        other = self.to_class(other)
        self.check_class(other)
        return (self.coordinates.__eq__(other.coordinates)).all()

    def __gt__(self, other: Union[Vector_raw, Vector]):
        """
        Returns whether `self` is farther from the origin than `other`.

        :param other: The object it's being compared to
        :type other: Union[Vector_raw, Vector]
        :return: Whether `self` is further from the origin than `other`
        :rtype: bool
        """
        other = self.to_class(other)
        self.check_class(other)
        return self.distance().__gt__(other.distance())

    def __lt__(self, other: Union[Vector_raw, Vector]):
        """
        Returns whether `self` is farther from the origin than `other`.

        :param other: The object it's being compared to
        :type other: Union[Vector_raw, Vector]
        :return: Whether `self` is closer from the origin than `other`
        :rtype: bool
        """
        other = self.to_class(other)
        self.check_class(other)
        return self.distance().__lt__(other.distance())

    def __le__(self, other: Union[Vector_raw, Vector]):
        """
        Returns whether `self` is closer or equal to the distance from the origin than `other`.

        :param other: The object it's being compared to
        :type other: Union[Vector_raw, Vector]
        :return: Whether `self` is closer or equal to the distance from the origin than `other`
        :rtype: bool
        """
        other = self.to_class(other)
        self.check_class(other)
        return self.distance().__le__(other.distance())

    def __ge__(self, other: Union[Vector_raw, Vector]):
        """
        Returns whether `self` is farther or equal to the distance from the origin than `other`.

        :param other: The object it's being compared to
        :type other: Union[Vector_raw, Vector]
        :return: Whether `self` is farther or equal to the distance from the origin than `other`
        :rtype: bool
        """
        other = self.to_class(other)
        self.check_class(other)
        return self.distance().__ge__(other.distance())

    # -------- Unary -----------

    def __pos__(self):
        """
        Called by the unitary positive (+) operator, returns a new object.

        :return: New object
        :rtype: Vector
        """
        return self.__class__(self.coordinates.__pos__())

    def __neg__(self):
        """
        Called by the unitary negative (-) operator, returns a new object.

        :return: New object
        :rtype: Vector
        """
        return self.__class__(self.coordinates.__neg__())

    def __abs__(self):
        """
        Called by `math.abs()`, returns a new object with all of the coordinates having been
        absolute valued.

        :return: New object
        :rtype: Vector
        """
        return self.__class__(self.coordinates.__abs__())

    def __invert__(self):
        """
        Called by unitary operator inverse (~), , returns a new object.

        :return: New object
        :rtype: Vector
        """
        return self.__class__(self.coordinates.__invert__())

    # -------- Math -------------

    def __add__(self, other: Union[Vector_raw, Scalar, Vector]):
        """
        Uses the numpy's matrix/scalar support to add matrices of the same size or a matrix and a
        scalar.

        :param other: The other vector/scalar to add
        :type other: Union[Vector_raw, Scalar, Vector]
        :return: The sum of `other` and `self`
        :rtype: Vector
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates.__add__(other)))

    def __sub__(self, other: Union[Vector_raw, Vector, Scalar]):
        """
        Uses the numpy's matrix/scalar support to subtract matrices of the same size or a matrix and
        a scalar.

        :param other: The other vector/scalar to subtract
        :type other: Union[Vector_raw, Vector, Scalar]
        :return: The difference of `other` and `self`
        :rtype: Vector
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates.__sub__(other)))

    def __mul__(self, other: Union[Vector_raw, Vector, Scalar]):
        """
        Uses the numpy's matrix/scalar support to multiply matrices of the same size or a matrix and
        a scalar.

        :param other: The other vector/scalar to multiply
        :type other: Union[Vector_raw, Vector, Scalar]
        :return: The product of `other` and `self`
        :rtype: Vector
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates.__mul__(other)))

    def __truediv__(self, other: Union[Vector_raw, Vector, Scalar]):
        """
        Uses the numpy's matrix/scalar support to divide matrices of the same size or a matrix and
        a scalar.

        :param other: The other vector/scalar to divide
        :type other: Union[Vector_raw, Vector, Scalar]
        :return: The quotient of `other` and `self`
        :rtype: Vector
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates.__truediv__(other)))

    def __floordiv__(self, other: Union[Vector_raw, Vector, Scalar]):
        """
        Uses the numpy's matrix/scalar support to int divide matrices of the same size or a matrix
        and a scalar.

        :param other: The other vector/scalar to int divide
        :type other: Union[Vector_raw, Vector, Scalar]
        :return: The int quotient of `other` and `self`
        :rtype: Vector
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates.__floordiv__(other)))

    def __mod__(self, other: Union[Vector_raw, Vector, Scalar]):
        """
        Uses the numpy's matrix/scalar support to modulus matrices of the same size or a matrix and
        a scalar.

        :param other: The other vector/scalar to mod by
        :type other: Union[Vector_raw, Vector, Scalar]
        :return: The modulus result of `other` and `self`
        :rtype: Vector
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.__class__((self.coordinates.__mod__(other)))

    def __pow__(self, power: Union[Vector_raw, Vector, Scalar], modulo=None):
        """
        Uses the numpy's matrix/scalar support to modulus matrices of the same size or a matrix and
        a scalar.

        :param power: The other vector/scalar to mod by
        :type power: Union[Vector_raw, Vector, Scalar]
        :return: The modulus result of `other` and `self`
        :rtype: Vector
        """
        power = self.to_class(power)
        power = power.coordinates if isinstance(power, self.__class__) else power
        return self.__class__((self.coordinates.__pow__(power, modulo)))

    def __matmul__(self, other: Union[Vector_raw, Vector, Scalar]):
        """
        Uses the numpy's matrix/scalar support to modulus matrices of the same size or a matrix and
        a scalar.

        :param other: The other vector/scalar to mod by
        :type other: Union[Vector_raw, Vector, Scalar]
        :return: The the matrix multiply result of `other` and `self`
        :rtype: float
        """
        other = self.to_class(other)
        other = other.coordinates if isinstance(other, self.__class__) else other
        return self.coordinates.__matmul__(other)
