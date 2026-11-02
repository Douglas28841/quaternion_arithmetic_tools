"""Core quaternion operations for four-component real quaternions.

A quaternion is represented as a plain 4-tuple of floats ``(w, x, y, z)``.
Plain tuples are used instead of a class so that quaternions can be passed
across process boundaries, serialized with ``json``, and compared with ``==``
in tests without any custom machinery. The cost is that we cannot attach
methods to the value; every operation is a free function.
"""

from __future__ import annotations

import math
from typing import Tuple

Quaternion = Tuple[float, float, float, float]
Vector = Tuple[float, float, float]

_EPS = 1e-12


def _check_quaternion(q: Quaternion) -> None:
    """Validate that *q* is a 4-tuple of real numbers."""
    if not isinstance(q, tuple) or len(q) != 4:
        raise ValueError("quaternion must be a 4-tuple (w, x, y, z)")
    for component in q:
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            raise TypeError("quaternion components must be real numbers")


def _check_vector(v: Vector) -> None:
    """Validate that *v* is a 3-tuple of real numbers."""
    if not isinstance(v, tuple) or len(v) != 3:
        raise ValueError("vector must be a 3-tuple (x, y, z)")
    for component in v:
        if not isinstance(component, (int, float)) or isinstance(component, bool):
            raise TypeError("vector components must be real numbers")


def hamilton_product(a: Quaternion, b: Quaternion) -> Quaternion:
    """Return the Hamilton product ``a * b``.

    The Hamilton product is non-commutative: ``hamilton_product(a, b)`` and
    ``hamilton_product(b, a)`` differ unless the vector parts are parallel.
    The formula below is the standard one, expanded so that no intermediate
    quaternion objects are allocated.
    """
    _check_quaternion(a)
    _check_quaternion(b)
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def conjugate(q: Quaternion) -> Quaternion:
    """Return the conjugate of *q*, i.e. ``(w, -x, -y, -z)``."""
    _check_quaternion(q)
    w, x, y, z = q
    return (w, -x, -y, -z)


def norm(q: Quaternion) -> float:
    """Return the Euclidean norm of *q*.

    The norm is the square root of the sum of the squares of the four
    components. ``math.hypot`` is used instead of ``math.sqrt`` on the sum of
    squares because it avoids overflow and underflow for extreme inputs.
    """
    _check_quaternion(q)
    return math.hypot(q[0], q[1], q[2], q[3])


def normalize(q: Quaternion) -> Quaternion:
    """Return *q* scaled to unit norm.

    Raises ``ValueError`` for the zero quaternion, which has no direction and
    therefore no unit-norm representative.
    """
    _check_quaternion(q)
    n = norm(q)
    if n <= _EPS:
        raise ValueError("cannot normalize the zero quaternion")
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def rotate_vector(q: Quaternion, v: Vector) -> Vector:
    """Rotate the 3-vector *v* by the unit quaternion *q*.

    The rotation is ``q * v * q_conjugate``, where *v* is embedded as the
    pure quaternion ``(0, x, y, z)``. The caller is responsible for passing a
    unit quaternion; a non-unit *q* scales the result by ``norm(q)**2``. The
    function does not normalize internally because doing so would silently
    hide a caller error and would add a square root to every rotation.
    """
    _check_quaternion(q)
    _check_vector(v)
    w, x, y, z = q
    vx, vy, vz = v

    # Expanded form of q * (0, v) * conjugate(q).
    # t = q * (0, v)
    tw = -x * vx - y * vy - z * vz
    tx = w * vx + y * vz - z * vy
    ty = w * vy - x * vz + z * vx
    tz = w * vz + x * vy - y * vx

    # result = t * conjugate(q)
    rw = tw * w - tx * -x - ty * -y - tz * -z
    rx = tw * -x + tx * w + ty * -z - tz * -y
    ry = tw * -y - tx * -z + ty * w + tz * -x
    rz = tw * -z + tx * -y - ty * -x + tz * w

    # rw should be zero for a unit quaternion; we drop it.
    return (rx, ry, rz)
