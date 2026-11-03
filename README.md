# quaternion_arithmetic_tools

Hamilton product, conjugate, norm, normalization, and vector rotation for
four-component real quaternions represented as tuples `(w, x, y, z)`.

```python
from quaternion_arithmetic_tools import (
    conjugate,
    hamilton_product,
    norm,
    normalize,
    rotate_vector,
)

a = (1.0, 2.0, 3.0, 4.0)
b = (5.0, 6.0, 7.0, 8.0)

print(hamilton_product(a, b))   # (-60.0, 12.0, 30.0, 24.0)
print(conjugate(a))             # (1.0, -2.0, -3.0, -4.0)
print(norm(a))                  # 5.477225575051661
print(normalize(a))             # (0.182574..., 0.365148..., 0.547722..., 0.730296...)

# Rotate (1, 0, 0) by +90 degrees about the z axis.
s = 0.7071067811865476
print(rotate_vector((s, 0.0, 0.0, s), (1.0, 0.0, 0.0)))  # (0.0, 1.0, 0.0)
```

## Why this exists

Quaternions are small enough that a dedicated class often costs more than it
saves: it adds allocation, serialization friction, and a custom equality
story. This library keeps the value as a plain `(w, x, y, z)` tuple and exposes
free functions. The trade-off is that there is no method syntax and no
operator overloading; in exchange, quaternions work with `json`, tuple
unpacking, and `==` out of the box.

## The awkward edge

`rotate_vector` does **not** normalize its quaternion argument. Passing a
non-unit quaternion scales the rotated vector by `norm(q)**2` instead of
rotating it. This is deliberate: normalizing inside the rotation would hide a
caller error and add a square root to every call. Call `normalize` first if
you are not certain the quaternion is unit length.

`normalize` raises `ValueError` for the zero quaternion, which has no unit-norm
representative. `hamilton_product` and `conjugate` validate the shape and
types of their arguments and raise `ValueError` or `TypeError` for malformed
input.
