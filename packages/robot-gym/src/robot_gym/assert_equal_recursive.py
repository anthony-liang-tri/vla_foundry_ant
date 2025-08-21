# TODO: Find a better place to put this function.
import dataclasses as dc
import unittest
from typing import Any

import numpy as np
from pydrake.math import RigidTransform


# TODO: Put this function under test. I.e.; assertRaises, etc.
# with appropriate input.
def assert_equal_recursive(test: unittest.TestCase, a: Any, b: Any, np_tol=0.0):
    """Recursively asserts equality between two objects, a and b."""

    def recurse(a, b, path):
        T = type(a)
        test.assertEqual(type(b), T, path)
        if T is np.ndarray:
            test.assertEqual(a.shape, b.shape, path)
            test.assertEqual(a.dtype, b.dtype, path)
            np.testing.assert_allclose(a, b, err_msg=path, atol=np_tol, rtol=0.0)
        elif isinstance(a, (list, tuple)):
            test.assertEqual(len(a), len(b), path)
            pair_iter = zip(a, b, strict=True)
            for i, (ai, bi) in enumerate(pair_iter):
                recurse(ai, bi, f"{path}[{i}]")
        elif T is dict:
            test.assertEqual(len(a), len(b), path)
            test.assertEqual(a.keys(), b.keys(), path)
            for k, ai in a.items():
                bi = b[k]
                recurse(ai, bi, f"{path}[{repr(k)}]")
        elif dc.is_dataclass(T):
            fields = dc.fields(a)
            test.assertEqual(dc.fields(b), fields)
            for field in fields:
                ai = getattr(a, field.name)
                bi = getattr(b, field.name)
                recurse(ai, bi, f"{path}.{field.name}")
        elif T is RigidTransform:
            recurse(a.GetAsMatrix4(), b.GetAsMatrix4(), path)
        else:
            # Assume a type that has a builtin equality operator.
            test.assertEqual(a, b, path)

    recurse(a, b, "")
