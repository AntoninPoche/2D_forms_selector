"""
Create cross forms
"""

import numpy as np
from typing import Dict, List


def create_crosses(
    shapes_dictionary: Dict[str, np.array],
    shape_length: int = 12,
    cross_width: int = 6,
    small_cross_width: int = 2,
):
    """
    Add cross forms to the dictionary of shapes

    Parameters
    ----------
    shapes_dictionary
        Dictionary of already defined shapes.
    shape_length
        Length of one side of a shape.
    cross_width
        Width of the cross.
    small_cross_width
        Width of the small cross.

    Returns
    -------
    shapes_dictionary
        Dictionary of shapes with added cross shapes.
    """
    shapes_size = (shape_length, shape_length)

    # full cross
    assert cross_width % 2 == 0 and cross_width < shape_length
    full_cross = np.full(shapes_size, -1)
    start = (shape_length - cross_width) // 2
    end = shape_length - start
    full_cross[:, start:end] = np.ones((shape_length, cross_width))
    full_cross[start:end, :] = np.ones((cross_width, shape_length))

    assert (
        full_cross.sum()
        == -((shape_length - cross_width) ** 2)
        + 2 * (cross_width * shape_length)
        - cross_width**2
    )
    shapes_dictionary["full cross"] = full_cross

    # small cross
    assert small_cross_width % 2 == 0 and small_cross_width < cross_width
    small_cross = np.full(shapes_size, -1)
    cross_length = shape_length - 2 * (cross_width - 2 * small_cross_width)
    lstart = (shape_length - cross_length) // 2
    lend = shape_length - lstart
    wstart = (shape_length - small_cross_width) // 2
    wend = shape_length - wstart
    small_cross[lstart:lend, wstart:wend] = np.ones((cross_length, small_cross_width))
    small_cross[wstart:wend, lstart:lend] = np.ones((small_cross_width, cross_length))

    assert (
        small_cross.sum()
        == 2 * (2 * (small_cross_width * cross_length) - small_cross_width**2)
        - shape_length**2
    )
    shapes_dictionary["small cross"] = small_cross

    # hollow cross
    hollow_cross = full_cross - (small_cross + 1)
    shapes_dictionary["hollow cross"] = hollow_cross

    return shapes_dictionary
