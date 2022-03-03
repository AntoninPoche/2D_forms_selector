"""
Create square forms
"""

import numpy as np
from typing import Dict, List


def create_squares(
    shapes_dictionary: Dict[str, np.array],
    shape_length: int = 12,
    small_square_width: int = 6,
):
    """
    Add square forms to the dictionary of shapes

    Parameters
    ----------
    shapes_dictionary
        Dictionary of already defined shapes.
    shape_length
        Length of one side of a shape.
    small_square_width
        Width of the small square.

    Returns
    -------
    shapes_dictionary
        Dictionary of shapes with added square shapes.
    """
    shapes_size = (shape_length, shape_length)

    # full square
    full_square = np.ones(shapes_size)

    assert full_square.sum() == shape_length**2
    shapes_dictionary["full square"] = full_square

    # small square
    assert small_square_width % 2 == 0 and small_square_width < shape_length
    small_square = np.full(shapes_size, -1)
    width = (shape_length - small_square_width) // 2
    begin = width
    end = shape_length - width
    length = end - begin
    small_square[begin:end, begin:end] = np.ones((length, length))

    assert small_square.sum() == 2 * length**2 - shape_length**2
    shapes_dictionary["small square"] = small_square

    # hollow square
    hollow_square = -small_square

    assert hollow_square.sum() == shape_length**2 - 2 * length**2
    shapes_dictionary["hollow square"] = hollow_square

    return shapes_dictionary
