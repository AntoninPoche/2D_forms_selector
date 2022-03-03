"""
Functions to build the dataset,
Calls for the forms scripts
"""

from itertools import product
import numpy as np
from typing import Dict, Any, List

from .forms import create_squares
from .forms import create_crosses


def create_dataset(
    nb_samples: int = 1000,
    shape_length: int = 12,
    margin: int = 3,
    max_shift: int = 0,
) -> (np.ndarray, np.array, np.array, np.array, np.array, List[str]):
    """
    Create a dataset of 2D forms.
    Each image contain 4 shapes and one is highlighted by a selector.
    The selected shape is the classification target.
    The different forms are defined in the forms directory.

    Parameters
    ----------
    nb_samples
        Number of samples in the dataset.
    shape_length
        Length of the window containing the shape. (In pixels)
    margin
        Distance between the window containing the shape and the border of the image.
        Should be at least 3 for the selection highlighting to be visible. (In pixels)
    max_shift
        Number of pixels the shape can be shifted.
        The shape can be shifted in the four directions.
        A max shift of 0 means that only one position is possible.
        For 1 their are 9 positions and 2 their are 25.
        The margin should be increased in accordance with max_shift.

    Returns
    -------
    image_data
        Dataset of images.
    raw_data
        Array of raw values (nb_samples x 4).
        Values are between 0 and nb_shapes - 1.
    selected_positions
        Array of indices between 0 and 3.
        It selects the values in the raw data to put as the target.
    target
        Array of labels between 0 and nb_shapes - 1.
    target_onehot
        One-hot encoding array of the target.
    shapes_names
        List of shapes names, used to associated a index to the name of the shape.
    shapes_dictionary
        Dictionary of all possible shapes, the key is the shape name.
    """
    assert shape_length % 2 == 0

    # create possible shapes dictionary
    shapes_dictionary = {}
    # add squares
    shapes_dictionary = create_squares(
        shapes_dictionary,
        shape_length=shape_length,
        small_square_width=shape_length // 2,
    )
    # add crosses
    shapes_dictionary = create_crosses(
        shapes_dictionary,
        shape_length=shape_length,
        cross_width=shape_length // 2,
        small_cross_width=shape_length // 6,
    )

    shapes_names = list(shapes_dictionary.keys())
    nb_shapes = len(shapes_names)

    # generate raw data
    raw_data, selected_positions, target, target_onehot = generate_raw_dataset(
        nb_samples=nb_samples,
        nb_shapes=nb_shapes,
        seed=0,
    )

    # define the arguments value to transform raw data into images
    aggregation_kwargs = def_aggregate_arguments(
        shape_length,
        margin,
        max_shift,
    )

    # define the selector filter
    aggregation_kwargs["selector"] = def_selector(
        shape_length,
        margin,
    )

    # transform raw data into images
    image_data = raw2images(
        raw_data,
        selected_positions,
        shapes_dictionary,
        shapes_names,
        aggregation_kwargs,
    )

    return image_data, raw_data, selected_positions, target, target_onehot, shapes_names, shapes_dictionary


def generate_raw_dataset(
    nb_samples: int,
    nb_shapes: int,
    seed: int = 0,
) -> (np.array, np.array, np.array, np.array):
    """
    Generates raw values for the dataset.
    Used to create a dataset.

    Parameters
    ----------
    nb_samples
        Number of samples to generate.
    nb_shapes
        Number of different values available for the dataset.
    seed
        Numpy seed to be able to obtain the same dataset.

    Returns
    -------
    raw_data
        Array of raw values (nb_samples x 4).
        Values are between 0 and nb_shapes - 1.
    selected_positions
        Array of indices between 0 and 3.
        It selects the values in the raw data to put as the target.
    target
        Array of labels between 0 and nb_shapes - 1.
    target_onehot
        One-hot encoding array of the target.
    """
    np.random.seed(seed)

    raw_data = np.random.randint(nb_shapes, size=(nb_samples, 4))
    selected_positions = np.random.randint(4, size=(nb_samples, 1))

    target = np.take_along_axis(raw_data, selected_positions, axis=1)
    target = target.transpose()[0]

    target_onehot = np.zeros((target.size, target.max() + 1))
    target_onehot[np.arange(target.size), target] = 1

    return raw_data, selected_positions, target, target_onehot


def def_aggregate_arguments(
    shape_length: int,
    margin: int,
    max_shift: int,
) -> Dict[str, Any]:
    """
    Define the different argument arguments necessary to transform raw data into images.
    Those arguments are stocked in a dictionary, they are global to all images,
    thus they are only computed once.

    Parameters
    ----------
    shape_length
        Length of the side of the window

    Returns
    -------
    """
    global_length = 2 * (2 * margin + shape_length)
    global_shape = (global_length, global_length)

    # set starts and ends for angles, parts and shapes
    parts_starts = [0, global_length // 2]
    parts_ends = [global_length // 2, global_length]
    shape_starts = [margin, global_length // 2 + margin]
    shape_ends = [margin + shape_length, global_length // 2 + margin + shape_length]

    positions = list(product([0, 1], repeat=2))

    # set possible shifts
    possible_shifts = list(range(-max_shift, max_shift + 1))

    return {
        "global_shape": global_shape,
        "parts_limits": (parts_starts, parts_ends),
        "shape_limits": (shape_starts, shape_ends),
        "positions": positions,
        "possible_shifts": possible_shifts,
    }


def def_selector(
    shape_length: int,
    margin: int,
) -> np.array:
    """

    Parameters
    ----------

    Returns
    -------
    selector
    """
    angle_length = 2 * (margin - 1)
    angle = 2 * np.tri(angle_length)
    global_length = 2 * (2 * margin + shape_length)
    positions = list(product([0, 1], repeat=2))

    # define beginning and ending index of the angle
    angle_starts = [0, global_length // 2 - angle_length]
    angle_ends = [angle_length, global_length // 2]

    # create selector
    rotations = [0, 3, 1, 2]
    selector = -np.ones((global_length // 2, global_length // 2))
    for i in range(4):
        x_pos, y_pos = positions[i]
        selector[
            angle_starts[x_pos] : angle_ends[x_pos],
            angle_starts[y_pos] : angle_ends[y_pos],
        ] = (
            np.rot90(angle, 3 + rotations[i]) - 1
        )

    return selector


def raw2images(
    raw_data: np.array,
    selected: list(),
    shapes_dictionary: Dict[str, np.array],
    shapes_names: List[str],
    aggregation_kwargs: Dict[str, Any],
) -> np.ndarray:
    """

    Parameters
    ----------

    Returns
    -------
    """
    nb_samples = raw_data.shape[0]

    dataset = [
        aggregate_parts(
            shapes=[
                shapes_dictionary[shapes_names[shape_id]]
                for shape_id in raw_data[sample_id, :]
            ],
            selected=selected[sample_id],
            kwargs=aggregation_kwargs,
        )
        for sample_id in range(nb_samples)
    ]
    data = np.expand_dims(dataset, axis=-1)
    return data


def aggregate_parts(
    shapes: List[np.array], selected: int, kwargs: Dict[str, Any]
) -> np.array:
    """

    Parameters
    ----------

    Returns
    -------
    """
    # extract arguments
    # TODO put them directly as arguments
    global_shape = kwargs["global_shape"]
    selector = kwargs["selector"]
    parts_starts, parts_ends = kwargs["parts_limits"]
    shape_starts, shape_ends = kwargs["shape_limits"]
    positions = kwargs["positions"]
    possible_shifts = kwargs["possible_shifts"]

    # to aggregate for images
    image = -np.ones(global_shape)
    for i, shape in enumerate(shapes):
        x_pos, y_pos = positions[i]
        if i == selected:
            image[
                parts_starts[x_pos] : parts_ends[x_pos],
                parts_starts[y_pos] : parts_ends[y_pos],
            ] = selector
        x_shift, y_shift = np.random.choice(possible_shifts, 2)
        image[
            shape_starts[x_pos] + x_shift : shape_ends[x_pos] + x_shift,
            shape_starts[y_pos] + y_shift : shape_ends[y_pos] + y_shift,
        ] = shape
    return image
