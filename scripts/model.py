from itertools import product

from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import DepthwiseConv2D
from keras.layers import Flatten
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.engine.keras_tensor import KerasTensor

import numpy as np
from typing import Dict, Tuple, List


def build_model(
        input_shape: Tuple,
        selector_filter: np.array,
        shapes: Dict[str, np.array],
        nb_parts: int = 4,
        bias: float = 0,
        max_shift: int = 0,
        name: str = "transparent_model",
) -> Model:
    """
    Creates the model adapted to the 2D forms selector dataset,
    it also fill the weights.
    The model have four parts:
        the first one that separates the different part of the image,
        the second that find which is the part to look at,
        the third in parallel of the second that find shapes scores for each part,
        the fourth one using both previous part to selected the shapes scores.


    Parameters
    ----------
    input_shape
        Shape of and image.
    selector_filter
        Weights used to see which part is selected. (Depend on the dataset).
    shapes
        Dictionary of all possible shapes in the dataset.
    nb_parts
        Number of parts in the dataset.
    bias
        Bias to apply after shapes filter are applied. (Should be between -1 and 1).
        The activation of the corresponding layer is Relu.
    max_shift
        Variable of the dataset setting inducing
        the number of possible positions of a shape.
        It influences the number of channel output by _possible_positions_separator().
    name
        Name of the model.

    Returns
    -------
    model
        The model with filled weights.
    """
    # initialize input
    inputs = Input(shape=input_shape)

    # separate parts
    parts = _parts_separator(inputs, nb_parts)

    # select parts (branch 1)
    select_parts = _part_selector(parts, selector_filter, nb_parts)
    # select_parts = tf.keras.backend.print_tensor(select_parts, "select_parts", summarize=-1)

    # compute shapes scores (branch 2)
    parts_shapes_scores = _shape_detector(
        parts, shapes, nb_parts, bias=bias, max_shift=max_shift
    )
    # parts_shapes_scores = tf.keras.backend.print_tensor(parts_shapes_scores, "parts_shapes_scores " + str(parts_shapes_scores.shape), summarize=-1)

    # select corresponding score
    output = _score_selector(select_parts, parts_shapes_scores, shapes, nb_parts)
    # output = tf.keras.backend.print_tensor(output, "output", summarize=-1)

    # build model
    model = Model(inputs, output, name=name)

    return model


def _parts_separator(
        inputs: KerasTensor,
        nb_parts: int = 4
) -> KerasTensor:
    """
    Separate the initial image between the four parts.
    """
    kernel_size = inputs.shape[1] // 2 + 1

    # set filters weights,
    # (19*19) filters with a 1 at one  of the angle
    # one filter for each part (one angle by filter)
    weights = np.zeros((kernel_size, kernel_size, 1, nb_parts))
    for i, (posx, posy) in enumerate(list(product([0, -1], repeat=2))):
        weights[posx, posy, :, i] = 1

    # set biases to 0
    bias = np.zeros((nb_parts,))

    # create Conv2D layer with those filters as weights
    parts = Conv2D(
        filters=nb_parts,
        kernel_size=kernel_size,
        strides=(1, 1),
        activation=None,
        name="separate_parts",
        weights=[weights, bias],
    )(inputs)

    return parts


def _part_selector(
        parts: KerasTensor,
        selector: np.array,
        nb_parts: int = 4
) -> KerasTensor:
    """
    Take the different parts filter and use the selector filter
    to see which part is selected.
    It outputs one value by part.
    """
    # verify that shapes are compatible
    parts_shape = parts.shape[1:-1]
    assert parts_shape == selector.shape

    # set weights with the selector (extended for each filter)
    extended_selector = np.reshape(selector, selector.shape + (1, 1))
    weights = np.tile(extended_selector, (1, 1, nb_parts, 1))

    # set biases to 0
    bias = np.zeros((parts.shape[-1]))

    # create layer with selector weights
    select_part = DepthwiseConv2D(
        depth_multiplier=1,
        kernel_size=parts_shape[0],
        strides=(1, 1),
        activation="sigmoid",
        name="select_part",
        weights=[weights, bias],
    )(parts)

    select_part = Flatten(name="flatten_selected_part")(select_part)

    return select_part


def _shape_detector(
        parts: KerasTensor,
        shapes: Dict,
        nb_parts: int = 4,
        bias: float = 0.0,
        max_shift: int = 0,
) -> KerasTensor:
    """
    Evaluate the scores of each shape in each part.
    If max_shift is not zero, it extracts the different possible positions,
    and also computes the different possible positions scores.
    """
    shape_length = list(shapes.values())[0].shape[0]

    possible_shifts = list(range(-max_shift, max_shift + 1))
    shifts_combinasons = list(product(possible_shifts, repeat=2))

    # separate each possible position
    shape_windows = _possible_positions_separator(
        parts, shape_length, shifts_combinasons
    )

    # evaluate position shape matching
    position_shape_scores = _shape_comparator(shape_windows, shapes, nb_parts)

    # fuse scores by positions
    parts_shapes_scores = _position_score_mergor(
        position_shape_scores, len(shapes), shape_length, nb_parts, bias
    )

    return parts_shapes_scores


def _score_selector(
        select_part: KerasTensor,
        parts_shapes_scores: KerasTensor,
        shapes: Dict,
        nb_parts: int = 4,
) -> KerasTensor:
    """
    Use the parts selection to select the corresponding shapes scores.
    """
    nb_shapes = len(shapes)

    # join both model branch
    join_layer = Concatenate(name="concatenate_part_and_scores")(
        [select_part, parts_shapes_scores]
    )
    # join_layer = tf.keras.backend.print_tensor(join_layer, "join_layer " + str(join_layer.shape), summarize=-1)

    # select part corresponding score
    selected_scores = _part_score_selector(join_layer, nb_shapes, nb_parts)
    # selected_scores = tf.keras.backend.print_tensor(selected_scores, "selected_scores " + str(selected_scores.shape), summarize=-1)

    # reduced scores
    output = _score_reductor(selected_scores, nb_shapes, nb_parts)
    # output = tf.keras.backend.print_tensor(output, "output " + str(output.shape), summarize=-1)

    return output


def _possible_positions_separator(
        parts: KerasTensor,
        shape_length: int = 12,
        shifts_combinaisons: list = [[0, 0]],
) -> KerasTensor:
    """
    There may be several possible positions for the shapes.
    (It depends on the max_shift variable).
    For each part all possible positions are separated.
    The output shapes is thus (shape_length * shape_length)
    with nb_part * nb_possible_positions filters.
    """
    part_length = parts.shape[1]
    nb_parts = parts.shape[-1]

    kernel_size = part_length - shape_length + 1

    # set one filter by possible positions (19*19) filters with a 1 at the shifted center and bias to 0
    weights = np.zeros((kernel_size, kernel_size, nb_parts, len(shifts_combinaisons)))
    bias = np.zeros((len(shifts_combinaisons) * nb_parts,))

    kernel_center = kernel_size // 2
    for i, (posx, posy) in enumerate(shifts_combinaisons):
        weights[kernel_center + posx, kernel_center + posy, :, i] = 1

    # create Conv2D layer with those filters as weights
    shape_windows = DepthwiseConv2D(
        depth_multiplier=len(shifts_combinaisons),
        kernel_size=kernel_size,
        strides=(1, 1),
        activation=None,
        name="extract_possible_shape_windows",
        weights=[weights, bias],
    )(parts)

    return shape_windows


def _shape_comparator(
        shape_windows: KerasTensor,
        shapes: Dict,
        nb_parts: int = 4
) -> KerasTensor:
    """
    Compare shape with known shape filters.
    Extract a score for each part, position and known shape, thus 4*9*nb_shapes.
    Then flatten this output.
    """
    nb_positions = shape_windows.shape[3] // nb_parts
    nb_shapes = len(shapes)
    shape_length = shape_windows.shape[1]

    # create weights array
    weights = np.zeros(shape_windows.shape[1:] + (nb_shapes,))
    bias = np.zeros(shape_windows.shape[3] * nb_shapes)

    for i, shape in enumerate(shapes.values()):
        assert shape_length == shape.shape[0]
        shape_weights = np.expand_dims(shape, axis=-1)
        weights[:, :, :, i] = np.tile(shape_weights, (1, 1, nb_positions * nb_parts))

    # create layer
    position_shape_scores = DepthwiseConv2D(
        depth_multiplier=nb_shapes,
        kernel_size=shape_length,
        strides=(1, 1),
        activation=None,
        name="apply_shapes_filters",
        weights=[weights, bias],
    )(shape_windows)

    position_shape_scores = Flatten(name="flatten_position_shape_scores")(
        position_shape_scores
    )

    return position_shape_scores


def _position_score_mergor(
        scores: KerasTensor,
        nb_shapes: int,
        shape_length: int = 12,
        nb_parts: int = 4,
        bias: float = 0.0,
) -> KerasTensor:
    """
    Take the mean score over the possible positions, score is normalized between -1 and 1.
    It outputs one score for each pair: part-shape.
    """
    nb_inputs = scores.shape[1]
    nb_outputs = nb_shapes * nb_parts

    assert 0 <= bias <= 1

    # set weights to create mean between positions (scores are normalized between -1 and 1)
    # The precedent DepthwiseConv2D group output by input channels
    # Thus here we will have the following order:
    # (part1, pos1, shape1), (part1, pos1, shape2), (part1, pos2, shape1)
    weights = np.zeros((nb_inputs, nb_outputs))
    biases = np.full((nb_outputs,), bias)

    nb_positions = nb_inputs // nb_outputs
    ratio = 1 / (nb_positions * shape_length**2)

    for part_id in range(nb_parts):
        part_index = part_id * nb_positions * nb_shapes
        part_outdex = part_id * nb_shapes
        for pos_id in range(nb_positions):
            pos_index = pos_id * nb_shapes
            for shape_id in range(nb_shapes):
                input_id = part_index + pos_index + shape_id
                output_id = part_outdex + shape_id
                weights[input_id, output_id] = ratio

    # create layer
    parts_shapes_scores = Dense(
        nb_outputs,
        activation="relu",
        name="merge_positions_scores",
        weights=[weights, biases],
    )(scores)

    return parts_shapes_scores


def _part_score_selector(
        join_layer: KerasTensor,
        nb_shapes: int,
        nb_parts: int = 4
) -> KerasTensor:
    """
    Use the selected part to filter the corresponding shapes scores.
    Only the selected part should have non-zero scores.
    """

    nb_inputs = join_layer.shape[1]
    nb_outputs = nb_shapes * nb_parts

    # set weights so that the good score is kept and others set to 0
    # both score and part selector are summed, the max is 2, thus we set a bias to -1
    # finally, with relu, scores are between 0 and 1
    weights = np.zeros((nb_inputs, nb_outputs))
    bias = np.full((nb_outputs,), -1)

    for part in range(nb_parts):
        weights[part, part * nb_shapes : (part + 1) * nb_shapes] = 1
        for shape in range(nb_shapes):
            id = shape + part * nb_shapes
            weights[nb_parts + id, id] = 1

    # create layer
    output = Dense(
        nb_outputs,
        activation="relu",
        name="select_score",
        weights=[weights, bias],
    )(join_layer)

    return output


def _score_reductor(
        selected_scores: KerasTensor,
        nb_shapes: int,
        nb_parts: int = 4
) -> KerasTensor:
    """
    Sum scores of each part to get only one score for each shape.
    """

    # set weights to reduce to one score for each shape
    # parts scores are fused (but only one should not be zero)
    weights = np.zeros((nb_parts * nb_shapes, nb_shapes))
    bias = np.zeros((nb_shapes,))

    for part in range(nb_parts):
        for shape in range(nb_shapes):
            weights[part * nb_shapes + shape, shape] = 1

    # create layer
    output = Dense(
        nb_shapes,
        activation="softmax",
        name="reduce_score",
        weights=[weights, bias],
    )(selected_scores)

    return output
