import numpy as np
from keras.utils.vis_utils import plot_model

from scripts.dataset import create_dataset, def_selector
from scripts.explainability import apply_explainability
from scripts.model import build_model
from scripts.visualization import pplot


def main():
    # _________________________________________________________________________
    # Set dataset parameters
    nb_parts = 4  # (cannot be changed)
    shape_length = 12  # (should be pair and at least 12)
    margin = 3  # (should be at least 2 and increased with max_shift)
    max_shift = 0  # (should be 0, 1 or 2)
    subset_length = 6  # (number of samples for explainability and plots)

    # _________________________________________________________________________
    # Generate dataset
    image_data, raw_data, selected_positions, target, \
    target_onehot, shapes_names, shapes_dictionary = \
        create_dataset(
            nb_samples=1000,
            shape_length=shape_length,
            margin=margin,
            max_shift=max_shift,
        )
    subset = image_data[:subset_length]

    # _________________________________________________________________________
    # modify the selector filter so that it can be used as weights
    input_shape = image_data.shape[1:]
    selector = def_selector(shape_length, margin)
    selector_filter = (selector + 1) // 2

    # _________________________________________________________________________
    # Create the model with fixed and known weights
    model = build_model(
        input_shape,
        selector_filter,
        shapes_dictionary,
        nb_parts=nb_parts,
        bias=0,
        max_shift=max_shift,
    )
    plot_model(model, to_file="visualization/model_graph.png",
               show_shapes=True, show_layer_names=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.evaluate(image_data, target_onehot)

    # _________________________________________________________________________
    # Compute the explanations
    explanations = apply_explainability(model, subset, target_onehot[:subset_length])

    # _________________________________________________________________________
    # Visualize the explanations
    images = np.tile(subset, reps=(len(explanations) + 1, 1, 1, 1))
    explanations_list = [subset] + [np.reshape(expl, subset.shape)
                                    for expl in explanations.values()]
    images_explanations = np.concatenate(explanations_list, axis=0)

    methods_names = list(explanations.keys())
    subtitles = [f"original_{str(i)}" for i in range(subset_length)]
    for method in methods_names:
        subtitles += [*[f"{method}_{str(i)}" for i in range(subset_length)]]

    pplot(images, images_explanations, subtitles=subtitles,
          ncols=subset_length, filename="visualization/test_explanation",
          expl_cmap="coolwarm")

    pplot(images_explanations, subtitles=subtitles,
          ncols=subset_length, filename="visualization/only_explanation")


if __name__ == "__main__":
    """
    Main
    """
    main()
