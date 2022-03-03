from keras.models import Model
import numpy as np
from time import time
import tensorflow as tf
from typing import Dict
from xplique.attributions import (Saliency, GradientInput, IntegratedGradients, SmoothGrad, VarGrad,
                                  SquareGrad, GradCAM, Occlusion, Rise, GuidedBackprop,
                                  GradCAMPP, Lime, KernelShap)


def apply_explainability(
        model: Model,
        inputs: np.ndarray,
        target_onehot: np.array
) -> Dict[str, np.ndarray]:
    """
    Apply all the attributions method from Xplique to the 2D forms.

    Parameters
    ----------
    model
        The model to study.
    inputs
        The inputs of the model.
    target_onehot
        One hot encoding of the target.

    Returns
    -------
    explanations
        Dictionary of explanations,
        the keys are the explainability methods names,
        the values are arrays of attributions,
        they have the same shape as the inputs.
    """
    # to explain the logits is to explain the class,
    # to explain the softmax is to explain why this class rather than another
    # it is therefore recommended to explain the logit
    model.layers[-1].activation = tf.keras.activations.linear
    batch_size = 64

    # Define expalainers
    explainers = {
        "Saliency": Saliency(model),
        "GradientInput": GradientInput(model),
        "GuidedBackprop": GuidedBackprop(model),
        "IntegratedGradients": IntegratedGradients(model, steps=80),
        "SmoothGrad": SmoothGrad(model, nb_samples=80),
        "SquareGrad": SquareGrad(model, nb_samples=80),
        "VarGrad": VarGrad(model, nb_samples=80),
        "GradCAM": GradCAM(model),
        "Rise": Rise(model, nb_samples=4000),
        "KernelShap": KernelShap(model, nb_samples=200, ref_value=0.0, ),
        "Lime": Lime(model, nb_samples=200, ref_value=0.0, distance_mode="euclidean", ),
        "Occlusion": Occlusion(model, patch_size=2, patch_stride=1, occlusion_value=0.0),
    }

    explanations = {}
    for method, explainer in explainers.items():
        t = time()
        if method in ["KernelShap", "Lime"]:
            explanation = explainer(np.reshape(inputs, inputs.shape[:-1]), target_onehot)
        else:
            explanation = explainer(inputs, target_onehot)
        # print(np.min(explanation), np.max(explanation))
        # print(method, round(time() - t, 2))
        explanations[method] = explanation

    return explanations
