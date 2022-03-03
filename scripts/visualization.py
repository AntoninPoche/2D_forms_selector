"""
Specific visualization functions
"""

import math
from matplotlib import pyplot as plt
import numpy as np


def pplot(
    data: np.array,
    explanation: np.array = None,
    title: str = "",
    subtitles: list = None,
    filename: str = None,
    show: bool = False,
    ncols: int = 6,
    img_cmap: str = "gray",
    expl_cmap: str = "jet",
):
    """
    Pretty plot function that shows one or several images.
    Explanations can be superposed to those images.

    Parameters
    ----------
    data
        Array representing an image or a list of images.
    explanation
        Array representing the explanation to superpose on data.
    title
        Title of the whole output.
    subtitles
        List of titles to give to each image when several images are shown.
    filename
        Path to the file where the output should be saved.
        If let to None, no file will be saved.
    show
        Boolean deciding if the output should be shown.
    ncols
        Number of columns if several images are ploted.
    img_cmap
        Colormap for the image.
    expl_cmap
        Colormap for the explanations.
    """
    subtitle = None
    # remove canal, otherwise plt.imshow() do not work
    if data.shape[-1] == 1:
        data = np.reshape(data, data.shape[:-1])
    if explanation is not None and explanation.shape[-1] == 1:
        explanation = np.reshape(explanation, explanation.shape[:-1])

    # verify that explanations can be superposed to data
    if explanation is not None:
        assert (
            data.shape == explanation.shape
        ), "Data and explanation shapes should match."

    # verify that length of subtitles correspond to the number of images
    if subtitles is not None:
        assert len(subtitles) == data.shape[0], "There should be one subtitle by image."

    # first case, we have several images to plot, (3 dimensions)
    if len(data.shape) == 3:
        # compute the number of rows and lines depending on the number of images
        subplot_kwargs = _arrange_subplots(data, ncols)
        plt.rcParams["figure.figsize"] = subplot_kwargs["figsize"]

        # iterate on all image and make a recursive call of the function
        for index, image in enumerate(data):
            plt.subplot(subplot_kwargs["nrows"], subplot_kwargs["ncols"], index + 1)

            # set subtitle
            if subtitles is not None:
                subtitle = subtitles[index]

            if explanation is not None:
                pplot(image, explanation[index],
                      title=subtitle, show=False,
                      img_cmap=img_cmap, expl_cmap=expl_cmap)
            else:
                pplot(image,
                      title=subtitle, show=False,
                      img_cmap=img_cmap, expl_cmap=expl_cmap)
        plt.title(title)
        plt.tight_layout()
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename)
    # second case, we only have one image
    else:
        # show the image
        plt.imshow(data, cmap=img_cmap)

        # superpose the explanation if there is one
        if explanation is not None:
            # may need fixed min and max for colors
            plt.imshow(explanation, cmap=expl_cmap, alpha=0.5)  # , vmin=-1, vmax=1

        plt.title(title, fontsize=16)
        plt.axis("off")
        if show:
            plt.show()
        if filename is not None:
            plt.savefig(filename)


def _arrange_subplots(image: np.array, ncols: int = 6) -> dict:
    """
    Determine number of rows and columns based on the number of images.
    Used when several images are ploted using pplot().

    Parameters
    ----------
    image
        Example of the image that will be shown.
    ncols
        Number of columns of images for pplot() output

    Returns
    -------
    subplot_kwargs
        Dictionary of arguments for matplotlib subplots
    """
    nb_plots, shape_length = image.shape[:2]
    nrows = math.ceil(nb_plots / ncols)
    if nrows == 1:
        ncols = nb_plots
    figsize = (shape_length * ncols // 8, shape_length * nrows // 8)
    subplot_kwargs = {"nrows": nrows, "ncols": ncols, "figsize": figsize}
    return subplot_kwargs
