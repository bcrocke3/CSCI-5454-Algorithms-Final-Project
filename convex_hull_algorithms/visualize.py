# python imports
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# types for n-dimensional arrays
NDFloatArray = npt.NDArray[np.float64]
NDIntArray = npt.NDArray[np.int64]


def draw2D(points: NDFloatArray, colors: NDIntArray, title: str = "Convex Hull Results") -> None:
    """ Plots 2D points in multiple colors

    :param points: 2D array of points to plot. Each row is a point. Must have 2 columns for x, y dimensions
    :param colors: Column vector of category/color that corresponds to each point
    :param title: title of output graph. Should describe how input data was computed
    :return: Nothing. Draws plots.
    """
    # check assumptions
    assert(len(points.shape) == 2)  # 2D points matrix
    assert(points.shape[1] == 2)    # all the points are 2D (2, y)
    assert(len(colors.shape) == 2)  # colors is column vector
    assert(colors.shape[1] == 1)
    assert(points.shape[0] == colors.shape[0])  # points and colors have same # of rows

    # define colors
    colormap = np.array(['black', 'red'])

    # graph
    plt.scatter(points[:, 0], points[:, 1], c=colormap[colors[:, 0]])
    plt.title(title)
    plt.show()
