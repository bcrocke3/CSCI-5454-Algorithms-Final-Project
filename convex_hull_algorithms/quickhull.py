# python imports
import numpy as np
import numpy.typing as npt

# types for n-dimensional arrays
NDFloatArray = npt.NDArray[np.float64]
NDIntArray = npt.NDArray[np.int64]


def convexhull(input_points: NDFloatArray) -> NDIntArray:
    """ Uses the Quickhull approach to compute the convex hull of the set of input_points

    :param input_points: a 2D array of input points. Each row is a point. The number of columns depends
                        on the dimension of the points
    :return: a column vector with as many rows as input_points. Values are binary = 1 for a point that
            is on the convex hull, 0 otherwise
    """
    assert(len(input_points.shape) == 2)  # check that input points is 2D array

    # some useful attributes of input points
    num_points: int = input_points.shape[0]
    dim: int = input_points.shape[1]

    print(f"Num Points: {num_points}")
    print(f"Dimension: {dim}")

    # fake result - every point is on the convex hull
    result: NDIntArray = np.zeros((num_points, 1), dtype=np.int64)

    # check that result is correct shape
    assert(result.shape[0] == input_points.shape[0])
    assert(result.shape[1] == 1)

    return result
