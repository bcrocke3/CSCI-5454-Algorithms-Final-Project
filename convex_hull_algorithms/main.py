# python imports
import numpy as np
import numpy.typing as npt

# local imports
import divide_and_conquer
import quickhull
import visualize

# types for n-dimensional arrays
NDFloatArray = npt.NDArray[np.float64]
NDIntArray = npt.NDArray[np.int64]

if __name__ == '__main__':
    # make up some input
    test_input: NDFloatArray = np.array([[0.0, 1.0],
                                         [1.0, 1.0],
                                         [1.0, 0.0],
                                         [0.0, 0.0],
                                         [0.5, 0.5]],
                                        dtype=np.float64)

    test_answer: NDIntArray = np.array([[1], [1], [1], [1], [0]], dtype=np.int64)

    # call the different algorithms
    dc_result: NDIntArray = divide_and_conquer.convexhull(test_input)
    quickhull_result: NDIntArray = quickhull.convexhull(test_input)

    # draw results
    visualize.draw2D(test_input, dc_result, "Divide and Conquer Result")
    visualize.draw2D(test_input, quickhull_result, "Quickhull Result")
    visualize.draw2D(test_input, test_answer, "Hard-Coded Answer")
