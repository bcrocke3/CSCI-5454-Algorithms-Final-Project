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


def gen_n_random_points_2d(n: int, min: float, max: float) -> NDFloatArray:
    return np.random.uniform(min, max, (n, 2))


def gen_n_random_points_3d(n: int, min: float, max: float) -> NDFloatArray:
    return np.random.uniform(min, max, (n, 3))


if __name__ == '__main__':
    # make up some input
    test_input: NDFloatArray = np.array([[0.0, 1.0],
                                         [1.0, 1.0],
                                         [1.0, 0.0],
                                         [0.0, 0.0],
                                         [0.5, 0.5]],
                                        dtype=np.float64)
    test_input_2: NDFloatArray = np.array([[0.5, 1.0],
                                           [1.5, 1.0],
                                           [1.0, 0.0],
                                           [0.0, 0.0],
                                           [0.5, 0.5]],
                                          dtype=np.float64)

    test_input_3: NDFloatArray = gen_n_random_points_2d(10, 0, 10)

    test_answer: NDIntArray = np.array([[1], [1], [1], [1], [0]], dtype=np.int64)

    # call the different algorithms
    dc_result: NDIntArray = divide_and_conquer.convexhull_2d(test_input)
    dc_result_2: NDIntArray = divide_and_conquer.convexhull_2d(test_input_2)
    dc_result_3: NDIntArray = divide_and_conquer.convexhull_2d(test_input_3)
    # quickhull_result: NDIntArray = quickhull.convexhull(test_input)

    # draw results
    visualize.draw2D(test_input, dc_result, "Divide and Conquer Result 1")
    visualize.draw2D(test_input_2, dc_result_2, "Divide and Conquer Result 2")
    visualize.draw2D(test_input_3, dc_result_3, "Divide and Conquer Result - Random Input")

    # visualize.draw2D(test_input, quickhull_result, "Quickhull Result")
    # visualize.draw2D(test_input, test_answer, "Hard-Coded Answer")

    test_input_3d: NDFloatArray = np.array([[0.0, 0.0, 0.0],
                                            [0.0, 0.0, 2.0],
                                            [2.0, 0.0, 0.0],
                                            [2.0, 0.0, 2.0],
                                            [2.0, 2.0, 0.0],
                                            [2.0, 2.0, 2.0],
                                            [0.0, 2.0, 0.0],
                                            [0.0, 2.0, 2.0],
                                            [1.0, 1.0, 1.0]],
                                           dtype=np.float64)

    # test_input_3d: NDFloatArray = gen_n_random_points_3d(10, 0, 100)

    test_answer_3d: NDIntArray = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [0]], dtype=np.int64)

    # quickhull_result_3d: NDIntArray = quickhull.convexhull3d(test_input_3d)

    # visualize.draw3D(test_input_3d, quickhull_result_3d, "3D Result")
