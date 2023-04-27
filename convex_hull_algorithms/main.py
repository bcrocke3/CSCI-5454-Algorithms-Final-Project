# python imports
import numpy as np
import numpy.typing as npt
import trimesh


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


def load_slt_to_point_cloud(file_name):
    mesh = trimesh.load(file_name)
    points = mesh.vertices
    faces = mesh.faces

    return points, faces


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
    quickhull_result: NDIntArray = quickhull.convexhull(test_input)

    # draw results
    visualize.draw2D(test_input, dc_result, "Divide and Conquer Result 1")
    visualize.draw2D(test_input_2, dc_result_2, "Divide and Conquer Result 2")
    visualize.draw2D(test_input_3, dc_result_3, "Divide and Conquer Result - Random Input 1")


    # Cube points
    # test_input_3d: NDFloatArray = np.array([[0.0, 0.0, 0.0],
    #                                         [0.0, 0.0, 2.0],
    #                                         [2.0, 0.0, 0.0],
    #                                         [2.0, 0.0, 2.0],
    #                                         [2.0, 2.0, 0.0],
    #                                         [2.0, 2.0, 2.0],
    #                                         [0.0, 2.0, 0.0],
    #                                         [0.0, 2.0, 2.0],
    #                                         [1.0, 1.0, 1.0]],
    #                                          dtype=np.float64)

    # Random points
    # test_input_3d: NDFloatArray = gen_n_random_points_3d(1000, 0, 100)

    # Add a bounding box around the random points to test hull
    # test_input_3d_box: NDFloatArray = np.array([[0.0, 0.0, 0.0],
    #                                         [0.0, 0.0, 102],
    #                                         [102, 0.0, 0.0],
    #                                         [102, 0.0, 102],
    #                                         [102, 102, 0.0],
    #                                         [102, 102, 102],
    #                                         [0.0, 102, 0.0],
    #                                         [0.0, 102, 102]],
    #                                          dtype=np.float64)
    # test_input_3d = np.concatenate((test_input_3d, test_input_3d_box), axis=0)

    # Load points STL file
    # stl_points, stl_faces = load_slt_to_point_cloud("chair.stl")
    # test_input_3d = stl_points

    # quickhull_result_3d, face_indices, face_points = quickhull.convexhull3d(test_input_3d)

    # visualize.draw3D(test_input_3d, quickhull_result_3d, "3D Result")

    # Use with non-stl data
    # visualize.interact3d(test_input_3d, quickhull_result_3d, face_indices, face_points)

    # Use with stl data to get original mesh too
    # visualize.interact3d(test_input_3d, quickhull_result_3d, face_indices, face_points, stl_faces, stl_points)
