# python imports
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

import visualize

# types for n-dimensional arrays
NDFloatArray = npt.NDArray[np.float64]
NDIntArray = npt.NDArray[np.int64]


def evaluate_point_line(point, line_start, line_end):
    """ returns positive number if point is above the line, negative number if point below line, zero if point is
        on the line; "above" and "below" determined by ordering of line_start and line_end

    :param point: a point to compare to the line
    :param line_start: the starting point defining the line
    :param line_end:  the ending point defining the line
    :return: float, how the point evaluates in the line equation
    """

    if line_end[0] - line_start[0] != 0:
        slope: float = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])

        return (slope * (point[0] - line_start[0])) - (point[1] - line_start[1])

    else:
        # line is vertical, so only x-coordinate determines point's side of line
        return point[0] - line_start[0]


def convexhull_2d(input_points: NDFloatArray) -> NDIntArray:
    """ Uses the divide and conquer approach to compute the convex hull of the set of input_points

    :param input_points: a 2D array of input points. Each row is a point. The number of columns depends
                        on the dimension of the points
    :return: a column vector with as many rows as input_points. Values are binary = 1 for a point that
            is on the convex hull, 0 otherwise
    """
    assert(len(input_points.shape) == 2)  # check that input points is 2D array

    # some useful attributes of input points
    num_points: int = input_points.shape[0]
    dim: int = input_points.shape[1]

    print("Computing 2D Hull with Divide and Conquer...")
    print(f"Num Points: {num_points}")
    print(f"Dimension: {dim}")

    # compute the Convex Hull
    result_points = convexhull_2d_return_points(input_points)

    # convert the list of hull points to a binary vector
    result: NDIntArray = np.zeros((num_points, 1), dtype=np.int64)  # assume nothing is in the hull

    for input_index, input_pt in enumerate(input_points):  # mark all the points that are in the hull
        for result_pt in result_points:

            if np.array_equal(input_pt, result_pt):
                result[input_index, 0] = 1

    # check that result is correct shape
    assert(result.shape[0] == input_points.shape[0])
    assert(result.shape[1] == 1)

    return result


def brute_hull(input_points: NDFloatArray) -> NDFloatArray:
    """ Brute force computation for finding the convex hull

    :param input_points: small set of input points to compute the convex hull of
    :return: 2 array representing the points that are on the hull, in counter-clockwise ordering
    """
    num_points = input_points.shape[0]

    hull_edges = []

    for i in range(num_points):
        for j in range(i+1, num_points):

            point_i = input_points[i]
            point_j = input_points[j]

            def evaluate_point(test_point):
                if (point_j[0] - point_i[0]) != 0:
                    m = (point_j[1] - point_i[1]) / (point_j[0] - point_i[0])

                    return (m * (test_point[0] - point_i[0])) - (test_point[1] - point_i[1])
                else:
                    return test_point[0] - point_i[0]

            above_line, below_line = 0, 0
            for p in input_points:
                eval_pt = evaluate_point(p)
                if eval_pt >= 0.0:
                    above_line += 1
                if eval_pt <= 0.0:
                    below_line += 1

            if above_line == num_points or below_line == num_points:
                # orient point
                any_pt = None
                if j + 1 < num_points:
                    any_pt = input_points[j + 1]
                else:
                    any_pt = input_points[i - 1]  # any pt not i or j

                vec1 = point_j - point_i  # vec from i to j
                vec2 = any_pt - point_j  # vec from any to j

                first, second = i, j
                if np.cross(vec1, vec2) < 0:
                    # winding is wrong, flip it
                    first, second = j, i

                # add the edge to the list
                hull_edges.append((first, second))

    sort_ordering(hull_edges)

    result_points = []
    for edge in hull_edges:
        result_points.append(input_points[edge[0]])

    result_array = np.array(result_points)
    print("HULL POINTS - IN CCW ORDER")
    print(result_array)

    return result_array


def convexhull_2d_return_points(input_points: NDFloatArray) -> NDFloatArray:
    num_points: int = input_points.shape[0]
    result_points = None

    if num_points <= 5:
        result_points = brute_hull(input_points)

    else:
        # divide
        print("DIVIDE")
        print("Input Points")
        print(input_points)
        sorted_points = input_points[np.argsort(input_points[:, 0])]
        print("Sorted Points")
        print(sorted_points)

        middle_index = num_points // 2
        left_half = sorted_points[0:middle_index, :]
        right_half = sorted_points[middle_index:, :]

        # conquer
        left_hull_points = convexhull_2d_return_points(left_half)
        right_hull_points = convexhull_2d_return_points(right_half)

        draw_partial(left_half, left_hull_points)
        draw_partial(right_half, right_hull_points)

        # merge
        left_max = left_hull_points[np.argmax(left_hull_points[:, 1])]  # point with max y coord, from left hull
        left_min = left_hull_points[np.argmin(left_hull_points[:, 1])]  # point with min y coord, from left hull
        right_max = right_hull_points[np.argmax(right_hull_points[:, 1])]  # point with max y coord, from right hull
        right_min = right_hull_points[np.argmin(right_hull_points[:, 1])]  # point with min y coord, from right hull

        combined_hull = []
        left_slope = (left_max[1] - left_min[1]) / (left_max[0] - left_min[0])
        for left_hull_pt in left_hull_points:
            pt_eval = evaluate_point_line(left_hull_pt, left_min, left_max)

            if left_slope > 0 and pt_eval > 0 or left_slope < 0 and pt_eval < 0:
                # point is no longer on the hull bc it's to the right of the min/max line
                pass
            else:
                combined_hull.append(left_hull_pt)

        right_slope = (right_max[1] - right_min[1]) / (right_max[0] - right_min[0])
        for right_hull_pt in right_hull_points:
            pt_eval = evaluate_point_line(right_hull_pt, right_min, right_max)

            if right_slope > 0 and pt_eval < 0 or right_slope < 0 and pt_eval > 0:
                # point is no longer on the hull bc it's to the left of the min/max line
                pass
            else:
                combined_hull.append(right_hull_pt)

        # return
        result_points = np.array(combined_hull)

    draw_partial(input_points, result_points)
    return result_points


def draw_partial(input_points, point_on_hull):
    num_points: int = input_points.shape[0]

    result: NDIntArray = np.zeros((num_points, 1), dtype=np.int64)  # assume nothing is in the hull

    for input_index, input_pt in enumerate(input_points):  # mark all the points that are in the hull
        for hull_pt in point_on_hull:

            if np.array_equal(input_pt, hull_pt):
                result[input_index, 0] = 1

    visualize.draw2D(input_points, result, "Divide and Conquer Intermediate Result")


def sort_ordering(unsorted_ordering: List[Tuple[int, int]]):
    """ inplace sort of tuples for edge ordering

    :param unsorted_ordering:
    :return:
    """
    num_pairs = len(unsorted_ordering)

    start = unsorted_ordering[0]

    for i in range(1, num_pairs):
        i_pair = unsorted_ordering[i]
        if start[1] == i_pair[0]:
            pass
        else:
            for j in range(i+1, num_pairs):
                j_pair = unsorted_ordering[j]
                # if they match, swap, and break loop
                if start[1] == j_pair[0]:
                    unsorted_ordering[i] = j_pair
                    unsorted_ordering[j] = i_pair
                    break

        start = unsorted_ordering[i]

    # there is wrap around
    assert(unsorted_ordering[0][0] == unsorted_ordering[-1][-1])


if __name__ == '__main__':
    test = [(0, 4), (6, 1), (5, 0), (2, 5), (4, 6), (3, 2), (1, 3)]

    print(test)
    sort_ordering(test)
    print(test)
