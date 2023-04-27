# python imports
import numpy as np
import numpy.typing as npt
from typing import List, Tuple

# types for n-dimensional arrays
NDFloatArray = npt.NDArray[np.float64]
NDIntArray = npt.NDArray[np.int64]


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

    assert(num_points >= 3)
    assert(dim == 2)

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


def convexhull_2d_return_points(input_points: NDFloatArray) -> NDFloatArray:
    """ Convex Hull function that does the dividing and conquering. Returns list of points, not binary array.

    :param input_points: 2D array representing input points
    :return: 2D array, a subset of input_points that make up the hull
    """
    num_points: int = input_points.shape[0]
    result_points = None

    # base case
    if num_points <= 5:
        result_points = brute_hull(input_points)

    # recursive case
    else:
        # divide
        sorted_points = input_points[np.argsort(input_points[:, 0])]  # sort by x-coord to split
        middle_index = num_points // 2
        left_half = sorted_points[0:middle_index, :]
        right_half = sorted_points[middle_index:, :]

        # conquer
        left_hull_points = convexhull_2d_return_points(left_half)
        right_hull_points = convexhull_2d_return_points(right_half)

        # merge
        # first find the upper and lower tangents to the two hull
        # use tangents to decide which points to keep from left and right hulls
        up_tan_left, up_tan_right, lo_tan_left, lo_tan_right = find_tangent_points(left_hull_points, right_hull_points)

        combined_hull = []
        if lo_tan_left < up_tan_left:
            lo_tan_left += left_hull_points.shape[0]
        for i in range(up_tan_left, lo_tan_left + 1):
            combined_hull.append(left_hull_points[np.mod(i, left_hull_points.shape[0])])

        if up_tan_right < lo_tan_right:
            up_tan_right += right_hull_points.shape[0]
        for i in range(lo_tan_right, up_tan_right + 1):
            combined_hull.append(right_hull_points[np.mod(i, right_hull_points.shape[0])])

        result_points = np.array(combined_hull)  # maintains CCW winding of points

    return result_points


def brute_hull(input_points: NDFloatArray) -> NDFloatArray:
    """ Brute force computation for finding the convex hull

    :param input_points: small set of input points to compute the convex hull of
    :return: 2D array representing the points that are on the hull, in counter-clockwise ordering
    """
    num_points = input_points.shape[0]
    assert(3 <= num_points <= 10)

    # draw all the points with labels of input order
    # visualize.draw2D(input_points, np.ones(shape=(len(input_points), 1), dtype=np.int64), label_points=True)

    hull_edges = []  # to fill with pairs of indices of points

    # Check every pair of points to see if it is a hull edge
    for i in range(num_points):
        for j in range(i+1, num_points):

            # draw a line between two points (point_i and point_j)
            # check if all the other points are on one side of the line
            # if yes, this is an edge on the hull

            point_i = input_points[i]
            point_j = input_points[j]

            above_line, below_line = 0, 0
            for p in input_points:
                eval_pt = evaluate_point_line(p, point_i, point_j)
                if eval_pt <= 0.0:
                    above_line += 1
                if eval_pt >= 0.0:
                    below_line += 1

            assert(above_line + below_line == num_points + 2) # the two points that make line should be double counted

            if above_line == num_points or below_line == num_points:
                # if all the points were on one side of the line, find which order the points should go in (CCW)

                # pick any point, not i or j to orient hull points
                rand_index = np.random.randint(0, num_points)
                while rand_index == i or rand_index == j:
                    rand_index = np.random.randint(0, num_points)
                any_pt = input_points[rand_index]

                vec1 = point_j - point_i  # vec from i to j
                vec2 = any_pt - point_j  # vec from any to j

                first, second = i, j
                if np.cross(vec1, vec2) < 0:
                    # winding is wrong, flip it
                    first, second = j, i

                # add the edge to the list
                hull_edges.append((first, second))

    assert(len(hull_edges) >= 3)  # a hull must be at least a triangle

    # hull_edges contains list of edges in no particular order
    # sort them so that each edge shares a point with the next
    sort_ordering(hull_edges)

    # turn list of indices into list of points
    result_points = []
    for edge in hull_edges:
        result_points.append(input_points[edge[0]])

    result_array = np.array(result_points)

    # draw just the hull points; should be labelled in CCW order
    # visualize.draw2D(result_array, np.ones(shape=(len(result_array), 1), dtype=np.int64), label_points=True)

    return result_array


def find_tangent_points(left_polygon:NDFloatArray, right_polygon: NDFloatArray) -> Tuple[int, int, int, int]:
    """ Given two polygons (as two list of CCW wound points), returns the indices of the upper and lower tangent points

    :param left_polygon: 2D array of points, with counter-clockwise winding
    :param right_polygon: 2D array of points, with counter-clockwise winding
    :return: indices of left upper, right upper, left lower, and right lower tangent points
    """
    num_left_pts: int = left_polygon.shape[0]
    num_right_pts: int = right_polygon.shape[0]
    total_pts: int = num_left_pts + num_right_pts
    print("Left Points: ", num_left_pts)

    # find upper tangent points
    left_upper_index = np.argmax(left_polygon[:, 0])
    right_upper_index = np.argmin(right_polygon[:, 0])

    left_upper_pt = left_polygon[left_upper_index]  # the rightmost point of left polygon
    right_upper_pt = right_polygon[right_upper_index]  # leftmost point of right polygon
    found_upper_tangent = False
    while not found_upper_tangent:

        # do the check - evaluate every point against line made with left/right tangent points
        l_below_cnt = 0
        r_below_cnt = 0
        for lp in left_polygon:
            e = evaluate_point_line(lp, left_upper_pt, right_upper_pt)
            if e >= 0.0:
                l_below_cnt += 1

        for rp in right_polygon:
            e = evaluate_point_line(rp, left_upper_pt, right_upper_pt)
            if e >= 0.0:
                r_below_cnt += 1

        if l_below_cnt + r_below_cnt == total_pts:
            found_upper_tangent = True
        else:
            if l_below_cnt < num_left_pts:
                # some points in left polygon above upper tangent
                # move left point counter-clockwise
                left_upper_index = np.mod(left_upper_index + 1, num_left_pts)
            elif r_below_cnt < num_right_pts:
                # some points in right polygon are above upper tangent
                # move right point clockwise
                right_upper_index = np.mod(right_upper_index - 1, num_right_pts)

            left_upper_pt = left_polygon[left_upper_index]
            right_upper_pt = right_polygon[right_upper_index]

    # find lower tangent points
    left_lower_index = np.argmax(left_polygon[:, 0])
    right_lower_index = np.argmin(right_polygon[:, 0])

    left_lower_pt = left_polygon[left_lower_index]  # the rightmost point of left polygon
    right_lower_pt = right_polygon[right_lower_index]  # leftmost point of right polygon
    found_lower_tangent = False
    while not found_lower_tangent:
        # for debug
        left_colors = np.zeros(shape=(left_polygon.shape[0], 1), dtype=np.int64)
        right_colors = np.ones(shape=(right_polygon.shape[0], 1), dtype=np.int64) + 1

        left_colors[left_lower_index, 0] += 1
        right_colors[right_lower_index, 0] += 1
        all_colors = np.concatenate((left_colors, right_colors), axis=0)

        # do the check - evaluate every point against line made with left/right tangent points
        l_above_cnt = 0
        r_above_cnt = 0
        for lp in left_polygon:
            e = evaluate_point_line(lp, left_lower_pt, right_lower_pt)
            if e <= 0.0:
                l_above_cnt += 1

        for rp in right_polygon:
            e = evaluate_point_line(rp, left_lower_pt, right_lower_pt)
            if e <= 0.0:
                r_above_cnt += 1

        if l_above_cnt + r_above_cnt == total_pts:
            found_lower_tangent = True
        else:
            if l_above_cnt < num_left_pts:
                # some points in left polygon below lower tangent
                # move left point clockwise
                left_lower_index = np.mod(left_lower_index - 1, num_left_pts)
            elif r_above_cnt < num_right_pts:
                # some points in right polygon are below lower tangent
                # move right point counter-clockwise
                right_lower_index = np.mod(right_lower_index + 1, num_right_pts)

            left_lower_pt = left_polygon[left_lower_index]
            right_lower_pt = right_polygon[right_lower_index]

    return left_upper_index, right_upper_index, left_lower_index, right_lower_index


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


def evaluate_point_line(point, line_start, line_end):
    """ returns positive number if point is below the line, negative number if point above line, zero if point is
        on the line; "above" and "below" determined by ordering of line_start and line_end

    :param point: a point to compare to the line
    :param line_start: the starting point defining the line
    :param line_end:  the ending point defining the line
    :return: float, how the point evaluates in the line equation
    """

    # if the point is exactly on the line then just return zero (and skip any floating point error nonsense)
    if np.array_equal(point, line_start) or np.array_equal(point, line_end):
        return 0.0

    if line_end[0] - line_start[0] != 0:
        slope: float = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])

        return (slope * (point[0] - line_start[0])) - (point[1] - line_start[1])

    else:
        # line is vertical, so only x-coordinate determines point's side of line
        return point[0] - line_start[0]
