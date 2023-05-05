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

            # assert(above_line + below_line >= num_points + 2)  # the two points that make line should be double counted

            if above_line == num_points or below_line == num_points:
                # if all the points were on one side of the line, find which order the points should go in (CCW)

                # pick any point, not i or j to orient hull points
                rand_index = np.random.randint(0, num_points)
                while rand_index == i or rand_index == j:
                    rand_index = np.random.randint(0, num_points)
                any_pt = input_points[rand_index]

                vec1 = point_j - point_i  # vec from i to j
                vec2 = any_pt - point_j  # vec from any to j

                cross_product = np.cross(vec1, vec2)

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
    # assert(unsorted_ordering[0][0] == unsorted_ordering[-1][-1])
    pass


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


def convexhull_3d(input_points: NDFloatArray) -> NDIntArray:
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

    assert(num_points >= 4)
    assert(dim == 3)

    # compute the Convex Hull
    result_points = convexhull_3d_return_points(input_points)

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


def convexhull_3d_return_points(input_points: NDFloatArray) -> NDFloatArray:
    """ Convex Hull function that does the dividing and conquering. Returns list of points, not binary array.

    :param input_points: 2D array representing input points
    :return: 2D array, a subset of input_points that make up the hull
    """
    num_points: int = input_points.shape[0]
    result_points = None

    # base case
    if num_points <= 12:
        result_points = brute_hull_3d(input_points)

    # recursive case
    else:
        # divide
        sorted_points = input_points[np.argsort(input_points[:, 0])]  # sort by x-coord to split
        middle_index = num_points // 2
        left_half = sorted_points[0:middle_index, :]
        right_half = sorted_points[middle_index:, :]

        # conquer
        left_hull_points = convexhull_3d_return_points(left_half)
        right_hull_points = convexhull_3d_return_points(right_half)

        # TODO: merge
        # first find the upper and lower tangents to the two hull
        # use tangents to decide which points to keep from left and right hulls
        up_tan_left, up_tan_right, lo_tan_left, lo_tan_right = find_tangent_lines(left_hull_points, right_hull_points)

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


def brute_hull_3d(input_points: NDFloatArray) -> NDFloatArray:
    """ Brute force computation for finding the convex hull of 3d points

    :param input_points: small set of input points to compute the convex hull of
    :return: 2D array representing the points that are on the hull
    """
    initial_num_points = input_points.shape[0]
    assert(4 <= initial_num_points <= 12)

    # visualize.draw3D(input_points, np.zeros((initial_num_points, 1), dtype=np.int64), "Input Points", label_points=True)
    # pre-process - remove the middles of co-linear points
    points_to_remove = set()
    for i in range(initial_num_points):
        for j in range(i+1, initial_num_points):
            for k in range(j+1, initial_num_points):
                point_i = input_points[i]
                point_j = input_points[j]
                point_k = input_points[k]

                vec1 = point_j - point_i  # vec from i to j
                vec2 = point_k - point_j  # vec from k to j

                if np.linalg.norm(np.cross(vec1, vec2)) < 0.001:  # so close to co-linear
                    ij = np.linalg.norm(point_i - point_j)
                    ik = np.linalg.norm(point_i - point_k)
                    jk = np.linalg.norm(point_j - point_k)

                    m = max(ij, jk, ik)
                    if m == ij:
                        points_to_remove.add(k)
                    elif m == jk:
                        points_to_remove.add(i)
                    elif m == ik:
                        points_to_remove.add(j)
                    else:
                        assert(False, "This should be impossible")

    keep_points = [c not in points_to_remove for c in range(initial_num_points)]
    input_points_processed = input_points[keep_points]

    hull_faces = []  # to fill with triples of indices of points
    hull_points = set()

    # Check every triple of points to see if it is a hull face
    num_points = input_points_processed.shape[0]
    for i in range(num_points):
        for j in range(i+1, num_points):
            for k in range(j+1, num_points):

                # draw a plane between three points (point_i, point_j, point_k)
                # check if all the other points are on one side of the plane
                # if yes, this is a face on the hull

                point_i = input_points_processed[i]
                point_j = input_points_processed[j]
                point_k = input_points_processed[k]

                vec1 = point_j - point_i  # vec from i to j
                vec2 = point_k - point_j  # vec from k to j

                if np.sum(np.cross(vec1, vec2)) == 0.0:
                    # points are co-linear
                    assert(False, "We removed these already")
                    pass

                above_plane, below_below = 0, 0
                for p in input_points_processed:
                    eval_pt = evaluate_point_plane(p, point_i, point_j, point_k)
                    if eval_pt <= 0.0:
                        above_plane += 1
                    if eval_pt >= 0.0:
                        below_below += 1

                # Sanity check: every point is counted once, at least the three points that make plane
                #                should be double counted. (More points can be double-counted if co-planar)
                assert(above_plane + below_below >= num_points + 3)

                if above_plane == num_points or below_below == num_points:
                    # if all the points were on one side of the plane,
                    #    add the points to the set of hull points
                    #    find which order the points should go in (CCW) to make a face

                    hull_points.add(i)
                    hull_points.add(j)
                    hull_points.add(k)

                    # orientation
                    # vec1 = point_j - point_i  # vec from i to j
                    # vec2 = point_k - point_j  # vec from k to j
                    # face_normal = np.cross(vec1, vec2)
                    # other_point = point_j + face_normal
                    #
                    # rand_index = np.random.randint(0, num_points)
                    # while rand_index == i or rand_index == j or rand_index == k or \
                    #         evaluate_point_plane(input_points[rand_index], point_i, point_j, point_k) == 0.0:
                    #     rand_index = np.random.randint(0, num_points)
                    # rand_point = input_points[rand_index]
                    #
                    # eval_other_pt = evaluate_point_plane(other_point, point_i, point_j, point_k)
                    # eval_rand_pt = evaluate_point_plane(rand_point, point_i, point_j, point_k)
                    # first, second, third = i, j, k
                    #
                    # if (eval_other_pt > 0 and eval_rand_pt > 0) or (eval_other_pt < 0 and eval_rand_pt < 0):
                    #     # winding is wrong, flip it
                    #     second, third = k, j

                    # add the edge to the list
                    hull_faces.append((i, j, k))

    assert(len(hull_faces) >= 4)  # a hull must be at least a tetrahedron

    # turn list of indices into list of points
    result_points = []
    for point in hull_points:
        result_points.append(input_points_processed[point])

    result_array = np.array(result_points)

    return result_array


def find_tangent_lines(left_polygon:NDFloatArray, right_polygon: NDFloatArray) -> Tuple[int, int, int, int]:
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
            e = evaluate_point_plane(lp, left_upper_pt, right_upper_pt)
            if e >= 0.0:
                l_below_cnt += 1

        for rp in right_polygon:
            e = evaluate_point_plane(rp, left_upper_pt, right_upper_pt)
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
            e = evaluate_point_plane(lp, left_lower_pt, right_lower_pt)
            if e <= 0.0:
                l_above_cnt += 1

        for rp in right_polygon:
            e = evaluate_point_plane(rp, left_lower_pt, right_lower_pt)
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


def evaluate_point_plane(point, plane_a, plane_b, plane_c):
    """ returns positive number if point is below the plane, negative number if point above plane, zero if point is
        on the plane; "above" and "below" determined by ordering of plane points

    :param point: a point to compare to the line
    :param plane_a: the first point to define the plane
    :param plane_b: the second point to define the plane
    :param plane_c: the third point to define the plane
    :return: float, how the point evaluates in the plane equation
    """

    # use the determinant of matrix to decide which side of plane point is on.
    # See https://math.stackexchange.com/a/2687168 for explanation

    if np.array_equal(point, plane_a) or np.array_equal(point, plane_b) or np.array_equal(point, plane_c):
        return 0.0

    #           |--                                      --|
    #           | point_x     point_y     point_z     1.0  |
    # matrix =  | plane_a_x   plane_a_y   plane_a_z   1.0  |
    #           | plane_b_x   plane_b_y   plane_b_z   1.0  |
    #           | plane_c_x   plane_c_y   plane_c_z   1.0  |
    #           |--                                      --|

    matrix = np.concatenate((np.array([point, plane_a, plane_b, plane_c]), np.ones((4, 1))), axis=1)
    det = np.linalg.det(matrix)

    return det  # zero if point on plane


