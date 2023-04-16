# python imports
import numpy as np
import numpy.typing as npt

# types for n-dimensional arrays
NDFloatArray = npt.NDArray[np.float64]
NDIntArray = npt.NDArray[np.int64]


def get_min_max_points(input_points: NDFloatArray) -> (NDFloatArray, NDFloatArray):
    # Find points from the input_points with min and max x values, if there are ties, take the point with the
    # min/ max y value
    min_point = np.array([np.inf, np.inf])
    max_point = np.array([-np.inf, -np.inf])
    for point in input_points:
        if point[0] < min_point[0]:
            min_point = point
        elif point[0] == min_point[0]:
            if point[1] < min_point[1]:
                min_point = point
        if point[0] > max_point[0]:
            max_point = point
        elif point[0] == max_point[0]:
            if point[1] > max_point[1]:
                max_point = point
    return min_point, max_point


def point_position(point, line_start, line_end):
    # Calculate the vectors from the line start to the end and from the start to the point
    line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    point_vector = (point[0] - line_start[0], point[1] - line_start[1])

    # Calculate the cross product of the two vectors
    cross_product = line_vector[0] * point_vector[1] - line_vector[1] * point_vector[0]

    # Determine the position of the point relative to the line
    if cross_product > 0:
        return 1
    elif cross_product < 0:
        return -1
    else:
        # The point is collinear with the line, check if it is on the line or not
        dot_product = line_vector[0] * point_vector[0] + line_vector[1] * point_vector[1]
        if dot_product < 0:
            return -1
        elif dot_product > line_vector[0] ** 2 + line_vector[1] ** 2:
            return 1
        else:
            return 0


def dist_to_line(point, line_start, line_end):
    # Calculate the distance from the point to the line formed by the line start and end points
    line_vector = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    point_vector = (point[0] - line_start[0], point[1] - line_start[1])

    cross_product = line_vector[0] * point_vector[1] - line_vector[1] * point_vector[0]
    return abs(cross_product) / np.sqrt(line_vector[0] ** 2 + line_vector[1] ** 2)


def find_hull_points(input_points, p0, p1):
    # Base case: no points to the left of line
    if len(input_points) == 0:
        return None  # no points on the hull

    upper_hull_points = []
    result_points = []

    max_distance = 0.0
    furthest_point = []
    for p in input_points:
        if point_position(p, p0, p1) > 0:  # if p is on the left side of the line
            upper_hull_points.append(p)
            dist = dist_to_line(p, p0, p1)
            if dist > max_distance:
                max_distance = dist
                furthest_point = p

    if furthest_point != []:
        result_points.append(furthest_point)

    # Recursively find the hull points outside of the two new lines formed by the triangle: region 1 and region 2
    # (r1 & r2). We don't care about the points in region 3 since they are inside the triangle. fp is furthest_point
    #
    #           r2
    #     fp- - - - -p1
    #     |  r3    /
    #  r1 |     /
    #     |  /
    #     p0
    r1 = find_hull_points(upper_hull_points, p0, furthest_point)
    r2 = find_hull_points(upper_hull_points, furthest_point, p1)

    # Concatenate the results
    if r1 is not None:
        result_points.extend(r1)

    if r2 is not None:
        result_points.extend(r2)

    if result_points == []:
        return None
    else:
        return result_points


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

    result_points = []

    # Step 1: Get min and max points
    min_point, max_point = get_min_max_points(input_points)
    result_points.append(min_point)
    result_points.append(max_point)

    upper = find_hull_points(input_points, min_point, max_point)
    lower = find_hull_points(input_points, max_point, min_point)

    if upper is not None:
        result_points.extend(upper)

    if lower is not None:
        result_points.extend(lower)

    # Default result - no point is on the convex hull
    result: NDIntArray = np.zeros((num_points, 1), dtype=np.int64)

    # Set the result to 1 for points that are on the convex hull
    for point in result_points:
        for i in range(num_points):
            if (point[0] == input_points[i, 0]) and (point[1] == input_points[i, 1]):
                result[i] = 1

    # check that result is correct shape
    assert(result.shape[0] == input_points.shape[0])
    assert(result.shape[1] == 1)

    return result
