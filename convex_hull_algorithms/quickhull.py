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


# returns the extreme points in each dimension. These points are in the input set
def get_min_max_points_nd(input_points: np.ndarray) -> (np.ndarray, np.ndarray):

    num_dimensions = input_points.shape[1]
    #                      dimension,      points
    min_points = np.zeros((num_dimensions, num_dimensions))
    max_points = np.zeros((num_dimensions, num_dimensions))

    # For each dimension, set the default max to neg inf and min to pos inf
    for i in range(num_dimensions):  # For each min/ max point in each dimension
        for j in range(num_dimensions):  # Set each value in this point
            min_points[i, j] = np.inf
            max_points[i, j] = -np.inf

    # For each point in the input_points, check if it is a new min or max point in each dimension
    for point in input_points:
        for i in range(num_dimensions):  # Check if this point is a new min or max point in each dimension
            if point[i] < min_points[i, i]:
                min_points[i] = point
            if point[i] > max_points[i, i]:
                max_points[i] = point

    return min_points, max_points


# Note: winding order is important for the 3D case. The plane points must follow the right hand rule
def distance_to_plane(point, plane_points):
    # Get the normal vector of the plane
    v1 = plane_points[0] - plane_points[1]
    v2 = plane_points[2] - plane_points[1]
    normal = np.cross(v1, v2)

    # Get the distance of the point to the plane
    dist = np.dot(normal, point - plane_points[0]) / np.linalg.norm(normal)

    return dist


def find_hull_points_3d(input_points, plane_points):
    # Base case: no points outside of the plane
    if len(input_points) == 0:
        return None

    upper_hull_points = []
    result_points = []

    # Find the furthest point from the plane
    max_distance = 0.0
    furthest_point = []
    for p in input_points:
        dist = distance_to_plane(p, plane_points)
        if dist > max_distance:
            max_distance = dist
            furthest_point = p

    if furthest_point != []:
        result_points.append(furthest_point)

    ccw_points = [plane_points[0], plane_points[1], furthest_point]
    cw_points = [furthest_point, plane_points[1], plane_points[0]]

    r1 = find_hull_points_3d(upper_hull_points, ccw_points)
    r2 = find_hull_points_3d(upper_hull_points, cw_points)

    # Concatenate the results
    if r1 is not None:
        result_points.extend(r1)

    if r2 is not None:
        result_points.extend(r2)

    if result_points == []:
        return None
    else:
        return result_points


def convexhull3d(input_points: NDFloatArray) -> NDIntArray:
    assert(len(input_points.shape) == 2)  # check that input points is 2D array

    # some useful attributes of input points
    num_points: int = input_points.shape[0]
    dim: int = input_points.shape[1]

    print(f"Num Points: {num_points}")
    print(f"Dimension: {dim}")

    result_points = []
    min_points, max_points = get_min_max_points_nd(input_points)

    # Add the min and max points to the result since they must be on the convex hull
    # Note: duplicates are not removed, we handle this later
    for point in min_points:
        result_points.append(point)
    for point in max_points:
        result_points.append(point)

    print(f"Min Points: {min_points}")
    print(f"Max Points: {max_points}")

    ccw_points = [min_points[0], min_points[1], min_points[2]]
    cw_points = [min_points[2], min_points[1], min_points[0]]
    upper = find_hull_points_3d(input_points, ccw_points)
    lower = find_hull_points_3d(input_points, cw_points)

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
