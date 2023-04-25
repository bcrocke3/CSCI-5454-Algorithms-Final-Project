# python imports
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd

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
    assert (len(input_points.shape) == 2)  # check that input points is 2D array

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
    assert (result.shape[0] == input_points.shape[0])
    assert (result.shape[1] == 1)

    return result


def is_less(a: np.ndarray, b: np.ndarray, dim: int, max_dim: int, count: int) -> bool:
    # Base case: we have checked all dimensions, they are the same
    if count == max_dim:
        return False

    if a[dim] == b[dim]:
        return is_less(a, b, (dim + 1) % max_dim, max_dim, count + 1)
    elif a[dim] < b[dim]:
        return True
    else:
        return False


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
            if is_less(point, min_points[i], i, num_dimensions, 0):
                min_points[i] = point
            if not is_less(point, max_points[i], i, num_dimensions, 0):
                max_points[i] = point

    return min_points, max_points


def compute_max_pair(min_points: np.ndarray, max_points: np.ndarray) -> (np.ndarray, np.ndarray):
    # Find the pair of points that are furthest apart
    max_dist = 0
    max_pair = (None, None)

    for i in range(min_points.shape[0]):
        for j in range(max_points.shape[0]):
            dist = np.linalg.norm(min_points[i] - max_points[j])
            if dist > max_dist:
                max_dist = dist
                max_pair = (min_points[i], max_points[j])

    return max_pair


def farthest_point_from_line(points: np.ndarray, line: (np.ndarray, np.ndarray)) -> np.ndarray:
    # Find the point that is furthest from the line
    max_dist = 0
    max_point = None

    for point in points:
        dist = dist_to_line(point, line[0], line[1])
        if dist > max_dist:
            max_dist = dist
            max_point = point

    return max_point


class Edge:  # Make an object of type Edge which have two points denoting the vertices of the edges
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

    def __hash__(self):
        return hash((self.p0[0], self.p0[1], self.p0[2], self.p1[0], self.p1[1], self.p1[2]))

    def __eq__(self, b):
        return (np.array_equal(self.p0, b.p0) and np.array_equal(self.p1, b.p1)) \
            or (np.array_equal(self.p1, b.p0) and np.array_equal(self.p0, b.p1))


class Plane:
    def __init__(self, p0, p1, p2):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.normal = np.cross(np.subtract(p1, p0), np.subtract(p2, p0))
        self.todo = set()

    def orient_normal_from_points(self, points):
        for point in points:
            dist = np.dot(self.normal, point - self.p0)
            if dist != 0 and dist > 10**-10:
                self.normal[0] = -1 * self.normal[0]
                self.normal[1] = -1 * self.normal[1]
                self.normal[2] = -1 * self.normal[2]
                return

    def dist(self, point):
        return np.dot(self.normal, np.subtract(point, self.p0))

    def calculate_to_do(self, points):
        for p in points:
            dist = self.dist(p)
            if dist > 10**-10:
                self.todo.add((p[0], p[1], p[2]))

    def get_edges(self):
        return [Edge(self.p0, self.p1), Edge(self.p1, self.p2), Edge(self.p2, self.p0)]


def distance_to_plane(point, plane: Plane):
    # Get the distance of the point to the plane
    dist = np.dot(plane.normal, point - plane.p0) / np.linalg.norm(plane.normal)
    return dist


def farthest_point_from_plane(points: np.ndarray, plane: Plane) -> np.ndarray:
    # Find the point that is furthest from the plane
    max_dist = 0
    max_point = None

    for point in points:
        dist = distance_to_plane(point, plane)
        if dist > max_dist:
            max_dist = dist
            max_point = point

    return max_point


def find_eye_point(plane, to_do_list):  # Calculate the maximum distance from the plane
    max_dist = 0
    max_dist_point = None
    for point in to_do_list:
        dist = plane.dist(point)
        if dist > max_dist:
            max_dist = dist
            max_dist_point = point

    return max_dist_point


def adjacent_plane(main_plane, edge, planes):  # Finding adjacent planes to an edge
    for plane in planes:
        edges = plane.get_edges()
        if (plane != main_plane) and (edge in edges):
            return plane


def calc_horizon(visited_planes, plane, eye_point, edge_list, planes):  # Calculating the horizon for an eye to make new faces
    if plane.dist(eye_point) > 10**-10:
        visited_planes.append(plane)
        edges = plane.get_edges()
        for edge in edges:
            neighbour = adjacent_plane(plane, edge, planes)
            if neighbour not in visited_planes:
                result = calc_horizon(visited_planes, neighbour, eye_point, edge_list, planes)
                if result == 0:
                    edge_list.add(edge)
        return 1

    else:
        return 0


def index_of_point(point, points):
    for i in range(len(points)):
        if np.array_equal(point, points[i]):
            return i


def compute_3d_hull(input_points: NDFloatArray):
    result_points = set()

    min_points, max_points = get_min_max_points_nd(input_points)
    print(f"Min Points: {min_points}")
    print(f"Max Points: {max_points}")

    initial_line = compute_max_pair(min_points, max_points)
    print(f"Initial line: {initial_line}")

    first_point = initial_line[0]
    second_point = initial_line[1]
    third_point = farthest_point_from_line(input_points, initial_line)

    first_plane = Plane(first_point, second_point, third_point)

    fourth_point = farthest_point_from_plane(input_points, first_plane)

    second_plane = Plane(first_point, second_point, fourth_point)
    third_plane = Plane(first_point, fourth_point, third_point)
    fourth_plane = Plane(second_point, third_point, fourth_point)

    # Make sure that the normals point away from the center of the hull
    potential_internal_points = [first_point, second_point, third_point, fourth_point]
    first_plane.orient_normal_from_points(potential_internal_points)
    second_plane.orient_normal_from_points(potential_internal_points)
    third_plane.orient_normal_from_points(potential_internal_points)
    fourth_plane.orient_normal_from_points(potential_internal_points)

    first_plane.calculate_to_do(input_points)
    second_plane.calculate_to_do(input_points)
    third_plane.calculate_to_do(input_points)
    fourth_plane.calculate_to_do(input_points)

    planes = [first_plane, second_plane, third_plane, fourth_plane]

    any_unfinished_planes = True

    while any_unfinished_planes:
        any_unfinished_planes = False
        for working_plane in planes:
            if len(working_plane.todo) > 0:
                any_unfinished_planes = True
                eye_point = find_eye_point(working_plane, working_plane.todo)  # Calculate the eye point of the face

                edge_list = set()
                visited_planes = []

                calc_horizon(visited_planes, working_plane, eye_point, edge_list, planes)  # Calculate the horizon

                for internal_plane in visited_planes:  # Remove the internal planes
                    planes.remove(internal_plane)

                for edge in edge_list:  # Make new planes
                    new_plane = Plane(edge.p0, edge.p1, eye_point)
                    new_plane.orient_normal_from_points(potential_internal_points)

                    temp_to_do = set()
                    for internal_plane in visited_planes:
                        temp_to_do = temp_to_do.union(internal_plane.todo)

                    new_plane.calculate_to_do(temp_to_do)

                    planes.append(new_plane)

    for plane in planes:
        result_points.add((plane.p0[0], plane.p0[1], plane.p0[2]))
        result_points.add((plane.p1[0], plane.p1[1], plane.p1[2]))
        result_points.add((plane.p2[0], plane.p2[1], plane.p2[2]))

    # Convert set to list
    result_points = list(result_points)

    # Build a face list using the result points indices
    i = []
    j = []
    k = []
    x = []
    y = []
    z = []
    index = 0
    for plane in planes:
        x.append(plane.p0[0])
        y.append(plane.p0[1])
        z.append(plane.p0[2])
        i.append(index)
        index += 1
        x.append(plane.p1[0])
        y.append(plane.p1[1])
        z.append(plane.p1[2])
        j.append(index)
        index += 1
        x.append(plane.p2[0])
        y.append(plane.p2[1])
        z.append(plane.p2[2])
        k.append(index)
        index += 1

    # Package the face points and indices into a numpy array
    face_points = np.array([x, y, z]).T
    face_indices = np.array([i, j, k]).T

    return result_points, face_indices, face_points


def convexhull3d(input_points: NDFloatArray):
    assert (len(input_points.shape) == 2)  # check that input points is 2D array

    # some useful attributes of input points
    num_points: int = input_points.shape[0]
    dim: int = input_points.shape[1]

    print(f"Num Points: {num_points}")
    print(f"Dimension: {dim}")

    result_points, face_indices, face_points = compute_3d_hull(input_points)
    print(f"Result Points: {result_points}")

    # Default result - no point is on the convex hull
    result: NDIntArray = np.zeros((num_points, 1), dtype=np.int64)

    # Set the result to 1 for points that are on the convex hull
    for point in result_points:
        for i in range(num_points):
            if np.array_equal(point, input_points[i]):
                result[i] = 1

    # check that result is correct shape
    assert (result.shape[0] == input_points.shape[0])
    assert (result.shape[1] == 1)

    return result, face_indices, face_points
