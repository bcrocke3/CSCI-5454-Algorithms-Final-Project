# python imports
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# types for n-dimensional arrays
NDFloatArray = npt.NDArray[np.float64]
NDIntArray = npt.NDArray[np.int64]


def draw2D(points: NDFloatArray, colors: NDIntArray, title: str = "Convex Hull Results") -> None:
    """ Plots 2D points in multiple colors

    :param points: 2D array of points to plot. Each row is a point. Must have 2 columns for x, y dimensions
    :param colors: Column vector of category/color that corresponds to each point
    :param title: title of output graph. Should describe how input data was computed
    :return: Nothing. Draws plots.
    """
    # check assumptions
    assert(len(points.shape) == 2)  # 2D points matrix
    assert(points.shape[1] == 2)    # all the points are 2D (2, y)
    assert(len(colors.shape) == 2)  # colors is column vector
    assert(colors.shape[1] == 1)
    assert(points.shape[0] == colors.shape[0])  # points and colors have same # of rows

    # define colors
    colormap = np.array(['black', 'red'])

    # graph
    plt.scatter(points[:, 0], points[:, 1], c=colormap[colors[:, 0]])
    plt.title(title)
    plt.show()


def draw3D(points: NDFloatArray, colors: NDIntArray, title: str = "Convex Hull Results") -> None:
    """ Plots 3D points in multiple colors

    :param points: 2D array of points to plot. Each row is a point. Must have 3 columns for x, y, z dimensions
    :param colors: Column vector of category/color that corresponds to each point
    :param title: title of output graph. Should describe how input data was computed
    :return: Nothing. Draws plots.
    """
    # check assumptions
    assert(len(points.shape) == 2)  # 2D points matrix
    assert(points.shape[1] == 3)    # all the points are 3D (3, y)
    assert(len(colors.shape) == 2)  # colors is column vector
    assert(colors.shape[1] == 1)
    assert(points.shape[0] == colors.shape[0])  # points and colors have same # of rows

    # define colors
    colormap = np.array(['black', 'red'])

    # graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colormap[colors[:, 0]])
    plt.title(title)
    plt.show()


def interact3d(points: NDFloatArray, colors: NDIntArray, face_indices, face_points, orgi_faces=[], orgi_points=[]) -> None:

    colormap = np.array(['black', 'red'])

    fig = go.Figure()

    # Extract the x, y, z from the points array
    points_x = []
    points_y = []
    points_z = []
    for point in points:
        points_x.append(point[0])
        points_y.append(point[1])
        points_z.append(point[2])

    if orgi_faces != [] and orgi_points != []:
        orgi_mesh = go.Mesh3d(x=orgi_points[:, 0],
                              y=orgi_points[:, 1],
                              z=orgi_points[:, 2],
                              i=orgi_faces[:, 0],
                              j=orgi_faces[:, 1],
                              k=orgi_faces[:, 2], showscale=True, opacity=1.0)
        fig.add_trace(orgi_mesh)

    mesh = go.Mesh3d(x=face_points[:, 0],
                     y=face_points[:, 1],
                     z=face_points[:, 2],
                     i=face_indices[:, 0],
                     j=face_indices[:, 1],
                     k=face_indices[:, 2],
                     showscale=True,
                     opacity=0.5)
    fig.add_trace(mesh)

    colors = colormap[colors[:, 0]]

    scatter = go.Scatter3d(
                x=points_x,
                y=points_y,
                z=points_z,
                marker=dict(size=6, color=colors, colorscale='Viridis', opacity=0.8),
                mode='markers')
    fig.add_trace(scatter)

    fig.show()
