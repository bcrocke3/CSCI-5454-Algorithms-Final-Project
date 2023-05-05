import numpy as np
import math

import visualize

def qh_worst_case_data_2d(arc_radius: float, num_points: int):
    result = []
    result.append(np.array([0.0, 0.0]))

    def arc_point(angle: float):
        x = arc_radius * math.cos(angle) + arc_radius
        y = arc_radius * math.sin(angle)
        return np.array([x, y])

    current_angle = 0  # radians
    step_size = math.pi / 2.0
    for i in range(num_points):

        ap = arc_point(current_angle)

        # ap[0] += 0.9 * arc_radius
        result.append(ap)
        current_angle += step_size
        step_size /= 2.0

        if np.linalg.norm(result[-1]) < 0.0001:
            break

    if len(result) < num_points:
        print("Couldn't generate enough points")
        print(f"{len(result)} / {num_points} points generated")

    return np.array(result)


if __name__ == '__main__':
    data = qh_worst_case_data_2d(10 ** 1,  10)
    data = qh_worst_case_data_2d(10 ** 12, 100)
    data = qh_worst_case_data_2d(10 ** 13, 1000)
    data = qh_worst_case_data_2d(10 ** 13, 10000)
    data = qh_worst_case_data_2d(10 ** 13, 100000)
    colors = np.zeros((data.shape[0], 1), dtype=np.int64)

    visualize.draw2D(data, colors, "Quickhull Worst Case Input Data")
    # print(data)
