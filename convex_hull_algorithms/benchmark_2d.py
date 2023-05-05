import timeit
import numpy as np
import math
import quickhull
import divide_and_conquer


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

    random_data_1 = np.random.uniform(0.0, 10 ** 1, (10,     2))
    random_data_2 = np.random.uniform(0.0, 10 ** 2, (100,    2))
    random_data_3 = np.random.uniform(0.0, 10 ** 3, (1000,   2))
    random_data_4 = np.random.uniform(0.0, 10 ** 4, (10000,  2))
    random_data_5 = np.random.uniform(0.0, 10 ** 5, (100000, 2))

    wc_data_1 = qh_worst_case_data_2d(10 ** 1,  10)
    wc_data_2 = qh_worst_case_data_2d(10 ** 15, 97)
    wc_data_3 = qh_worst_case_data_2d(10 ** 15, 1000)
    wc_data_4 = qh_worst_case_data_2d(10 ** 15, 10000)
    wc_data_5 = qh_worst_case_data_2d(10 ** 15, 100000)

    tests = {
        "Random Data - 10 Points":      random_data_1,
        "Random Data - 100 Points":     random_data_2,
        "Random Data - 1,000 Points":   random_data_3,
        "Random Data - 10,000 Points":  random_data_4,
        "Random Data - 100,000 Points": random_data_5,
        "WC Data - 10 Points":          wc_data_1,
        "WC Data - 100 Points":         wc_data_2,
        "WC Data - 1,000 Points":       wc_data_3,
        "WC Data - 10,000 Points":      wc_data_4,
        "WC Data - 100,000 Points":     wc_data_5
    }

    for name, data in tests.items():
        print(name)
        qh_time = timeit.timeit(lambda: quickhull.convexhull(data), number=5)
        print("  QH: ", qh_time)

        dc_time = timeit.timeit(lambda: divide_and_conquer.convexhull_2d(data), number=5)
        print("  DC: ", dc_time)



