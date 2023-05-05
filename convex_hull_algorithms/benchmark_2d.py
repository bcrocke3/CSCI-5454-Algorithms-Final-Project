import timeit
import numpy as np

import quickhull
import divide_and_conquer
import visualize
import generate_data

if __name__ == '__main__':

    random_data_1 = np.random.uniform(0.0, 10 ** 1, (10,     2))
    random_data_2 = np.random.uniform(0.0, 10 ** 2, (100,    2))
    random_data_3 = np.random.uniform(0.0, 10 ** 3, (1000,   2))
    random_data_4 = np.random.uniform(0.0, 10 ** 4, (10000,  2))
    random_data_5 = np.random.uniform(0.0, 10 ** 5, (100000, 2))

    wc_data_1 = generate_data.qh_worst_case_data_2d(10 ** 1,  10)
    wc_data_2 = generate_data.qh_worst_case_data_2d(10 ** 15, 97)
    wc_data_3 = generate_data.qh_worst_case_data_2d(10 ** 15, 1000)
    wc_data_4 = generate_data.qh_worst_case_data_2d(10 ** 15, 10000)
    wc_data_5 = generate_data.qh_worst_case_data_2d(10 ** 15, 100000)

    tests = {
        # "Random Data - 10 Points":      random_data_1,
        # "Random Data - 100 Points":     random_data_2,
        # "Random Data - 1,000 Points":   random_data_3,
        # "Random Data - 10,000 Points":  random_data_4,
        # "Random Data - 100,000 Points": random_data_5,
        # "WC Data - 10 Points":          wc_data_1,
        "WC Data - 100 Points":         wc_data_2,
        # "WC Data - 1,000 Points":       wc_data_3,
        # "WC Data - 10,000 Points":      wc_data_4,
        # "WC Data - 100,000 Points":     wc_data_5
    }

    for name, data in tests.items():
        print(name)
        qh_time = timeit.timeit(lambda: quickhull.convexhull(data), number=5)
        print("  QH: ", qh_time)

        dc_time = timeit.timeit(lambda: divide_and_conquer.convexhull_2d(data), number=5)
        print("  DC: ", dc_time)



