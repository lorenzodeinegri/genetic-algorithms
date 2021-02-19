import geneticalgorithm2 as ga
import numpy as np
from matplotlib import pyplot as plt


def get_cities():
    return 13


def get_distance(start, finish):
    distances = [[0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
                 [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
                 [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
                 [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
                 [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
                 [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
                 [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
                 [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
                 [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
                 [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
                 [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
                 [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
                 [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0]]
    return distances[start][finish]


def get_maximum_distance():
    distances = [[0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
                 [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
                 [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
                 [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
                 [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
                 [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
                 [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
                 [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
                 [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
                 [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
                 [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
                 [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
                 [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0]]
    return max(map(max, distances))


def objective_function(x):
    x = x.astype(int)

    penalty = get_maximum_distance() * get_cities() if x[0] != 0 else 0
    total_distance = 0

    for i in range(get_cities()):
        total_distance += get_distance(x[i], x[(i + 1) % get_cities()])

        if np.count_nonzero(x == x[i]) > 1:
            penalty += get_maximum_distance() * get_cities()

    return total_distance + penalty


def plot_algorithm_progress(data):
    plt.figure(figsize=(12.8, 7.2))

    plt.title('Algorithm score progress')
    plt.legend('Best of each generation')

    plt.xlabel('Iteration number')
    plt.ylabel('Objective function value')

    plt.plot(data)
    plt.show()


def plot_algorithm_generation(data):
    plt.figure(figsize=(12.8, 7.2))

    plt.title('Algorithm last generation scores')
    plt.xticks([], [])

    plt.xlabel('Generation solutions')
    plt.ylabel('Objective function values')

    plt.bar(np.flip(np.arange(len(data))), data)
    plt.show()


def print_algorithm_solution(solution, score):
    route = '\nCities route: '
    for i in solution:
        route += str(int(i)) + ' -> '
    route += '0'

    print(route)
    print('Miles: {}'.format(score))


def genetic_algorithm():
    model = ga.geneticalgorithm2(function=objective_function,
                                 dimension=get_cities(),
                                 variable_type='int',
                                 variable_boundaries=np.array([[0, 12]] * get_cities()),
                                 algorithm_parameters={'max_num_iteration': 10000})

    model.run(no_plot=True)

    plot_algorithm_progress(np.array(model.report))
    plot_algorithm_generation(model.output_dict['last_generation']['scores'])
    print_algorithm_solution(model.output_dict['variable'], model.output_dict['function'])


if __name__ == '__main__':
    genetic_algorithm()
