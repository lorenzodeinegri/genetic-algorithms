import geneticalgorithm2 as ga
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def get_queens():
    return 16


def check_vertical_conflicts(x):
    total_conflicts = []

    for i in range(get_queens()):
        conflicts = 0

        for j in np.linspace(i, i + get_queens() * (get_queens() - 1), get_queens()):
            conflicts += x[int(j)]

        total_conflicts.append(conflicts)

    return total_conflicts


def check_horizontal_conflicts(x):
    total_conflicts = []

    for i in range(get_queens()):
        conflicts = 0

        for j in range(i * get_queens(), get_queens() + i * get_queens()):
            conflicts += x[j]

        total_conflicts.append(conflicts)

    return total_conflicts


def check_major_diagonal_conflicts(x):
    x = x.reshape(get_queens(), get_queens())
    total_conflicts = []

    for i in range(get_queens()):
        conflicts = 0

        for j in range(get_queens() - i):
            conflicts += x[i + j][j]

        total_conflicts.append(conflicts)

    for j in range(1, get_queens()):
        conflicts = 0

        for i in range(get_queens() - j):
            conflicts += x[i][j + i]

        total_conflicts.append(conflicts)

    return total_conflicts


def check_minor_diagonal_conflicts(x):
    x = x.reshape(get_queens(), get_queens())
    total_conflicts = []

    for j in range(get_queens()):
        conflicts = 0

        for i in range(j + 1):
            conflicts += x[i][j - i]

        total_conflicts.append(conflicts)

    for i in range(1, get_queens()):
        conflicts = 0

        for j in range(get_queens() - 1, i - 1, -1):
            conflicts += x[get_queens() - 1 + i - j][j]

        total_conflicts.append(conflicts)

    return total_conflicts


def objective_function(x):
    x = x.astype(int)

    cost = 0

    vertical = check_vertical_conflicts(x)
    for i in vertical:
        if i > 0:
            cost += (i - 1) * 1000

    horizontal = check_horizontal_conflicts(x)
    for i in horizontal:
        if i > 0:
            cost += (i - 1) * 1000

    major_diagonal = check_major_diagonal_conflicts(x)
    for i in major_diagonal:
        if i > 0:
            cost += (i - 1) * 100

    minor_diagonal = check_minor_diagonal_conflicts(x)
    for i in minor_diagonal:
        if i > 0:
            cost += (i - 1) * 100

    if sum(x) != get_queens():
        cost += get_queens() * 1000000

    return cost


def plot_algorithm_progress(data, name):
    plt.figure(figsize=(12.8, 7.2))

    plt.title('Algorithm score progress')
    plt.legend('Best of each generation')

    plt.xlabel('Iteration number')
    plt.ylabel('Objective function value')

    plt.plot(data)
    # plt.show()

    plt.savefig(fname=name, format='png')


def plot_algorithm_generation(data, name):
    plt.figure(figsize=(12.8, 7.2))

    plt.title('Algorithm last generation scores')
    plt.xticks([], [])

    plt.xlabel('Generation solutions')
    plt.ylabel('Objective function values')

    plt.bar(np.flip(np.arange(len(data))), data)
    # plt.show()

    plt.savefig(fname=name, format='png')


def print_algorithm_solution(solution):
    results = []
    for i in range(get_queens()):
        result = [str(i)]

        for j in range(i * get_queens(), get_queens() + i * get_queens()):
            result.append('*' if solution[j] == 1 else '-')

        results.append(result)

    print()
    print(pd.DataFrame.from_records(results, columns=['x'] + [str(i) for i in range(get_queens())]))


def genetic_algorithm(generations, population, elit, studEA, types):
    model = ga.geneticalgorithm2(function=objective_function,
                                 dimension=get_queens() ** 2,
                                 variable_type='bool',
                                 algorithm_parameters={'max_num_iteration': generations,
                                                       'population_size': population,
                                                       'elit_ratio': elit})

    model.run(no_plot=True,
              studEA=studEA)

    plot_algorithm_progress(np.array(model.report), '16_Convergence_' + types + '.png')
    plot_algorithm_generation(model.output_dict['last_generation']['scores'], '16_Scores_' + types + '.png')
    print_algorithm_solution(model.output_dict['variable'])


if __name__ == '__main__':
    # print('Standard_1000')
    # genetic_algorithm(generations=1000,
    #                   population=100,
    #                   elit=0.0,
    #                   studEA=False,
    #                   types='Standard_1000')
    #
    # print('\n\nElitist_1000')
    # genetic_algorithm(generations=1000,
    #                   population=100,
    #                   elit=0.01,
    #                   studEA=False,
    #                   types='Elitist_1000')

    print('\n\nElitist_Stud_1000')
    genetic_algorithm(generations=1000,
                      population=100,
                      elit=0.01,
                      studEA=True,
                      types='Elitist_Stud_1000')

    exit(0)

    print('\n\nStandard_3000')
    genetic_algorithm(generations=3000,
                      population=100,
                      elit=0.0,
                      studEA=False,
                      types='Standard_3000')

    print('\n\nElitist_3000')
    genetic_algorithm(generations=3000,
                      population=100,
                      elit=0.01,
                      studEA=False,
                      types='Elitist_3000')

    print('\n\nElitist_Stud_3000')
    genetic_algorithm(generations=3000,
                      population=100,
                      elit=0.01,
                      studEA=True,
                      types='Elitist_Stud_3000')

    print('\n\nStandard_9000')
    genetic_algorithm(generations=9000,
                      population=100,
                      elit=0.0,
                      studEA=False,
                      types='Standard_9000')

    print('\n\nElitist_9000')
    genetic_algorithm(generations=9000,
                      population=100,
                      elit=0.01,
                      studEA=False,
                      types='Elitist_9000')

    print('\n\nElitist_Stud_9000')
    genetic_algorithm(generations=9000,
                      population=100,
                      elit=0.01,
                      studEA=True,
                      types='Elitist_Stud_9000')
