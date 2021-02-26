import geneticalgorithm2 as ga
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def get_employees():
    return {'Amy':   10,
            'Bob':   12,
            'Cathy': 10,
            'Dan':    8,
            'Ed':     8,
            'Fred':   9,
            'Gu':    11}


def get_shifts_requirements():
    return {0:  3,
            1:  2,
            2:  4,
            3:  4,
            4:  5,
            5:  6,
            6:  5,
            7:  2,
            8:  2,
            9:  3,
            10: 4,
            11: 6,
            12: 7,
            13: 5}


def get_employees_shift_availabilities():
    return {'Amy':   [1, 2, 4, 6, 8, 9, 10, 11, 12, 13],
            'Bob':   [0, 1, 4, 5, 7, 10, 12],
            'Cathy': [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13],
            'Dan':   [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13],
            'Ed':    [0, 1, 2, 3, 4, 6, 7, 8, 10, 12, 13],
            'Fred':  [0, 1, 2, 5, 7, 8, 11, 12, 13],
            'Gu':    [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}


def get_employee(i):
    return list(get_employees().keys())[i]


def get_employees_number():
    return len(get_employees())


def get_shift(i):
    return get_shifts_requirements()[i]


def get_shifts_number():
    return len(get_shifts_requirements())


def check_shift_availabilities(x):
    x = x.reshape((get_employees_number(), get_shifts_number()))
    conflicts = 0

    for i in range(get_employees_number()):
        for j in range(get_shifts_number()):
            if x[i][j] == 1 and j not in get_employees_shift_availabilities()[get_employee(i)]:
                conflicts += 1

    return conflicts


def check_shift_requirements(x):
    x = x.reshape((get_employees_number(), get_shifts_number()))
    slack = []

    for j in range(get_shifts_number()):
        shifts = 0

        for i in range(get_employees_number()):
            shifts += x[i][j]

        requirements = get_shift(j)
        slack.append(requirements - shifts if requirements - shifts > 0 else 0)

    return slack


def check_maximum_minimum_shifts(x):
    x = x.reshape((get_employees_number(), get_shifts_number()))

    maximum = 0
    minimum = get_shifts_number()

    for i in range(get_employees_number()):
        total = 0

        for j in range(get_shifts_number()):
            total += x[i][j]

        if total > maximum:
            maximum = total
        if total < minimum:
            minimum = total

    return maximum - minimum


def objective_function(x):
    x = x.astype(int)

    cost = 0

    slack = check_shift_requirements(x)
    for i in slack:
        if i > 0:
            cost += 10 * i

    conflicts = check_shift_availabilities(x)
    if conflicts > 0:
        cost += 1000 * conflicts

    delta = check_maximum_minimum_shifts(x)
    cost += 10 * delta

    return cost


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


def print_algorithm_solution(solution):
    solution = solution.reshape((get_employees_number(), get_shifts_number()))
    employees = list(get_employees().keys())
    
    print()
    
    employees_shifts = []
    results = []
    for i in range(len(employees)):
        total_employee_shifts = 0
        result = [employees[i]]

        for j in range(get_shifts_number()):
            total_employee_shifts += solution[i][j]
            result.append('*' if solution[i][j] == 1 else '-')

        employees_shifts.append(total_employee_shifts)
        results.append(result)

        print('{}: {}'.format(employees[i], int(total_employee_shifts)))

    print('\nSlack: {}\n'.format(sum(check_shift_requirements(solution))))
    print(pd.DataFrame.from_records(results, columns=['worker'] + [str(i) for i in range(get_shifts_number())]))


def genetic_algorithm(generations, population, elit, studEA):
    model = ga.geneticalgorithm2(function=objective_function,
                                 dimension=get_employees_number() * get_shifts_number(),
                                 variable_type='bool',
                                 algorithm_parameters={'max_num_iteration': generations,
                                                       'population_size': population,
                                                       'elit_ratio': elit})

    model.run(no_plot=True,
              studEA=studEA)

    plot_algorithm_progress(np.array(model.report))
    plot_algorithm_generation(model.output_dict['last_generation']['scores'])
    print_algorithm_solution(model.output_dict['variable'])


if __name__ == '__main__':
    genetic_algorithm(generations=1000,
                      population=100,
                      elit=0.01,
                      studEA=True)
