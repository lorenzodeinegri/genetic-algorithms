import geneticalgorithm2 as ga
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def get_employees():
    return 4


def get_days():
    return 3


def get_shifts():
    return 3


def get_shift_requests():
    return [[[0, 0, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [1, 0, 0]]]


# def get_shift_requests():
#     return [[[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]],
#             [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]],
#             [[0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
#             [[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
#             [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]]


def check_shift_to_single_employee_per_day(x):
    days = []

    for i in range(get_days()):
        shifts = []

        for j in range(get_shifts()):
            shifts_per_day = 0

            for k in np.linspace(j + i * get_shifts(),
                                 j + i * get_shifts() + get_shifts() * get_days() * (get_employees() - 1),
                                 get_employees()):
                shifts_per_day += x[int(k)]

            shifts.append(shifts_per_day)

        days.append(shifts)

    return days == [[1] * get_shifts()] * get_days()


def check_employee_one_shift_per_day(x):
    days = []

    for i in range(get_employees()):
        shifts = []

        for j in range(get_days()):
            shifts_per_day = 0

            for k in range(get_shifts()):
                shifts_per_day += x[int(k + j * get_shifts() + i * get_shifts() * get_days())]

            shifts.append(shifts_per_day)

        days.append(shifts)

    return all(shifts <= 1 for day in days for shifts in day)


def check_employee_shifts_distribution(x):
    minimum_shifts = (get_shifts() * get_days()) // get_employees()
    maximum_shifts = minimum_shifts + (0 if get_shifts() * get_days() % get_employees() == 0 else 1)

    shifts_worked = []
    for i in range(get_employees()):
        shifts = 0

        for j in range(get_days()):
            for k in range(get_shifts()):
                shifts += x[int(k + j * get_shifts() + i * get_shifts() * get_days())]

        shifts_worked.append(shifts)

    return all(minimum_shifts <= shifts for shifts in shifts_worked) and all(shifts <= maximum_shifts for shifts in shifts_worked)


def check_met_requests(x):
    requests = get_shift_requests()
    shifts = x.reshape(get_employees(), get_days(), get_shifts()).tolist()

    requests_met = 0
    for i in range(get_employees()):
        for j in range(get_days()):
            if requests[i][j] == shifts[i][j]:
                requests_met += 1

    return requests_met


def objective_function(x):
    x = x.astype(int)

    cost = get_days() * get_employees() - check_met_requests(x)

    if not check_shift_to_single_employee_per_day(x):
        cost += 10000

    if not check_employee_one_shift_per_day(x):
        cost += 10000

    if not check_employee_shifts_distribution(x):
        cost += 10000

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
    print(pd.DataFrame.from_records(solution.reshape(get_employees(), get_days() * get_shifts()).astype(int),
                                    columns=[str(i + 1) for i in range(get_shifts())] * get_days()))


def genetic_algorithm(generations, population, elit, studEA):
    model = ga.geneticalgorithm2(function=objective_function,
                                 dimension=get_employees() * get_shifts() * get_days(),
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
