import folium
import geneticalgorithm2 as ga
import json
import math
from matplotlib import pyplot as plt
import numpy as np


def get_cities():
    capitals_json = json.load(open('capitals.json'))

    capital_cities = []
    capital_coordinates = {}

    for state in capitals_json:
        if state not in ['AK', 'HI']:
            capital = capitals_json[state]['capital']

            capital_cities.append(capital)
            capital_coordinates[capital] = (float(capitals_json[state]['lat']), float(capitals_json[state]['long']))

    capital_cities = capital_cities[:10]

    return capital_cities, capital_coordinates


capitals, coordinates = get_cities()
cities = len(capitals)


def haversine(start_latitude, start_longitude, finish_latitude, finish_longitude):
    degree_to_radiant = float(math.pi / 180.0)

    delta_latitude = (finish_latitude - start_latitude) * degree_to_radiant
    delta_longitude = (finish_longitude - start_longitude) * degree_to_radiant

    factor = pow(math.sin(delta_latitude / 2), 2) + math.cos(start_latitude * degree_to_radiant) * math.cos(finish_latitude * degree_to_radiant) * pow(math.sin(delta_longitude / 2), 2)
    return 7912 * math.atan2(math.sqrt(factor), math.sqrt(1 - factor))


def get_distance(start, finish):
    start = coordinates[start]
    finish = coordinates[finish]

    return haversine(start[0], start[1], finish[0], finish[1])


def get_maximum_distance():
    maximum = 0

    for i in capitals:
        for j in capitals:
            if i != j and get_distance(i, j) > maximum:
                maximum = get_distance(i, j)

    return maximum


maximum_distance = get_maximum_distance()


def objective_function(x):
    x = x.astype(int)

    penalty = 0
    total_distance = 0

    for i in range(cities):
        total_distance += get_distance(capitals[x[i]], capitals[x[(i + 1) % cities]])

        if x.tolist().count(x[i]) > 1:
            penalty += maximum_distance * cities

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
        route += capitals[int(i)] + ' -> '
    route += capitals[int(solution[0])]

    print(route)
    print('Miles: {}'.format(score))


def plot_map(solution):
    travel_map = folium.Map(location=[40, -95], zoom_start=5)

    points = []
    for i, s in enumerate(solution, 1):
        points.append(coordinates[capitals[int(s)]])
        folium.Marker(location=coordinates[capitals[int(s)]],
                      popup=folium.Popup(html=str(i) + '-' + capitals[int(s)],
                                         show=True),
                      icon=folium.Icon(color='red' if i == 1 else 'blue',
                                       icon='info-sign')).add_to(travel_map)

    points.append(points[0])

    folium.PolyLine(points).add_to(travel_map)
    travel_map.save('map.html')


def genetic_algorithm(generations, population, elit, studEA):
    model = ga.geneticalgorithm2(function=objective_function,
                                 dimension=cities,
                                 variable_type='int',
                                 variable_boundaries=np.array([[0, cities - 1]] * cities),
                                 algorithm_parameters={'max_num_iteration': generations,
                                                       'population_size': population,
                                                       'elit_ratio': elit})

    model.run(no_plot=True,
              studEA=studEA)

    plot_algorithm_progress(np.array(model.report))
    plot_algorithm_generation(model.output_dict['last_generation']['scores'])
    print_algorithm_solution(model.output_dict['variable'], model.output_dict['function'])
    plot_map(model.output_dict['variable'])


if __name__ == '__main__':
    genetic_algorithm(generations=9000,
                      population=100,
                      elit=0.01,
                      studEA=True)
