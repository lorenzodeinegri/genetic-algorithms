import numpy as np
import geneticalgorithm as ga


def f(x):
    return np.sum(x)


var_bound = np.array([[0, 10]] * 3)
model = ga.geneticalgorithm(function=f, dimension=3, variable_type='real', variable_boundaries=var_bound)
model.run()
