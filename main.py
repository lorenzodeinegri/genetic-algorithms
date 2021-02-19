import numpy as np
import geneticalgorithm2 as ga


def f(x):
    return np.sum(x)


var_bound = np.array([[0, 10]] * 3)
model = ga.geneticalgorithm2(function=f, dimension=3, variable_type='real', variable_boundaries=var_bound)
model.run()
