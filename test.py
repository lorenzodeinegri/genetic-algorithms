import numpy as np
import geneticalgorithm2 as ga


def f(x):
    return np.sum(x)


model = ga.geneticalgorithm2(function=f,
                             dimension=3,
                             variable_type='real',
                             variable_boundaries=np.array([[0, 10]] * 3))
model.run()
