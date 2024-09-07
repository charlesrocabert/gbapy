import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_tol import *
from GBA_model import *
from GBA_algorithms import *

### Convert to binary ###
algo = GBA_algorithms("EFM7")

### Calculate first optimum ###
algo.generate_random_initial_solutions(condition="1", nb_solutions=10)
print(algo.random_f)
plt.ion()
for i in range(10):
    algo.load_random_initial_solution(i+1)
    algo.compute_gradient_ascent()
    plt.clf()
    algo.plot_gradient_ascent_trajectory()
    plt.draw()
    plt.pause(1)