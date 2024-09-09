import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_tol import *
from GBA_model import *
from GBA_algorithms import *

### Convert to binary ###
load_and_backup_model("FCR_EFM2")
algo = GBA_algorithms("FCR_EFM2")

### Calculate first optimum ###
algo.generate_random_initial_solutions(condition="1", nb_solutions=10)
print(algo.random_f)


for i in algo.random_f.keys():
    algo.load_random_initial_solution(i+1)
    algo.compute_gradient_ascent(condition="1", max_time=200.0, initial_dt=0.01, save_f=True)
    print(algo.gba_model.mu)