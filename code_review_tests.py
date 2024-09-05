import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_tol import *
from GBA_model import *
from GBA_algorithms import *

algo = GBA_algorithms("A")
algo.load_optimums()

algo.initialize_f0(condition="1")
algo.compute_gradient_ascent(condition="20", max_time=200, initial_dt=0.01)

#algo.compute_gradient_ascent_with_noise(condition="1", max_time=200, initial_dt=0.01, sigma=0.1)
#algo.compute_optimum_for_all_conditions(max_time=200, initial_dt=0.01)

