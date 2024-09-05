import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_tol import *
from GBA_model import *
from GBA_algorithms import *

algo = GBA_algorithms("A")
#algo.compute_gradient_ascent(condition = "1", max_time=200, initial_dt = 0.01, dt_changeRate=0.2, nameOfCSV = None)

algo.compute_gradient_ascent_with_noise(condition = "1", max_time=200, initial_dt = 0.01, dt_changeRate=0.2, sigma=0.1, nameOfCSV = None)