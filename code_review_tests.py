import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_tol import *
from GBA_model import *
from GBA_algorithms import *

algo = GBA_algorithms("A")
#algo.compute_gradient_ascent(condition="1", max_time=200, initial_dt=0.01)
#algo.compute_gradient_ascent_with_noise(condition="1", max_time=200, initial_dt=0.01, sigma=0.1)

algo.compute_optimum_for_all_conditions(max_time=200, initial_dt=0.01)

# plt.figure()
# plt.subplot(2,1,1)
# algo.plot_mu_to_condition()
# plt.subplot(2,1,2)
# algo.plot_f_to_condition()
# plt.show()