import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_tol import *
from GBA_model import *
from GBA_algorithms import *

algo = GBA_algorithms("C")
algo.load_optimums()

### Test random solutions generation ###
plt.ion()
algo.generate_random_initial_solutions("1", 10)
algo.gba_model.set_condition("1")
for key in algo.random_f.keys():
    print(algo.random_f[key])
    algo.gba_model.set_f0(algo.random_f[key])
    algo.compute_gradient_ascent(condition="1", max_time=200, initial_dt=0.01)
    plt.clf()
    algo.plot_trajectory()
    plt.draw()
    plt.pause(1)

# for key in algo.random_f.keys():
#     plt.plot(algo.random_f[key], label=key)
# plt.legend()
# plt.show()

### Test gradient ascent ###
# algo.load_LP_initial_solution()
# algo.compute_gradient_ascent(condition="25", max_time=200, initial_dt=0.01)
# algo.plot_trajectory()
# algo.save_trajectory("./output/test_trajectory.csv")

### Test trajectory with noise ###
# algo.load_LP_initial_solution()
# algo.compute_gradient_ascent_with_noise(condition="1", max_time=200, initial_dt=0.01, sigma=0.1)

### Test optimum over all conditions ###
# algo.load_LP_initial_solution()
# algo.compute_optimum_for_all_conditions(max_time=200, initial_dt=0.01)
# plt.figure()
# plt.subplot(2,1,1)
# algo.plot_mu_to_condition()
# plt.subplot(2,1,2)
# algo.plot_f_to_condition()
# plt.show()
