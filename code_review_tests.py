import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_tol import *
from GBA_model import *
from GBA_algorithms import *

### Convert to binary ###
load_and_backup_model("NFCR_EFM2")
algo = GBA_algorithms("NFCR_EFM2")

### Calculate first optimum ###
algo.generate_random_initial_solutions(condition="1", nb_solutions=10)
print(algo.random_f)
plt.ion()
#ax = plt.figure().add_subplot(projection='3d')
for i in range(10):
    algo.load_random_initial_solution(i+1)
    algo.compute_gradient_ascent(condition="1", max_time=200.0, initial_dt=0.01, save_f=True)
    #algo.save_gradient_ascent_trajectory(save_f=True, filename="./output/trajectory"+str(i+1)+".csv")
    plt.clf()
    algo.plot_EFM_trajectory()
    #plt.show()
    plt.draw()
    plt.pause(1)
plt.show()