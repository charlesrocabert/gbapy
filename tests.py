#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_model import *

##################
#      MAIN      #
##################

if __name__ == "__main__":
    
    model = load_model("A")
    model.generate_random_initial_solutions("1", 1, 10000, 1e-6)
    # for i in range(1, 11):
    #     model.set_f0(model.random_solutions[i])
    #     model.gradient_ascent(condition="1", max_time=5.0, initial_dt=0.01, index=i, track=True, add=True)
    # model.save_trajectory("test")
    # model.plot_trajectory()

    model.set_f0(model.random_solutions[1])
    model.compute_gradient_ascent_with_noise(condition = "10", max_time = 5, initial_dt = 0.01, sigma = 1.0, index=1, track=True, add=False)
    model.plot_trajectory()