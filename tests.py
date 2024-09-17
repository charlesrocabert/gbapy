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
    print(model.optimum_solutions)
    model.generate_random_initial_solutions("1", 10, 10000, 1e-6)
    print(model.random_solutions)