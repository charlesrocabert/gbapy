#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import dill
import argparse

sys.path.append('./src/')

from Model import *
from Graphics import *


##################
#      MAIN      #
##################

if __name__ == "__main__":
    
    model = load_model("A")
    model.set_f0(model.LP_solution)
    model.mean_evolutionary_trajectory(condition = "1")