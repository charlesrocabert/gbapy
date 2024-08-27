#!/usr/bin/env python3
# coding: utf-8

#***************************************************************************
# Copyright © 2023-2024 Charles Rocabert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# save_f0.py
# ----------
# Load the model and save the f0 vector.
# (LOCAL SCRIPT)
#***************************************************************************

import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_model import *

### Parse command line arguments ###
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-model-name", help="Model name", required=True)
    args = parser.parse_args()
    return(vars(args))


##################
#      MAIN      #
##################

if __name__ == "__main__":
    print("#***************************************************************************")
    print("# Copyright © 2023-2024 Charles Rocabert")
    print("# Web: https://github.com/charlesrocabert/GBA_PredictiveEvolution")
    print("#")
    print("# load_model.py")
    print("# -------------")
    print("# Load the GBA model from CSV files and save it in binary format.")
    print("# (LOCAL SCRIPT)")
    print("#***************************************************************************")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Parse command line arguments             #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    config = parse_arguments()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Load the model, calculate f0 and save it #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    model = load_model(config["model_name"])
    model.solve_local_linear_problem()
    model.write_f0()

