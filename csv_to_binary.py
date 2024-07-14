#!/usr/bin/env python3
# coding: utf-8

#***************************************************************************
# Copyright © 2023-2024 Charles Rocabert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# load_model.py
# -------------
# Load the GBA model from CSV files and save it in binary format.
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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Parse command line arguments #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    config = parse_arguments()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Generate the binary model    #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    print("> Export CSV model "+config["model_name"]+" to binary format (./binary_models/"+config["model_name"]+".gba)")
    load_and_backup_model(config["model_name"])

