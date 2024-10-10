#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# Copyright © 2024 Charles Rocabert, Furkan Mert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# csv_to_gba.py
# -------------
# Load the GBA model from CSV files and save it in binary format.
# (LOCAL SCRIPT)
#***********************************************************************

import os
import sys
import dill
import argparse

sys.path.append('./src/gbapy/')

from gbapy import *

### Parse command line arguments ###
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", "-folder", help="Model folder", required=True)
    parser.add_argument("--gba-path", "-gba", help="Path of the newly created GBA model", required=True)
    parser.add_argument("--save-LP-solution", "-LP", action="store_true", help="Find and save the LP initial solution")
    parser.add_argument("--save-optimums", "-optimums", action="store_true", help="Find and save optimums for all conditions")
    args = parser.parse_args()
    return(vars(args))


##################
#      MAIN      #
##################

if __name__ == "__main__":
    print("#***********************************************************************")
    print("# Copyright © 2024 Charles Rocabert, Furkan Mert")
    print("# Web: https://github.com/charlesrocabert/GBA_Evolution")
    print("#")
    print("# csv_to_gba.py")
    print("# -------------")
    print("# Load the GBA model from CSV files and save it in binary format .gba")
    print("# (LOCAL SCRIPT)")
    print("#***********************************************************************")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Parse command line arguments #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    config = parse_arguments()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Generate the binary model    #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    print("> Export CSV model "+config["model_folder"]+" to binary format (into "+config["gba_path"]+")")
    create_gba_model(config["model_folder"], config["gba_path"], config["save_LP_solution"], config["save_optimums"])

