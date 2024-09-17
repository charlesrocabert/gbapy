#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# Copyright © 2024 Charles Rocabert, Furkan Mert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# csv_to_binary.py
# ----------------
# Load the GBA model from CSV files and save it in binary format.
# (LOCAL SCRIPT)
#***********************************************************************

import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_model import *

### Parse command line arguments ###
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-name", help="Model name", required=True)
    parser.add_argument("--save-f0", "-f0", action="store_true", help="Find and save the f0 vector")
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
    print("# csv_to_binary.py")
    print("# ----------------")
    print("# Load the GBA model from CSV files and save it in binary format.")
    print("# (LOCAL SCRIPT)")
    print("#***********************************************************************")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Parse command line arguments #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    config = parse_arguments()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Generate the binary model    #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    print("> Export CSV model "+config["model_name"]+" to binary format (./binary_models/"+config["model_name"]+".gba)")
    load_and_backup_model(config["model_name"], config["save_f0"], config["save_optimums"])

