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

### Dump the model in a binary file ###
def dump_model( gba_model, model_name ):
    filename = "./binary_models/"+model_name+".gba"
    ofile = open(filename, "wb")
    dill.dump(gba_model, ofile)
    ofile.close()
    assert os.path.isfile(filename), "ERROR: dump_model: model dump failed."

### Load a model and dump the binary backup ###
def load_and_backup_model( model_name ):
    model = GBA_model()
    model.load_model("./csv_models/", model_name)
    dump_model(model, model_name)


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

