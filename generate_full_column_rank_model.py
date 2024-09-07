#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# Copyright © 2023-2024 Charles Rocabert, Furkan Mert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# save_f0.py
# ----------
# Load the model and save the f0 vector.
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
    parser.add_argument("--nb-efms", "-nb-efms", help="Number of EFMs", required=True)
    args = parser.parse_args()
    return(vars(args))


##################
#      MAIN      #
##################

if __name__ == "__main__":
    print("#***********************************************************************")
    print("# Copyright © 2023-2024 Charles Rocabert, Furkan Mert")
    print("# Web: https://github.com/charlesrocabert/GBA_PredictiveEvolution")
    print("#")
    print("# generate_full_column_rank_model.py")
    print("# ----------------------------------")
    print("# Generate a full column rank model.")
    print("# (LOCAL SCRIPT)")
    print("#***********************************************************************")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Parse command line arguments             #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    config = parse_arguments()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Create the model from scratch            #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    KM            = 1.0
    max_kcat      = 10.0
    gradient_kcat = 1.0
    x_conc        = 0.1
    nb_efms       = int(config["nb_efms"])
    model_name    = "EFM"+config["nb_efms"]

    ### Create folder ###
    if not os.path.exists("./csv_models/"+model_name):
        os.makedirs("./csv_models/"+model_name)
    else:
        os.system("rm -r ./csv_models/"+model_name)
        os.makedirs("./csv_models/"+model_name)
    
    ### Create conditions file ###
    f = open("./csv_models/"+model_name+"/conditions.csv", "w")
    f.write(";1\n")
    f.write("rho;340\n")
    for i in range(1, nb_efms+1):
        f.write("x_C"+str(i)+";"+str(x_conc)+"\n")
    f.close()
    
    ### Create mass fraction matrix file ###
    f = open("./csv_models/"+model_name+"/M.csv", "w")
    for i in range(1, nb_efms+1):
        f.write(";rxn"+str(i))
    f.write(";Ribosome\n")
    for i in range(1, nb_efms+1):
        f.write("x_C"+str(i))
        for j in range(1, nb_efms+1):
            if i == j:
                f.write(";1")
            else:
                f.write(";0")
        f.write(";0\n")
    f.write("AA")
    for i in range(1, nb_efms+1):
        f.write(";1")
    f.write(";-1\n")
    f.write("Protein")
    for i in range(1, nb_efms+1):
        f.write(";0")
    f.write(";1\n")
    f.close()

    ### Create the kcat matrix ###
    f = open("./csv_models/"+model_name+"/kcat.csv", "w")
    for i in range(1, nb_efms+1):
        f.write(";rxn"+str(i))
    f.write(";Ribosome\n")
    f.write("kcat_forward")
    for i in range(1, nb_efms+1):
        f.write(";"+str(max_kcat-gradient_kcat*(i-1)))
    f.write(";4.55\n")
    f.write("kcat_backward")
    for i in range(1, nb_efms+1):
        f.write(";0")
    f.write(";0\n")
    f.close()

    ### Create the KM forward matrix ###
    f = open("./csv_models/"+model_name+"/KM_forward.csv", "w")
    for i in range(1, nb_efms+1):
        f.write(";rxn"+str(i))
    f.write(";Ribosome\n")
    for i in range(1, nb_efms+1):
        f.write("x_C"+str(i))
        for j in range(1, nb_efms+1):
            if i == j:
                f.write(";"+str(KM))
            else:
                f.write(";0")
        f.write(";0\n")
    f.write("AA")
    for i in range(1, nb_efms+1):
        f.write(";0.0")
    f.write(";8.3\n")
    f.write("Protein")
    for i in range(1, nb_efms+1):
        f.write(";0.0")
    f.write(";0.0\n")
    f.close()

