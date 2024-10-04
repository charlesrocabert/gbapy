#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# Copyright © 2024 Charles Rocabert, Furkan Mert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# generate_toy_model.py
# ---------------------
# Generate full column rank (FCR) and non-full column rank (NFCR)
# models with a given number of EFMs.
# (LOCAL SCRIPT)
#***********************************************************************

import os
import sys
import argparse

sys.path.append('./src/')

from Model import *

### Parse command line arguments ###
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb-efms", "-efms", help="Number of EFMs", required=True)
    args = parser.parse_args()
    return(vars(args))

### Create folder ###
def create_folder( model_name ):
    if not os.path.exists("./csv_models/"+model_name):
        os.makedirs("./csv_models/"+model_name)
    else:
        os.system("rm -r ./csv_models/"+model_name)
        os.makedirs("./csv_models/"+model_name)
    
### Create full column rank (FCR) conditions file ###
def create_FCR_conditions_file( model_name, nb_efms, x_conc ):
    f = open("./csv_models/"+model_name+"/conditions.csv", "w")
    f.write(";1\n")
    f.write("rho;340\n")
    for i in range(1, nb_efms+1):
        f.write("x_C"+str(i)+";"+str(x_conc)+"\n")
    f.close()

### Create non-full column rank (NFCR) conditions file ###
def create_NFCR_conditions_file( model_name, x_conc ):
    f = open("./csv_models/"+model_name+"/conditions.csv", "w")
    f.write(";1\n")
    f.write("rho;340\n")
    f.write("x_C;"+str(x_conc)+"\n")
    f.close()

### (TESTED) Create full column rank (FCR) mass fraction matrix file ###
def create_FCR_mass_fraction_matrix_file( model_name, nb_efms ):
    f = open("./csv_models/"+model_name+"/M.csv", "w")
    ### Write header ###
    for i in range(1, (nb_efms*2)+1):
        f.write(";rxn"+str(i))
    f.write(";Ribosome\n")
    ### Write external metabolite lines ###
    for i in range(1, nb_efms+1):
        f.write("x_C"+str(i))
        for j in range(1, nb_efms+1):
            if i == j:
                f.write(";-1")
            else:
                f.write(";0")
        for j in range(1, nb_efms+1):
            f.write(";0")
        f.write(";0\n")
    ### Write internal metabolite lines ###
    for i in range(1, nb_efms+1):
        f.write("C"+str(i))
        for j in range(1, nb_efms+1):
            if i == j:
                f.write(";1")
            else:
                f.write(";0")
        for j in range(1, nb_efms+1):
            if i == j:
                f.write(";-1")
            else:
                f.write(";0")
        f.write(";0\n")
    ### Write amino-acid line ###
    f.write("AA")
    for i in range(1, nb_efms+1):
        f.write(";0")
    for i in range(1, nb_efms+1):
        f.write(";1")
    f.write(";-1\n")
    ### Write protein line ###
    f.write("Protein")
    for i in range(1, (nb_efms*2)+1):
        f.write(";0")
    f.write(";1\n")
    f.close()

### (TESTED) Create non-full column rank (NFCR) mass fraction matrix file ###
def create_NFCR_mass_fraction_matrix_file( model_name, nb_efms ):
    f = open("./csv_models/"+model_name+"/M.csv", "w")
    ### Write header ###
    for i in range(1, (nb_efms*2)+1):
        f.write(";rxn"+str(i))
    f.write(";Ribosome\n")
    ### Write first line ###
    f.write("x_C")
    for i in range(1, nb_efms+1):
        f.write(";-1")
    for i in range(1, nb_efms+1):
        f.write(";0")
    f.write(";0\n")
    ### Write intermediate metabolite lines ###
    for i in range(1, nb_efms+1):
        f.write("C"+str(i))
        for j in range(1, nb_efms+1):
            if i == j:
                f.write(";1")
            else:
                f.write(";0")
        for j in range(1, nb_efms+1):
            if i == j:
                f.write(";-1")
            else:
                f.write(";0")
        f.write(";0\n")
    ### Write amino-acid line ###
    f.write("AA")
    for i in range(1, nb_efms+1):
        f.write(";0")
    for i in range(1, nb_efms+1):
        f.write(";1")
    f.write(";-1\n")
    ### Write protein line ###
    f.write("Protein")
    for i in range(1, (nb_efms*2)+1):
        f.write(";0")
    f.write(";1\n")
    f.close()

### Create full column rank (FCR) kcat matrix ###
def create_FCR_kcat_matrix( model_name, nb_efms, kcat_max, kcat_gradient ):
    f = open("./csv_models/"+model_name+"/kcat.csv", "w")
    ### Write header ###
    for i in range(1, (nb_efms*2)+1):
        f.write(";rxn"+str(i))
    f.write(";Ribosome\n")
    ### Write kcat forward line ###
    f.write("kcat_forward")
    for i in range(1, nb_efms+1):
        f.write(";"+str(kcat_max))
    for i in range(1, nb_efms+1):
        f.write(";"+str(kcat_max-kcat_gradient*(i-1)))
    f.write(";4.55\n")
    ### Write kcat backwward line ###
    f.write("kcat_backward")
    for i in range(1, (nb_efms*2)+1):
        f.write(";0")
    f.write(";0\n")
    f.close()

### Create non-full column rank (NFCR) kcat matrix ###
def create_NFCR_kcat_matrix( model_name, nb_efms, kcat_max, kcat_gradient ):
    f = open("./csv_models/"+model_name+"/kcat.csv", "w")
    ### Write header ###
    for i in range(1, (nb_efms*2)+1):
        f.write(";rxn"+str(i))
    f.write(";Ribosome\n")
    ### Write kcat forward line ###
    f.write("kcat_forward")
    for i in range(1, nb_efms+1):
        f.write(";"+str(kcat_max))
    for i in range(1, nb_efms+1):
        f.write(";"+str(kcat_max-kcat_gradient*(i-1)))
    f.write(";4.55\n")
    ### Write kcat backwward line ###
    f.write("kcat_backward")
    for i in range(1, (nb_efms*2)+1):
        f.write(";0")
    f.write(";0\n")
    f.close()

### Create full column rank (FCR) KM forward matrix ###
def create_FCR_KM_forward_matrix( model_name, nb_efms, KM ):
    f = open("./csv_models/"+model_name+"/KM_forward.csv", "w")
    ### Write header ###
    for i in range(1, (nb_efms*2)+1):
        f.write(";rxn"+str(i))
    f.write(";Ribosome\n")
    ### Write external metabolite lines ###
    for i in range(1, nb_efms+1):
        f.write("x_C"+str(i))
        for j in range(1, nb_efms+1):
            if i == j:
                f.write(";"+str(KM))
            else:
                f.write(";0")
        for j in range(1, nb_efms+1):
            f.write(";0")
        f.write(";0\n")
    ### Write internal metabolite lines ###
    for i in range(1, nb_efms+1):
        f.write("C"+str(i))
        for j in range(1, nb_efms+1):
            f.write(";0")
        for j in range(1, nb_efms+1):
            if i == j:
                f.write(";"+str(KM))
            else:
                f.write(";0")
        f.write(";0\n")
    ### Write amino-acid line ###
    f.write("AA")
    for i in range(1, (nb_efms*2)+1):
        f.write(";0.0")
    f.write(";8.3\n")
    ### Write protein line ###
    f.write("Protein")
    for i in range(1, (nb_efms*2)+1):
        f.write(";0.0")
    f.write(";0.0\n")
    f.close()

### Create non-full column rank (NFCR) KM forward matrix ###
def create_NFCR_KM_forward_matrix( model_name, nb_efms, KM ):
    f = open("./csv_models/"+model_name+"/KM_forward.csv", "w")
    ### Write header ###
    for i in range(1, (nb_efms*2)+1):
        f.write(";rxn"+str(i))
    f.write(";Ribosome\n")
    ### Write external metabolite line ###
    f.write("x_C")
    for j in range(1, nb_efms+1):
        f.write(";"+str(KM))
    for j in range(1, nb_efms+1):
        f.write(";0")
    f.write(";0\n")
    ### Write internal metabolite lines ###
    for i in range(1, nb_efms+1):
        f.write("C"+str(i))
        for j in range(1, nb_efms+1):
            f.write(";0")
        for j in range(1, nb_efms+1):
            if i == j:
                f.write(";"+str(KM))
            else:
                f.write(";0")
        f.write(";0\n")
    ### Write amino-acid line ###
    f.write("AA")
    for i in range(1, (nb_efms*2)+1):
        f.write(";0.0")
    f.write(";8.3\n")
    ### Write protein line ###
    f.write("Protein")
    for i in range(1, (nb_efms*2)+1):
        f.write(";0.0")
    f.write(";0.0\n")
    f.close()

### Generate a full column rank model ###
def generate_full_column_rank_model( model_name, nb_efms, x_conc, kcat_max, kcat_gradient, KM ):
    create_folder(model_name)
    create_FCR_conditions_file(model_name, nb_efms, x_conc)
    create_FCR_mass_fraction_matrix_file(model_name, nb_efms)
    create_FCR_kcat_matrix(model_name, nb_efms, kcat_max, kcat_gradient)
    create_FCR_KM_forward_matrix(model_name, nb_efms, KM)
    load_and_backup_model(model_name)
    model = load_model(model_name)
    print("> Built full column rank model: "+model_name)
    print(model)

### Generate a non-full column rank model ###
def generate_non_full_column_rank_model( model_name, nb_efms, x_conc, kcat_max, kcat_gradient, KM ):
    create_folder(model_name)
    create_NFCR_conditions_file(model_name, x_conc)
    create_NFCR_mass_fraction_matrix_file(model_name, nb_efms)
    create_NFCR_kcat_matrix(model_name, nb_efms, kcat_max, kcat_gradient)
    create_NFCR_KM_forward_matrix(model_name, nb_efms, KM)
    load_and_backup_model(model_name)
    model = load_model(model_name)
    print("> Built non-full column rank model: "+model_name)
    print(model)


##################
#      MAIN      #
##################

if __name__ == "__main__":
    print("#***********************************************************************")
    print("# Copyright © 2024 Charles Rocabert, Furkan Mert")
    print("# Web: https://github.com/charlesrocabert/GBA_Evolution")
    print("#")
    print("# generate_toy_model.py")
    print("# ---------------------")
    print("# Generate full column rank (FCR) and non-full column rank (NFCR)")
    print("# models with a given number of EFMs.")
    print("# (LOCAL SCRIPT)")
    print("#***********************************************************************")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Parse command line arguments   #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    config = parse_arguments()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Create the models from scratch #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    nb_efms       = int(config["nb_efms"])
    x_conc        = 0.1
    kcat_max      = 10.0
    kcat_gradient = 0.01
    KM            = 1.0
    FCR_name      = "FCR_EFM"+str(nb_efms)
    NFCR_name     = "NFCR_EFM"+str(nb_efms)
    generate_full_column_rank_model(FCR_name, nb_efms, x_conc, kcat_max, kcat_gradient, KM)
    generate_non_full_column_rank_model(NFCR_name, nb_efms, x_conc, kcat_max, kcat_gradient, KM)

