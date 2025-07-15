#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# GBApy (Growth Balance Analysis for Python)
# Copyright © 2024-2025 Charles Rocabert, Furkan Mert
# Web: https://github.com/charlesrocabert/gbapy
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#***********************************************************************

"""
Filename: generate_toy_model.py
Author: Charles Rocabert
Date: 2024-09-07
Description:
    Executable to generate.
License: GNU General Public License v3.0
Copyright: © 2024-2025 Charles Rocabert, Furkan Mert
"""

import os
import sys
import gba
import argparse
from typing import Optional


def parse_arguments() -> dict:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-path", help="Path to save the model", required=True)
    parser.add_argument("--name", "-name", help="Name of the model", required=True)
    parser.add_argument("--nb-efms", "-efms", help="Number of EFMs", required=True)
    parser.add_argument("--max-kcat", "-kcat", help="Maximum kcat value", default=10.0)
    parser.add_argument("--kcat-gradient", "-gradient", help="Gradient of the kcat values", default=0.01)
    parser.add_argument("--KM", "-KM", help="KM value", default=1.0)
    parser.add_argument("--x-conc", "-x", help="Concentration of external metabolites", default=0.1)
    args = parser.parse_args()
    return(vars(args))

def create_folder( path: str, model_name: str ) -> None:
    """
    Create a folder to store the model files.

    Parameters
    ----------
    path : str
        Path to save the model.
    model_name : str
        Name of the model.
    """
    if not os.path.exists(path+"/"+model_name):
        os.makedirs(path+"/"+model_name)
    else:
        os.system("rm -r "+path+"/"+model_name)
        os.makedirs(path+"/"+model_name)

def create_conditions_file( path: str, model_name: str, nb_efms: int, x_conc: float ) -> None:
    """
    Create the conditions file.

    Parameters
    ----------
    path : str
        Path to save the model.
    model_name : str
        Name of the model.
    nb_efms : int
        Number of EFMs.
    x_conc : float
        Concentration of external metabolites.
    """
    f = open(path+"/"+model_name+"/conditions.csv", "w")
    f.write(";1\n")
    f.write("rho;340\n")
    for i in range(1, nb_efms+1):
        f.write("x_C"+str(i)+";"+str(x_conc)+"\n")
    f.close()

def create_mass_fraction_matrix_file( path: str, model_name: str, nb_efms: int ) -> None:
    """
    Create the mass fraction matrix file.

    Parameters
    ----------
    path : str
        Path to save the model.
    model_name : str
        Name of the model.
    nb_efms : int
        Number of EFMs.
    """
    f = open(path+"/"+model_name+"/M.csv", "w")
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
    for j in range(nb_efms):
        f.write(";0")
    for j in range(nb_efms):
        f.write(";1")
    f.write(";-1\n")
    ### Write protein line ###
    f.write("Protein")
    for i in range(1, (nb_efms*2)+1):
        f.write(";0")
    f.write(";1\n")
    f.close()

def create_kcat_matrix( path: str, model_name: str, nb_efms: int, kcat_max: float, kcat_gradient: float ) -> None:
    """
    Create the kcat matrix file.

    Parameters
    ----------
    path : str
        Path to save the model.
    model_name : str
        Name of the model.
    nb_efms : int
        Number of EFMs.
    kcat_max : float
        Maximum kcat value.
    kcat_gradient : float
        Gradient of the kcat values.
    """
    f = open(path+"/"+model_name+"/kcat.csv", "w")
    ### Write header ###
    for i in range(1, (nb_efms*2)+1):
        f.write(";rxn"+str(i))
    f.write(";Ribosome\n")
    ### Write kcat forward line ###
    f.write("kcat_forward")
    for i in range(1, nb_efms+1):
        f.write(";"+str(kcat_max-kcat_gradient*(i-1)))
    for i in range(1, nb_efms+1):
        f.write(";"+str(kcat_max))
    f.write(";4.55\n")
    ### Write kcat backwward line ###
    f.write("kcat_backward")
    for i in range(1, (nb_efms*2)+1):
        f.write(";0")
    f.write(";0\n")
    f.close()

def create_KM_forward_matrix( path: str, model_name: str, nb_efms: int, KM: float ) -> None:
    """
    Create the KM forward matrix file.

    Parameters
    ----------
    path : str
        Path to save the model.
    model_name : str
        Name of the model.
    nb_efms : int
        Number of EFMs.
    KM : float
        KM value.
    """
    f = open(path+"/"+model_name+"/KM_forward.csv", "w")
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
        f.write("x_C"+str(i))
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

def create_directions( path: str, model_name: str, nb_efms: int ) -> None:
    """
    Create the directions file.
    
    Parameters
    ----------
    path : str
        Path to save the model.
    model_name : str
        Name of the model.
    nb_efms : int
        Number of EFMs.
    """
    f = open(path+"/"+model_name+"/directions.csv", "w")
    for i in range(1, (nb_efms*2)+1):
        f.write("rxn"+str(i)+";forward\n")
    f.write("Ribosome;forward\n")
    f.close()

def generate_full_column_rank_model( path: str, model_name: str, nb_efms: int, x_conc: float, kcat_max: float, kcat_gradient: float, KM: float ) -> None:
    """
    Generate a full column rank model.

    Parameters
    ----------
    path : str
        Path to save the model.
    model_name : str
        Name of the model.
    nb_efms : int
        Number of EFMs.
    x_conc : float
        Concentration of external metabolites.
    kcat_max : float
        Maximum kcat value.
    kcat_gradient : float
        Gradient of the kcat values.
    KM : float
        KM value.
    """
    create_folder(path, model_name)
    create_conditions_file(path, model_name, nb_efms, x_conc)
    create_mass_fraction_matrix_file(path, model_name, nb_efms)
    create_kcat_matrix(path, model_name, nb_efms, kcat_max, kcat_gradient)
    create_KM_forward_matrix(path, model_name, nb_efms, KM)
    create_directions(path, model_name, nb_efms)


##################
#      MAIN      #
##################

if __name__ == "__main__":
    print("#***********************************************************************")
    print("# GBApy (Growth Balance Analysis for Python)")
    print("# Copyright © 2024-2025 Charles Rocabert")
    print("# Web: https://github.com/charlesrocabert/gbapy")
    print("#")
    print("# generate_toy_model.py")
    print("# ---------------------")
    print("# Generate non full column rank toy models with a given number of EFMs.")
    print("# (LOCAL SCRIPT)")
    print("#***********************************************************************")

    config        = parse_arguments()
    path          = config["path"]
    name          = config["name"]
    nb_efms       = int(config["nb_efms"])
    max_kcat      = float(config["max_kcat"])
    kcat_gradient = float(config["kcat_gradient"])
    KM            = float(config["KM"])
    x_conc        = float(config["x_conc"])
    generate_full_column_rank_model(path, name, nb_efms, x_conc, max_kcat, kcat_gradient, KM)

