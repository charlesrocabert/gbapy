#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# Copyright © 2024 Charles Rocabert, Furkan Mert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# EFM2_random_solutions.py
# ------------------------
# Generate random initial solutions for the EFM2 model, where the value
# of rxn3 is blocked to a given value.
# (LOCAL SCRIPT)
#***********************************************************************

import os
import sys
import argparse
import numpy as np

sys.path.append('./src/')

from Model import *


##################
#      MAIN      #
##################

if __name__ == "__main__":
    print("#***********************************************************************")
    print("# Copyright © 2024 Charles Rocabert, Furkan Mert")
    print("# Web: https://github.com/charlesrocabert/GBA_Evolution")
    print("#")
    print("# EFM2_random_solutions.py")
    print("# ------------------------")
    print("# Generate random initial solutions for the EFM2 model, where the value")
    print("# of rxn3 is blocked to a given value.")
    print("# (LOCAL SCRIPT)")
    print("#***********************************************************************")

    model = load_model("FCR_EFM2")

    # reaction;f0
    # rxn1;0.025000000100000003
    # rxn2;0.9749999999
    # rxn3;1e-10
    # rxn4;0.9499999999
    # Ribosome;0.9249999999999999

    rxn3     = 0.1
    rxn3_pos = 2
    for i in range(10000):
        negative_term = True
        while negative_term:
            model.f_trunc             = np.random.rand(model.nj-1)
            model.f_trunc             = model.f_trunc*(MAX_FLUX_FRACTION-MIN_FLUX_FRACTION)+MIN_FLUX_FRACTION
            model.f_trunc[rxn3_pos-1] = rxn3
            model.set_f()
            if model.f[0] >= 0.0:
                negative_term = False
        model.calculate()
        model.check_model_consistency()
        if model.consistent:
            print(model.consistent)
        

