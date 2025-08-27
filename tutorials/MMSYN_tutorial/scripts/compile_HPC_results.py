#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#***********************************************************************
# GBAcpp (Growth Balance Analysis for C++)
# Copyright © 2024-2025 Charles Rocabert
# Web: https://github.com/charlesrocabert/GBAcpp
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
Filename: compile_HPC_results.py
Author: Charles Rocabert
Date: 2024-16-12
Description:
    Script used to compile downloaded HPC results.
License: GNU General Public License v3.0
Copyright: © 2024-2025 Charles Rocabert
"""


##################
#      MAIN      #
##################

if __name__ == "__main__":
    MODELS      = ["mmsyn_fcr_v2"]
    CONDITIONS  = range(1, 71)
    CONDITIONS  = [str(c) for c in CONDITIONS]
    LABELS      = ["b", "c", "f", "p", "state", "v"]

    outputs = {}
    for model in MODELS:
        for label in LABELS:
            filename                 = "../output/"+model+"_"+label+"_optimum.csv"
            outputs[model+"_"+label] = open(filename, "w")

    for model in MODELS:
        for label in LABELS:
            header_written = False
            for condition in CONDITIONS:
                filename = "../hpc_download/"+model+"_"+condition+"_"+label+"_optimum.csv"
                try:
                    f = open(filename, "r")
                    if not header_written:
                        outputs[model+"_"+label].write(f.readline())
                        header_written = True
                    else:
                        f.readline()
                    outputs[model+"_"+label].write(f.readline())
                    f.close()
                except:
                    print("> Warning: file "+filename+" not found")

    for item in outputs.items():
        item[1].close()