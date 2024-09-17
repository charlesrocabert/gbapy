#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import dill
import argparse

sys.path.append('./src/')

from GBA_model import *


##################
#      MAIN      #
##################

if __name__ == "__main__":
    load_and_backup_model("A", save_f0=True, save_optimums=True)

