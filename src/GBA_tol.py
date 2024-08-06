#!/usr/bin/env python3
# coding: utf-8

#***************************************************************************
# Copyright © 2023-2024 Charles Rocabert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# GBA_tol.py
# ----------
# Tolerance values associated to the various GBA algorithms.
# (LOCAL SCRIPT)
#***************************************************************************

MIN_CONCENTRATION          = 1e-10 # Minimum concentration value
DENSITY_CONSTRAINT_TOL     = 1e-5  # Density tolerance threshold (|1-rho| < ε)
NEGATIVE_C_TOL             = 1e-5  # Negative C tolerance threshold (C > -ε)
NEGATIVE_P_TOL             = 1e-5  # Negative P tolerance threshold (P > -ε)
TRAJECTORY_STABLE_MU_COUNT = 1000  # Number of stable mu values required to consider a trajectory stable
TRAJECTORY_CONVERGENCE_TOL = 1e-5  # Analytical trajectory convergence tolerance
MCMC_CONVERGENCE_TOL       = 1e-5  # MCMC trajectory convergence tolerance
POPLEVEL_CONVERGENCE_TOL   = 1e-5  # Population-level trajectory convergence tolerance
EFM_TOL                    = 1e-5  # Tolerance threshold below which EFM values are considered to be zero
MIN_FLUXFRACTION           = 1e-10

##################
#      MAIN      #
##################

if __name__ == "__main__":
    print("#***************************************************************************")
    print("# Copyright © 2023-2024 Charles Rocabert")
    print("# Web: https://github.com/charlesrocabert/GBA_Evolution")
    print("#")
    print("# GBA_tol.py")
    print("# ----------")
    print("# Tolerance values associated to the various GBA algorithms.")
    print("# (LOCAL SCRIPT)")
    print("#***************************************************************************")

