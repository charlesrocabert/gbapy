#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# Copyright © 2023-2024 Charles Rocabert, Furkan Mert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# GBA_tol.py
# ----------
# Tolerance values associated to the various GBA algorithms.
# (LOCAL SCRIPT)
#***********************************************************************


MIN_CONCENTRATION          = 1e-10 # Minimum concentration value
MIN_FLUX_FRACTION          = 1e-10 # Flux fraction to correct slightly negative values
FLUX_BOUNDARY              = 2.0   # Maximal flux fraction value
DENSITY_CONSTRAINT_TOL     = 1e-10 # Density tolerance threshold (|1-rho| < ε)
NEGATIVE_C_TOL             = 1e-10 # Negative C tolerance threshold (C > -ε)
NEGATIVE_P_TOL             = 1e-10 # Negative P tolerance threshold (P > -ε)
TRAJECTORY_STABLE_MU_COUNT = 1000  # Number of stable mu values required to consider a trajectory stable
TRAJECTORY_CONVERGENCE_TOL = 1e-10 # Analytical trajectory convergence tolerance
DECREASING_DT_FACTOR       = 5.0   # Factor by which the time step is decreased when the trajectory is unstable
INCREASING_DT_FACTOR       = 2.0   # Factor by which the time step is increased when the trajectory is stable
MCMC_CONVERGENCE_TOL       = 1e-5  # MCMC trajectory convergence tolerance
POPLEVEL_CONVERGENCE_TOL   = 1e-5  # Population-level trajectory convergence tolerance
EFM_TOL                    = 1e-5  # Tolerance threshold below which EFM values are considered to be zero

POSSIBLY_STABLE_MU_ATTEMPTS = TRAJECTORY_STABLE_MU_COUNT * (5/6) # Attempts after which mu is possibly stable

