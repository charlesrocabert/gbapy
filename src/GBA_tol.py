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
MIN_FLUX_FRACTION          = 1e-10 # Minimum flux fraction value
MAX_FLUX_FRACTION          = 2.0   # Maximum flux fraction value
DENSITY_TOL                = 1e-10 # Density tolerance threshold (|1-rho| < ε)
NEGATIVE_C_TOL             = 1e-10 # Negative C tolerance threshold (C > -ε)
NEGATIVE_P_TOL             = 1e-10 # Negative P tolerance threshold (P > -ε)
TRAJECTORY_STABLE_MU_COUNT = 1000  # Number of iterations with equal mu values to consider the trajectory stable
TRAJECTORY_CONVERGENCE_TOL = 1e-10 # Mu threshold below which growth rates are considered equal
DECREASING_DT_FACTOR       = 5.0   # Factor by which the time step is decreased when the trajectory is unstable
INCREASING_DT_FACTOR       = 2.0   # Factor by which the time step is increased when the trajectory is stable
MCMC_CONVERGENCE_TOL       = 1e-5  # MCMC trajectory convergence tolerance
POPLEVEL_CONVERGENCE_TOL   = 1e-5  # Population-level trajectory convergence tolerance
EFM_TOL                    = 1e-5  # Tolerance threshold below which EFM values are considered to be zero

