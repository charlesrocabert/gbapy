#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#***********************************************************************
# GBApy (Growth Balance Analysis for Python)
# Copyright © 2024-2025 Charles Rocabert
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
Filename: Enumerations.py
Author: Charles Rocabert
Date: 2024-16-12
Description:
    List of enumerations of the GBApy module.
License: GNU General Public License v3.0
Copyright: © 2024-2025 Charles Rocabert
"""

import os
import sys
import enum

class SpeciesType(enum.Enum):
    """
    Species type enumeration.
    - DNA          : DNA species (DNA sequence available).
    - RNA          : RNA species (RNA sequence available).
    - Protein      : Protein species (amino-acid sequence available).
    - SmallMolecule: Small molecule species (chemical formula available).
    - MacroMolecule: Macro-molecule species (chemical formula with radical).
    """
    DNA           = 1
    RNA           = 2
    Protein       = 3
    SmallMolecule = 4
    MacroMolecule = 5
    Unknown       = 6

class SpeciesLocation(enum.Enum):
    """
    Species location enumeration.
    - Internal: Species located inside the cell.
    - External: Species located outside the cell.
    - Unknown : Species location is unknown.
    """
    Internal = 1
    External = 2
    Unknown  = 3

class ReactionType(enum.Enum):
    """
    Reaction type enumeration.
    - Metabolic: Metabolic (internal) reaction.
    - Transport: Transport (external) reaction.
    - Exchange : Exchange reaction (specific to FBA models).
    """
    Metabolic = 1
    Transport = 2
    Exchange  = 3

class ReactionDirection(enum.Enum):
    """
    Reaction direction enumeration.
    - Forward   : Forward reaction.
    - Backward  : Backward reaction.
    - Reversible: Reversible reaction.
    """
    Forward    = 1
    Backward   = 2
    Reversible = 3

class ReactionGPR(enum.Enum):
    """
    Reaction GPR logic enumeration.
    - And:     Logical AND operator.
    - Or:      Logical OR operator.
    - Unknown: Unknown logical operator.
    """
    NONE = 1
    AND  = 2
    OR   = 3

class CgmReactionType(enum.Enum):
    """
    Reaction direction enumeration.
    - iMM  : Simple irreversible Michaelis-Menten reaction.
    - iMMa : Irreversible Michaelis-Menten reaction with activation.
    - iMMi : Irreversible Michaelis-Menten reaction with inhibition.
    - iMMia: Irreversible Michaelis-Menten reaction with activation and inhibition.
    - rMM  : Reversible Michaelis-Menten reaction.
    """
    iMM   = 1
    iMMa  = 2
    iMMi  = 3
    iMMia = 4
    rMM   = 5

class CgmConstants(float, enum.Enum):
    """
    Constant for CGM algorithms.
    - MIN_CONCENTRATION            : Minimum concentration value.
    - MIN_FLUX_FRACTION            : Minimum flux fraction value.
    - MAX_FLUX_FRACTION            : Maximum flux fraction value.
    - DENSITY_TOL                  : Density tolerance threshold (|1-rho| < ε).
    - NEGATIVE_C_TOL               : Negative C tolerance threshold (C > -ε).
    - NEGATIVE_P_TOL               : Negative P tolerance threshold (P > -ε).
    - TRAJECTORY_CONVERGENCE_COUNT : Number of iterations with equal mu values to consider the trajectory stable.
    - TRAJECTORY_CONVERGENCE_TOL   : Mu threshold below which growth rates are considered equal.
    - DECREASING_DT_FACTOR         : Factor by which the time step is decreased when the trajectory is unstable.
    - INCREASING_DT_FACTOR         : Factor by which the time step is increased when the trajectory is stable.
    - INCREASING_DT_COUNT          : Number of iterations with equal mu values to increase the time step.
    - MIN_DT                       : Minimum time step value.
    - PRINT_DATA_COUNT             : Frequency of data printing.
    - EXPORT_DATA_COUNT            : Frequency of data export.
    """
    MIN_CONCENTRATION            = 1e-10
    MIN_FLUX_FRACTION            = 1e-10
    DENSITY_TOL                  = 1e-10
    NEGATIVE_C_TOL               = 1e-10
    NEGATIVE_P_TOL               = 1e-10
    TRAJECTORY_CONVERGENCE_COUNT = 100
    TRAJECTORY_CONVERGENCE_TOL   = 1e-10
    DECREASING_DT_FACTOR         = 5.0
    INCREASING_DT_FACTOR         = 2.0
    INCREASING_DT_COUNT          = 100
    MIN_DT                       = 1e-100
    PRINT_DATA_COUNT             = 1
    EXPORT_DATA_COUNT            = 1

class MessageType(enum.Enum):
    """
    Message type.
    - Info    : Throw an information message.
    - Warning : Throw a warning message.
    - Error   : Throw an error message.
    - Plain   : Throw a plain message.
    """
    Info    = 1
    Warning = 2
    Error   = 3
    Plain   = 4

