#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#*******************************************************************************
# gbapy (Growth Balance Analysis for Python)
# Web: https://github.com/charlesrocabert/gbapy
# 
# MIT License
# 
# Copyright © 2024-2025 Charles Rocabert. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#*******************************************************************************

"""
Filename: Enumerations.py
Author: Charles Rocabert
Date: 2024-16-12
Description:
    List of enumerations of the gbapy module.
License: MIT License
Copyright: © 2024-2025 Charles Rocabert. All rights reserved.
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
    - Unknown      : Species type is unknown.
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
    - Transporter: Transport (boundary) reaction.
    - Spontaneous: Spontaneous (boundary) reaction.
    - Exchange : Exchange reaction (specific to FBA models).
    """
    Metabolic   = 1
    Transport   = 2
    Spontaneous = 3
    Exchange    = 4

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
    - iMMr : Irreversible Michaelis-Menten reaction with regulation.
    - rMM  : Reversible Michaelis-Menten reaction.
    """
    iMM   = 1
    iMMa  = 2
    iMMi  = 3
    iMMia = 4
    iMMr  = 5
    rMM   = 6

class CgmConstants(float, enum.Enum):
    """
    Constant for CGM algorithms.
    - TOL                          : Tolerance value.
    - TRAJECTORY_CONVERGENCE_COUNT : Number of iterations with equal mu values to consider the trajectory stable.
    - TRAJECTORY_CONVERGENCE_TOL   : Mu threshold below which growth rates are considered equal.
    - DECREASING_DT_FACTOR         : Factor by which the time step is decreased when the trajectory is unstable.
    - INCREASING_DT_FACTOR         : Factor by which the time step is increased when the trajectory is stable.
    - INCREASING_DT_COUNT          : Number of iterations with equal mu values to increase the time step.
    - MIN_DT                       : Minimum time step value.
    - PRINT_DATA_COUNT             : Frequency of data printing.
    - EXPORT_DATA_COUNT            : Frequency of data export.
    - REGULATION_SIGMA             : Width of the regulation Gaussian kernel.
    """
    TOL                          = 1e-10
    TRAJECTORY_CONVERGENCE_COUNT = 10000
    DECREASING_DT_FACTOR         = 5.0
    INCREASING_DT_FACTOR         = 2.0
    INCREASING_DT_COUNT          = 100
    MIN_DT                       = 1e-100
    PRINT_DATA_COUNT             = 1
    EXPORT_DATA_COUNT            = 100
    REGULATION_SIGMA             = 2.0

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

