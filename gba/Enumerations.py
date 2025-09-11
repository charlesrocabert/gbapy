#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#***********************************************************************
# gbapy (Growth Balance Analysis for Python)
# Web: https://github.com/charlesrocabert/gbapy
# Copyright © 2024-2025 Charles Rocabert.
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#***********************************************************************

"""
Filename: Enumerations.py
Author: Charles Rocabert
Date: 2024-16-12
Description:
    List of enumerations of the gbapy module.
License: GNU General Public License v3 (GPLv3)
Copyright: © 2024-2025 Charles Rocabert.
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
    PROTEIN       = 3
    SMALLMOLECULE = 4
    MACROMOLECULE = 5
    UNKNOWN       = 6

class SpeciesLocation(enum.Enum):
    """
    Species location enumeration.
    - Internal: Species located inside the cell.
    - External: Species located outside the cell.
    - Unknown : Species location is unknown.
    """
    INTERNAL = 1
    EXTERNAL = 2
    UNKNOWN  = 3

class ReactionType(enum.Enum):
    """
    Reaction type enumeration.
    - Metabolic: Metabolic (internal) reaction.
    - Transporter: Transport (boundary) reaction.
    - Spontaneous: Spontaneous (boundary) reaction.
    - Exchange : Exchange reaction (specific to FBA models).
    """
    METABOLIC   = 1
    TRANSPORT   = 2
    SPONTANEOUS = 3
    EXCHANGE    = 4

class ReactionDirection(enum.Enum):
    """
    Reaction direction enumeration.
    - Forward   : Forward reaction.
    - Backward  : Backward reaction.
    - Reversible: Reversible reaction.
    """
    FORWARD    = 1
    BACKWARD   = 2
    REVERSIBLE = 3

class ReactionGPR(enum.Enum):
    """
    Reaction GPR logic enumeration.
    - NONE:    No logical operator.
    - AND:     Logical AND operator.
    - OR:      Logical OR operator.
    """
    NONE = 1
    AND  = 2
    OR   = 3

class GbaReactionType(enum.Enum):
    """
    Reaction direction enumeration.
    - iMM  : Simple irreversible Michaelis-Menten reaction.
    - iMMa : Irreversible Michaelis-Menten reaction with activation.
    - iMMi : Irreversible Michaelis-Menten reaction with inhibition.
    - iMMia: Irreversible Michaelis-Menten reaction with activation+inhibition.
    - iMMr : Irreversible Michaelis-Menten reaction with regulation.
    - rMM  : Reversible Michaelis-Menten reaction.
    """
    IMM   = 1
    IMMA  = 2
    IMMI  = 3
    IMMIA = 4
    RMM   = 6

class GbaConstants(float, enum.Enum):
    """
    Constant for GBA algorithms.
    - TOL: Tolerance value.
    """
    TOL = 1e-10

class MessageType(enum.Enum):
    """
    Message type.
    - INFO    : Throw an information message.
    - WARNING : Throw a warning message.
    - ERROR   : Throw an error message.
    - PLAIN   : Throw a plain message.
    """
    INFO    = 1
    WARNING = 2
    ERROR   = 3
    PLAIN   = 4

