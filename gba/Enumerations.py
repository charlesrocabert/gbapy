#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#***********************************************************************
# gbapy (growth balance analysis for Python)
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

import enum
from enum import auto

class GeneEssentiality(enum.Enum):
    """
    Gene essentiality enumeration.
    - ESSENTIAL      : Gene is essential.
    - QUASI_ESSENTIAL: Gene is quasi-essential.
    - NON_ESSENTIAL  : Gene is non-essential.
    - UNKNOWN        : Gene essentiality is unknown.
    """
    ESSENTIAL       = auto()
    QUASI_ESSENTIAL = auto()
    NON_ESSENTIAL   = auto()
    UNKNOWN         = auto()

class SpeciesType(enum.Enum):
    """
    Species type enumeration.
    - DNA          : DNA species (DNA sequence available).
    - RNA          : RNA species (RNA sequence available).
    - PROTEIN      : Protein species (amino-acid sequence available).
    - SMALLMOLECULE: Small molecule species (chemical formula available).
    - MACROMOLECULE: Macro-molecule species (chemical formula with radical).
    - UNKNOWN      : Species type is unknown.
    """
    DNA           = auto()
    RNA           = auto()
    PROTEIN       = auto()
    SMALLMOLECULE = auto()
    MACROMOLECULE = auto()
    UNKNOWN       = auto()

class SpeciesLocation(enum.Enum):
    """
    Species location enumeration.
    - INTERNAL: Species located inside the cell.
    - EXTERNAL: Species located outside the cell.
    - UNKNOWN : Species location is unknown.
    """
    INTERNAL = auto()
    EXTERNAL = auto()
    UNKNOWN  = auto()

class ReactionType(enum.Enum):
    """
    Reaction type enumeration.
    - METABOLIC:   Metabolic (internal) reaction.
    - TRANSPORT:   Transport (boundary) reaction.
    - SPONTANEOUS: Spontaneous (boundary) reaction.
    - EXCHANGE :   Exchange reaction (specific to FBA models).
    """
    METABOLIC   = auto()
    TRANSPORT   = auto()
    SPONTANEOUS = auto()
    EXCHANGE    = auto()

class ReactionDirection(enum.Enum):
    """
    Reaction direction enumeration.
    - FORWARD   : Forward reaction.
    - BACKWARD  : Backward reaction.
    - REVERSIBLE: Reversible reaction.
    """
    FORWARD    = auto()
    BACKWARD   = auto()
    REVERSIBLE = auto()

class ReactionGPR(enum.Enum):
    """
    Reaction GPR logic enumeration.
    - NONE: No logical operator.
    - AND:  Logical AND operator.
    - OR:   Logical OR operator.
    """
    NONE = auto()
    AND  = auto()
    OR   = auto()

class GbaReactionType(enum.Enum):
    """
    Reaction direction enumeration.
    - IMM  : Simple irreversible Michaelis-Menten reaction.
    - IMMA : Irreversible Michaelis-Menten reaction with activation.
    - IMMI : Irreversible Michaelis-Menten reaction with inhibition.
    - IMMIA: Irreversible Michaelis-Menten reaction with activation+inhibition.
    - RMM  : Reversible Michaelis-Menten reaction.
    """
    IMM   = auto()
    IMMA  = auto()
    IMMI  = auto()
    IMMIA = auto()
    RMM   = auto()

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
    INFO    = auto()
    WARNING = auto()
    ERROR   = auto()
    PLAIN   = auto()

