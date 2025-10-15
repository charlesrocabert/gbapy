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

class GeneEssentiality(enum.Enum):
    """
    Gene essentiality enumeration.
    - ESSENTIAL      : Gene is essential.
    - QUASI_ESSENTIAL: Gene is quasi-essential.
    - NON_ESSENTIAL  : Gene is non-essential.
    - UNKNOWN        : Gene essentiality is unknown.
    """
    ESSENTIAL       = enum.auto()
    QUASI_ESSENTIAL = enum.auto()
    NON_ESSENTIAL   = enum.auto()
    UNKNOWN         = enum.auto()

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
    DNA           = enum.auto()
    RNA           = enum.auto()
    PROTEIN       = enum.auto()
    SMALLMOLECULE = enum.auto()
    MACROMOLECULE = enum.auto()
    UNKNOWN       = enum.auto()

class SpeciesLocation(enum.Enum):
    """
    Species location enumeration.
    - INTERNAL: Species located inside the cell.
    - EXTERNAL: Species located outside the cell.
    - UNKNOWN : Species location is unknown.
    """
    INTERNAL = enum.auto()
    EXTERNAL = enum.auto()
    UNKNOWN  = enum.auto()

class ReactionType(enum.Enum):
    """
    Reaction type enumeration.
    - METABOLIC:   Metabolic (internal) reaction.
    - TRANSPORT:   Transport (boundary) reaction.
    - SPONTANEOUS: Spontaneous (boundary) reaction.
    - EXCHANGE :   Exchange reaction (specific to FBA models).
    """
    METABOLIC   = enum.auto()
    TRANSPORT   = enum.auto()
    SPONTANEOUS = enum.auto()
    EXCHANGE    = enum.auto()

class ReactionDirection(enum.Enum):
    """
    Reaction direction enumeration.
    - FORWARD   : Forward reaction.
    - BACKWARD  : Backward reaction.
    - REVERSIBLE: Reversible reaction.
    """
    FORWARD    = enum.auto()
    BACKWARD   = enum.auto()
    REVERSIBLE = enum.auto()

class ReactionGPR(enum.Enum):
    """
    Reaction GPR logic enumeration.
    - NONE: No logical operator.
    - AND:  Logical AND operator.
    - OR:   Logical OR operator.
    """
    NONE = enum.auto()
    AND  = enum.auto()
    OR   = enum.auto()

class GbaReactionType(enum.Enum):
    """
    Reaction direction enumeration.
    - IMM  : Simple irreversible Michaelis-Menten reaction.
    - IMMA : Irreversible Michaelis-Menten reaction with activation.
    - IMMI : Irreversible Michaelis-Menten reaction with inhibition.
    - IMMIA: Irreversible Michaelis-Menten reaction with activation+inhibition.
    - RMM  : Reversible Michaelis-Menten reaction.
    """
    IMM   = enum.auto()
    IMMA  = enum.auto()
    IMMI  = enum.auto()
    IMMIA = enum.auto()
    RMM   = enum.auto()

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
    INFO    = enum.auto()
    WARNING = enum.auto()
    ERROR   = enum.auto()
    PLAIN   = enum.auto()

