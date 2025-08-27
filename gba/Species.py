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
Filename: Species.py
Author: Charles Rocabert
Date: 2024-16-12
Description:
    Species class of the gbapy module.
License: MIT License
Copyright: © 2024-2025 Charles Rocabert. All rights reserved.
"""

import os
import sys
import molmass
import pandas as pd
from Bio import SeqIO
from typing import Optional
import Bio.SeqUtils as SeqUtils
from IPython.display import display_html
from Bio.SeqUtils.ProtParam import ProteinAnalysis

try:
    from .Enumerations import *
except:
    from Enumerations import *


class Species:
    """
    Class describing a molecular species in the model.

    Attributes
    ----------
    id : str
        Identifier of the species.
    name : str
        Name of the species.
    species_location : SpeciesLocation
        Location of the species in the cell (Internal, External, Unknown).
    species_type : SpeciesType
        Type of the species (DNA, RNA, Protein, SmallMolecule, MacroMolecule,
        Unknown).
    formula : str
        Chemical formula of the species.
    mass : float
        Molecular mass of the species.

    Methods
    -------
    calculate_mass() -> None
        Calculate the molecular mass of the species.
    has_missing_mass( verbose: Optional[bool] = False ) -> bool
        Does the species have a missing mass (None or zero)?
    set_builder( builder ) -> None
        Set the reference to the model builder.
    build_dataframe() -> pd.DataFrame
        Build a pandas DataFrame with the species data.
    summary() -> None
        Print a summary of the model.
    """
    
    def __init__( self,
                  id: Optional[str] = None,
                  name: Optional[str] = None,
                  species_location: Optional[SpeciesLocation] = None,
                  species_type: Optional[SpeciesType] = None,
                  formula: Optional[str] = None,
                  mass: Optional[float] = None
                ) -> None:
        """
        Main constructor of the class

        Parameters
        ----------
        id : str
            Identifier of the species.
        name : str
            Name of the species.
        species_location : SpeciesLocation
            Location of the species in the cell (Internal, External, Unknown).
        species_type : SpeciesType
            Type of the species (DNA, RNA, Protein, SmallMolecule, MacroMolecule,
            Unknown).
        formula : str
            Chemical formula of the species.
        mass : float
            Molecular mass of the species.
        """
        self._builder         = None
        self.id               = id
        self.name             = name
        self.species_location = species_location
        self.species_type     = species_type
        self.formula          = formula
        self.mass             = mass
    
    def calculate_mass( self ) -> None:
        """
        Calculate the molecular mass of the species.
        """
        if self.species_type == SpeciesType.DNA and self.formula not in ["", None]:
            self.mass = SeqUtils.molecular_weight(self.formula, "DNA")
        elif self.species_type == SpeciesType.RNA and self.formula not in ["", None]:
            self.mass = SeqUtils.molecular_weight(self.formula, "RNA")
        elif self.species_type == SpeciesType.Protein and self.formula not in ["", None]:
            self.mass = ProteinAnalysis(self.formula).molecular_weight()
        elif self.species_type == SpeciesType.SmallMolecule and self.formula not in ["", None]:
            self.mass = molmass.Formula(self.formula).mass
        elif self.species_type in [SpeciesType.MacroMolecule, SpeciesType.Unknown] and self.formula not in ["", None]:
            try:
                formula   = self.formula.replace("R", "")
                self.mass = (molmass.Formula(formula).mass if formula != "" else 0.0)
            except:
                throw_message(MessageType.Warning, f"Could not calculate the molecular mass of <code>{self.id}</code>.")
        else:
            throw_message(MessageType.Warning, f"Could not calculate the molecular mass of <code>{self.id}</code>.")

    def has_missing_mass( self, verbose: Optional[bool] = False ) -> bool:
        """
        Does the species have a missing mass (None or zero)?

        Parameters
        ----------
        verbose : bool
            Verbosity of the output.
        """
        if self.mass == None or self.mass == 0.0:
            if verbose:
                throw_message(MessageType.Warning, f"Mass of species <code>{self.id}</code> is missing.")
            return True
        return False
    
    def set_builder( self, builder ) -> None:
        """
        Set the reference to the model builder.

        Parameters
        ----------
        builder : Builder
            Reference to the model builder.
        """
        self._builder = builder

    def build_dataframe( self ) -> pd.DataFrame:
        """
        Build a pandas DataFrame with the species data.

        Returns
        -------
        pd.DataFrame
            DataFrame with the species data.
        """
        df = {"Name": "-", "Location": "-", "Type": "-", "Formula": "-", "Mass": "-"}
        if self.name is not None:
            df["Name"] = self.name
        if self.species_location is not None:
            df["Location"] = ("Internal" if self.species_location == SpeciesLocation.Internal else
                              "External" if self.species_location == SpeciesLocation.External else
                              "Unknown")
        if self.species_type is not None:
            df["Type"] = ("DNA" if self.species_type == SpeciesType.DNA else
                          "RNA" if self.species_type == SpeciesType.RNA else
                          "Protein" if self.species_type == SpeciesType.Protein else
                          "Small molecule" if self.species_type == SpeciesType.SmallMolecule else
                          "Macro-molecule" if self.species_type == SpeciesType.MacroMolecule else
                          "Unknown")
        if self.formula is not None and self.formula != "":
            text = self.formula
            if len(text) > 20:
                text = text[:20] + "..."
            df["Formula"] = text
        if self.mass is not None:
            df["Mass"] = self.mass
        return pd.DataFrame.from_dict(df, orient="index", columns=[self.id])
    
    def summary( self ) -> None:
        """
        Print a summary of the species.
        """
        df       = self.build_dataframe()
        html_str = df.to_html(escape=False)
        display_html(html_str,raw=True)

class Protein(Species):
    """
    Class describing a protein species in the model.
    This class inherits from the Species class.

    Attributes
    ----------
    gene : str
        Gene encoding the protein.
    product : str
        Product of the gene (description).
    """
    
    def __init__( self,
                 id: Optional[str] = None,
                 name: Optional[str] = None,
                 sequence: Optional[str] = None,
                 mass: Optional[float] = None,
                 gene: Optional[str] = None,
                 product: Optional[str] = None
                ) -> None:
        """
        Main constructor of the class

        Parameters
        ----------
        id : str
            Identifier of the species.
        name : str
            Name of the species.
        sequence : str
            Amino-acid sequence of the protein.
        mass : float
            Molecular mass of the protein.
        gene : str
            Gene encoding the protein.
        product : str
            Product of the gene (description).
        """
        super().__init__(id, name, SpeciesLocation.Internal, SpeciesType.Protein, sequence, mass)
        self.gene    = gene
        self.product = product

class Metabolite(Species):
    """
    Class describing a metabolite species in the model.
    This class inherits from the Species class.

    Attributes
    ----------
    annotation : dict
        Annotation of the metabolite (dictionary of references).
    """
    
    def __init__( self,
                 id: Optional[str] = None,
                 name: Optional[str] = None,
                 species_location: Optional[SpeciesLocation] = None,
                 species_type: Optional[SpeciesType] = None,
                 formula: Optional[str] = None,
                 mass: Optional[float] = None,
                 annotation: Optional[dict] = None
                ) -> None:
        """
        Main constructor of the class

        Parameters
        ----------
        id : str
            Identifier of the species.
        name : str
            Name of the species.
        species_location : SpeciesLocation
            Location of the species in the cell (Internal, External, Unknown).
        species_type : SpeciesType
            Type of the species (DNA, RNA, Protein, SmallMolecule, MacroMolecule,
            Unknown).
        formula : str
            Chemical formula of the species.
        mass : float
            Molecular mass of the protein.
        annotation : dict
            Annotation of the metabolite (dictionary of references).
        """
        super().__init__(id, name, species_location, species_type, formula, mass)
        self.annotation = annotation

#~~~~~~~~~~~~~~~~~~~#
# Utility functions #
#~~~~~~~~~~~~~~~~~~~#

def throw_message( type: MessageType, message: str ) -> None:
    """
    Throw a message to the user.

    Parameters
    ----------
    type : MessageType
        Type of message (MessageType.Info, MessageType.Warning, MessageType.Error, MessageType.Plain).
    message : str
        Content of the message.
    """
    html_str  = "<table>"
    html_str += "<tr style='text-align:left'><td style='vertical-align:top'>"
    if type == MessageType.Plain:
        html_str += "<td><strong>&#10095;</strong></td>"
    elif type == MessageType.Info:
        html_str += "<td style='color:rgba(0,85,194);'><strong>&#10095; Info</strong></td>"
    elif type == MessageType.Warning:
        html_str += "<td style='color:rgba(240,147,1);'><strong>&#9888; Warning</strong></td>"
    elif type == MessageType.Error:
        html_str += "<td style='color:rgba(236,3,3);'><strong>&#10006; Error</strong></td>"
    html_str += "<td>"+message+"</td>"
    html_str += "</tr>"
    html_str += "</table>"
    display_html(html_str, raw=True)

