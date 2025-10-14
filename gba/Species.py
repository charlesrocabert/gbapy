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
Filename: Species.py
Author: Charles Rocabert
Date: 2024-16-12
Description:
    Species class of the gbapy module.
License: GNU General Public License v3 (GPLv3)
Copyright: © 2024-2025 Charles Rocabert.
"""

import molmass
import pandas as pd
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
        Location of the species in the cell (INTERNAL, EXTERNAL, UNKNOWN).
    species_type : SpeciesType
        Type of the species (DNA, RNA, PROTEIN, SMALLMOLECULE, MACROMOLECULE,
        UNKNOWN).
    formula : str
        Chemical formula of the species.
    mass : float
        Molecular mass of the species.
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
        Main constructor of the Species class.

        Parameters
        ----------
        id : str
            Identifier of the species.
        name : str
            Name of the species.
        species_location : SpeciesLocation
            Location of the species in the cell (INTERNAL, EXTERNAL, UNKNOWN).
        species_type : SpeciesType
            Type of the species (DNA, RNA, PROTEIN, SMALLMOLECULE,
            MACROMOLECULE, UNKNOWN).
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
        elif self.species_type == SpeciesType.PROTEIN and self.formula not in ["", None]:
            self.mass = ProteinAnalysis(self.formula).molecular_weight()
        elif self.species_type == SpeciesType.SMALLMOLECULE and self.formula not in ["", None]:
            self.mass = molmass.Formula(self.formula).mass
        elif self.species_type in [SpeciesType.MACROMOLECULE, SpeciesType.UNKNOWN] and self.formula not in ["", None]:
            try:
                formula   = self.formula.replace("R", "")
                self.mass = (molmass.Formula(formula).mass if formula != "" else 0.0)
            except:
                throw_message(MessageType.WARNING, f"Could not calculate the molecular mass of <code>{self.id}</code>.")
        else:
            throw_message(MessageType.WARNING, f"Could not calculate the molecular mass of <code>{self.id}</code>.")

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
                throw_message(MessageType.WARNING, f"Mass of species <code>{self.id}</code> is missing.")
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
            df["Location"] = ("Internal" if self.species_location == SpeciesLocation.INTERNAL else
                              "External" if self.species_location == SpeciesLocation.EXTERNAL else
                              "Unknown")
        if self.species_type is not None:
            df["Type"] = ("DNA" if self.species_type == SpeciesType.DNA else
                          "RNA" if self.species_type == SpeciesType.RNA else
                          "Protein" if self.species_type == SpeciesType.PROTEIN else
                          "Small molecule" if self.species_type == SpeciesType.SMALLMOLECULE else
                          "Macro-molecule" if self.species_type == SpeciesType.MACROMOLECULE else
                          "Unknown")
        if self.formula is not None and self.formula != "":
            text = self.formula
            if len(text) > 20:
                text = text[:20] + "..."
            df["Formula"] = text
        if self.mass is not None:
            df["Mass"] = self.mass
        if "gene" in self.__dict__:
            df["Gene"] = (self.gene if self.gene is not None else "-")
        if "product" in self.__dict__:
            df["Product"] = (self.product if self.product is not None else "-")
        if "essentiality" in self.__dict__:
            df["Essentiality"] = ("Essential" if self.essentiality == GeneEssentiality.ESSENTIAL else
                                  "Quasi-essential" if self.essentiality == GeneEssentiality.QUASI_ESSENTIAL else
                                  "Non-essential" if self.essentiality == GeneEssentiality.NON_ESSENTIAL else
                                  "Unknown")
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
    essentiality : GeneEssentiality
        Essentiality of the protein (essential, quasi-essential,
        non-essential, unknown).
    """
    
    def __init__( self,
                 id: Optional[str] = None,
                 name: Optional[str] = None,
                 sequence: Optional[str] = None,
                 mass: Optional[float] = None,
                 gene: Optional[str] = None,
                 product: Optional[str] = None,
                 essentiality: Optional[GeneEssentiality] = None
                ) -> None:
        """
        Main constructor of the Protein class.

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
        essentiality : GeneEssentiality
            Essentiality of the protein (essential, quasi-essential,
            non-essential, unknown).
        """
        super().__init__(id, name, SpeciesLocation.INTERNAL, SpeciesType.PROTEIN, sequence, mass)
        self.gene         = gene
        self.product      = product
        self.essentiality = essentiality

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
        Main constructor of the Metabolite class.

        Parameters
        ----------
        id : str
            Identifier of the species.
        name : str
            Name of the species.
        species_location : SpeciesLocation
            Location of the species in the cell (INTERNAL, EXTERNAL, UNKNOWN).
        species_type : SpeciesType
            Type of the species (DNA, RNA, PROTEIN, SMALLMOLECULE,
            MACROMOLECULE, UNKNWON).
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
        Type of message (MessageType.INFO, MessageType.WARNING,
        MessageType.ERROR, MessageType.PLAIN).
    message : str
        Content of the message.
    """
    html_str  = "<table>"
    html_str += "<tr style='text-align:left'><td style='vertical-align:top'>"
    if type == MessageType.PLAIN:
        html_str += "<td><strong>&#10095;</strong></td>"
    elif type == MessageType.INFO:
        html_str += "<td style='color:rgba(0,85,194);'><strong>&#10095; Info</strong></td>"
    elif type == MessageType.WARNING:
        html_str += "<td style='color:rgba(240,147,1);'><strong>&#9888; Warning</strong></td>"
    elif type == MessageType.ERROR:
        html_str += "<td style='color:rgba(236,3,3);'><strong>&#10006; Error</strong></td>"
    html_str += "<td>"+message+"</td>"
    html_str += "</tr>"
    html_str += "</table>"
    display_html(html_str, raw=True)

