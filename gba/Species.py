#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#***********************************************************************
# GBApy (Growth Balance Analysis for Python)
# Copyright © 2024-2025 Charles Rocabert
# Web: https://github.com/charlesrocabert/GBApy
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
Filename: Species.py
Author: Charles Rocabert
Date: 2024-16-12
Description:
    Species class of the GBApy module.
License: GNU General Public License v3.0
Copyright: © 2024-2025 Charles Rocabert
"""

import os
import sys
import molmass
from Bio import SeqIO
from typing import Optional
import Bio.SeqUtils as SeqUtils
from Bio.SeqUtils.ProtParam import ProteinAnalysis

try:
    from .Enumerations import *
    from .GbaBuilder import *
except:
    from Enumerations import *
    from GbaBuilder import *


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
    set_builder( builder: GbaBuilder ) -> None
        Set the reference to the GBA builder.
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
                print(f"> Warning: Could not calculate the molecular mass of '{self.id}'")
        else:
            print(f"> Warning: Could not calculate the molecular mass of '{self.id}'")

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
                print(f"> Warning: mass of species '{self.id}' is missing")
            return True
        return False
    
    def set_builder( self, builder ) -> None:
        """
        Set the reference to the GBA builder.

        Parameters
        ----------
        builder : GbaBuilder
            Reference to the GBA builder.
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
        if self.formula is not None:
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
        html_str = df.to_html()#.replace('table','table style="display:inline"')
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

