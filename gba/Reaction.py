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
Filename: Reaction.py
Author: Charles Rocabert
Date: 2024-17-12
Description:
    Reaction class of the GBApy module.
License: GNU General Public License v3.0
Copyright: © 2024-2025 Charles Rocabert
"""

import os
import sys
import numpy as np
from typing import Optional

try:
    from .Enumerations import *
    from .Species import *
    from .GbaBuilder import *
except:
    from Enumerations import *
    from Species import *
    from GbaBuilder import *


class Reaction:
    """
    Class describing an enzymatic reaction in the model.

    Attributes
    ----------
    id : str
        Identifier of the species.
    name : str
        Name of the species.
    reaction_type : ReactionType
        Type of the reaction (metabolic, transport, exchange).
    lb : float
        Lower bound of the reaction.
    ub : float
        Upper bound of the reaction.
    direction : ReactionDirection
        Direction of the reaction.
    metabolites : dict[str,float]
        Dictionary containing the metabolite IDs and their stoichiometry.
    reactants : list[str]
        List of reactants in the reaction.
    products : list[str]
        List of products in the reaction.
    expression : str
        Mathematical expression of the reaction.
    proteins : dict[str,float]
        Dictionary containing the protein IDs and their stoichiometry.
    GPR : ReactionGPR
        GPR logic of the reaction.
    enzyme_mass : float
        Molecular mass of the enzyme.
    protein_contribution : dict[str,float]
        Contribution of each protein to the proteomics.
    kcat : dict[ReactionDirection,float]
        Dictionary containing the kcat values of the reaction.
    km : dict[str,float]
        Dictionary containing the KM values of the reaction.
    GBA_metabolites : dict[str,float]
        Dictionary containing the metabolite IDs and their stoichiometry in GBA format.
    GBA_kcat : dict[ReactionDirection,float]
        Dictionary containing the kcat values of the reaction in GBA format.
    GBA_km : dict[str,float]
        Dictionary containing the KM values of the reaction in GBA format.
    kcat_is_converted : bool
        Are the kcat values converted to GBA format?
    km_is_converted : bool
        Are the KM values converted to GBA format?
    stoichiometry_is_converted : bool
        Is the stoichiometry converted to GBA format?

    Methods
    -------
    add_metabolites( metabolites: dict[str,float] ) -> None
        Add metabolites and their stoichiometry to the reaction.
    add_proteins( proteins: dict[str,float] ) -> None
        Add proteins and their stoichiometry to the enzyme composition.
    remove_metabolite( metabolite_id: str ) -> None
        Remove a metabolite from the stoichiometry of the reaction.
    remove_protein( protein_id: str ) -> None
        Remove a protein from the enzyme composition of the reaction.
    rename_metabolite( previous_id: str, new_id: str ) -> None
        Rename a metabolite in the stoichiometry of the reaction.
    add_kcat_value( direction: ReactionDirection, kcat_value: float ) -> None
        Add a kcat value to the reaction.
    add_km_value( metabolite_id: str, km_value: float ) -> None
        Add a KM value to the reaction.
    enforce_kcat_irreversibility() -> None
        Enforce the irreversibility of the reaction at the level of kcat values.
    enforce_km_irreversibility() -> None
        Enforce the irreversibility of the reaction at the level of KM values.
    complete_kcat_values( kcat_value: float ) -> None
        Complete the kcat values of the reaction.
    complete_km_values( km_value: float ) -> None
        Complete the KM values of the reaction.
    define_direction() -> None
        Define the direction of the reaction based on the lower and upper bounds.
    define_expression() -> None
        Define the expression of the reaction based on the metabolite stoichiometry
        and the direction.
    calculate_enzyme_mass() -> None
        Calculate the molecular mass of the enzyme based on its composition in proteins.
    has_missing_mass( verbose: Optional[bool] = False ) -> bool
        Does the reaction have a missing enzyme mass (None or zero)?
    has_missing_kcat_value( verbose: Optional[bool] = False ) -> bool
        Does the reaction have a missing kcat value?
    has_missing_km_value( verbose: Optional[bool] = False ) -> bool
        Does the reaction have a missing KM value?
    check_mass_balance( verbose: Optional[bool] = False, threshold: Optional[float] = 0.1 ) -> bool
        Check the mass balance of the reaction.
    check_mass_normalization( verbose: Optional[bool] = False, threshold: Optional[float] = 1e-8 ) -> bool
        Check if the mass stoichiometry of the reaction is normalized.
    check_no_conversion( verbose: Optional[bool] = False ) -> bool
        Check if the reaction has not been converted to GBA format.
    check_conversion( verbose: Optional[bool] = False ) -> bool
        Check if the reaction has been converted to GBA format.
    convert_kcat_values() -> None
        Convert the kcat values of the reaction to GBA format (mass units).
    convert_km_values() -> None
        Convert the KM values of the reaction to GBA format (mass units).
    convert_stoichiometry() -> None
        Convert the stoichiometry of the reaction to GBA format (normalized mass units).
    convert() -> None
        Convert the reaction to GBA format.
    reset_conversion() -> None
        Reset the conversion of the reaction to GBA format.
    set_builder( builder: GbaBuilder ) -> None
        Set the GBA builder for the reaction.
    build_dataframe() -> pd.DataFrame
        Build a DataFrame of the reaction
    summary() -> None
        Print a summary of the reaction.
    """

    def __init__( self,
                  id: Optional[str] = None,
                  name: Optional[str] = None,
                  reaction_type: Optional[ReactionType] = None,
                  lb: Optional[float] = None,
                  ub: Optional[float] = None,
                  metabolites: Optional[dict[str,float]] = None,
                  proteins: Optional[dict[str,float]] = None,
                  GPR: Optional[ReactionGPR] = None,
                  enzyme_mass: Optional[float] = None
                ) -> None:
        """
        Main constructor of the Reaction class.

        Parameters
        ----------
        id : str
            Identifier of the reaction.
        name : str
            Name of the reaction.
        reaction_type : ReactionType
            Type of the reaction (metabolic, transport, exchange).
        lb : float
            Lower bound of the reaction.
        ub : float
            Upper bound of the reaction.
        metabolites : dict[str,float]
            Dictionary containing the metabolite IDs and their
            stoichiometry.
        proteins : dict[str,float]
            Dictionary containing the protein IDs and their
            stoichiometry.
        GPR : ReactionGPR
            GPR logic of the reaction.
        enzyme_mass : float
            Molecular mass of the enzyme.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) General properties       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self._builder       = None
        self.id             = id
        self.name           = name
        self.reaction_type  = reaction_type
        self.lb             = lb
        self.ub             = ub
        self.direction      = None
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Reaction's stoichiometry #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.metabolites = None
        self.reactants   = []
        self.products    = []
        self.expression  = None
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Enzyme composition       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.proteins    = None
        self.GPR         = GPR
        self.enzyme_mass = enzyme_mass
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Proteomics               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.protein_contribution = None
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Kinetic parameters       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.kcat = None
        self.km   = None
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) GBA parameters           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.GBA_metabolites            = None
        self.GBA_kcat                   = None
        self.GBA_km                     = None
        self.kcat_is_converted          = False
        self.km_is_converted            = False
        self.stoichiometry_is_converted = False
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 7) Routine functions        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.add_metabolites(metabolites)
        self.add_proteins(proteins)
        self.define_direction()
        self.define_expression()
    
    #~~~~~~~~~~~~~~~~~~~~~#
    # 1) Setters          #
    #~~~~~~~~~~~~~~~~~~~~~#
    
    def add_metabolites( self, metabolites: dict[str,float] ) -> None:
        """
        Add metabolites and their stoichiometry to the reaction.

        Parameters
        ----------
        metabolites : dict[str,float]
            Dictionary containing the metabolite IDs and their
            stoichiomety.
        """
        assert self.check_no_conversion(), f"Reaction '{self.id}' has been converted to GBA format. Consider to reset the conversion."
        if metabolites == None:
            return
        if self.metabolites == None:
            self.metabolites = {}
        for m_id in metabolites:
            assert m_id not in self.metabolites, throw_message(MessageType.Error, f"Metabolite <code>{m_id}</code> already in the stoichiometry of reaction <code>{self.id}</code>.")
            assert metabolites[m_id] != 0, throw_message(MessageType.Error, f"Stoichiometry of metabolite <code>{m_id}</code> cannot be zero (reaction <code>{self.id}</code>).")
            self.metabolites[m_id] = metabolites[m_id]
            if self.metabolites[m_id] < 0:
                assert m_id not in self.reactants, throw_message(MessageType.Error, f"Metabolite <code>{m_id}</code> already in the list of reactants.")
                self.reactants.append(m_id)
            elif self.metabolites[m_id] > 0:
                assert m_id not in self.products, throw_message(MessageType.Error, f"Metabolite <code>{m_id}</code> already in the list of products.")
                self.products.append(m_id)
    
    def add_proteins( self, proteins: dict[str,float] ) -> None:
        """
        Add proteins and their stoichiometry to the enzyme composition.

        Parameters
        ----------
        proteins : dict[str,float]
            Dictionary containing the protein IDs and their
            stoichiomety.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        if proteins == None:
            return
        if self.proteins == None:
            self.proteins = {}
        for p_id in proteins:
            assert p_id not in self.proteins, throw_message(MessageType.Error, f"Protein <code>{p_id}</code> already in the enzyme composition of reaction <code>{self.id}</code>.")
            assert proteins[p_id] > 0, throw_message(MessageType.Error, f"Stoichiometry of protein <code>{p_id}</code> must be positive.")
            self.proteins[p_id] = proteins[p_id]

    def remove_metabolite( self, metabolite_id: str ) -> None:
        """
        Remove a metabolite from the stoichiometry of the reaction.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the metabolite to remove.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        assert self.metabolites != None, throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has no metabolites.")
        assert metabolite_id in self.metabolites, throw_message(MessageType.Error, f"Metabolite <code>{metabolite_id}</code> not in the stoichiometry of reaction <code>{self.id}</code>.")
        del self.metabolites[metabolite_id]
        if len(self.metabolites) == 0:
            throw_message(MessageType.Warning, f"Reaction <code>{self.id}</code> has no metabolites.")
        if metabolite_id in self.reactants:
            self.reactants.remove(metabolite_id)
            if len(self.reactants) == 0:
                throw_message(MessageType.Warning, f"Reaction <code>{self.id}</code> has no reactants.")
        elif metabolite_id in self.products:
            self.products.remove(metabolite_id)
            if len(self.products) == 0:
                throw_message(MessageType.Warning, f"Reaction <code>{self.id}</code> has no products.")
        if not self.km is None and metabolite_id in self.km:
            del self.km[metabolite_id]
    
    def remove_protein( self, protein_id: str ) -> None:
        """
        Remove a protein from the enzyme composition of the reaction.

        Parameters
        ----------
        protein_id : str
            Identifier of the protein to remove.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        assert self.proteins != None, throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has no proteins.")
        assert protein_id in self.proteins, throw_message(MessageType.Error, f"Protein <code>{protein_id}</code> not in the enzyme composition of reaction <code>{self.id}</code>.")
        del self.proteins[protein_id]
        assert len(self.proteins) > 0, throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> must have at least one protein.")
    
    def rename_metabolite( self, previous_id: str, new_id: str ) -> None:
        """
        Rename a metabolite in the stoichiometry of the reaction.

        Parameters
        ----------
        previous_id : str
            Previous identifier of the metabolite.
        new_id : str
            New identifier of the metabolite.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        assert previous_id in self.metabolites, throw_message(MessageType.Error, f"Metabolite <code>{previous_id}</code> not in the stoichiometry of reaction <code>{self.id}</code>.")
        assert new_id not in self.metabolites, throw_message(MessageType.Error, f"Metabolite <code>{new_id}</code> already in the stoichiometry of reaction <code>{self.id}</code>.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Update the stoichiometries    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.metabolites[new_id] = self.metabolites[previous_id]
        del self.metabolites[previous_id]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Update reactants and products #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if previous_id in self.reactants:
            self.reactants.remove(previous_id)
            self.reactants.append(new_id)
        elif previous_id in self.products:
            self.products.remove(previous_id)
            self.products.append(new_id)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Update KM values              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if not self.km is None and previous_id in self.km:
            self.km[new_id] = self.km[previous_id]
            del self.km[previous_id]
        
    def add_kcat_value( self, direction: ReactionDirection, kcat_value: float ) -> None:
        """
        Add a kcat value to the reaction.

        Parameters
        ----------
        direction : ReactionDirection
            Direction of the kcat value.
        kcat_value : float
            kcat value to add.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction '{self.id}' has been converted to GBA format. Consider to reset the conversion.")
        if direction == None or kcat_value == None:
            return
        if self.kcat == None:
            self.kcat = {}
        assert direction in [ReactionDirection.Forward, ReactionDirection.Backward], throw_message(MessageType.Error, f"Direction <code>{dir}</code> not recognized for reaction <code>{self.id}</code>.")
        assert kcat_value >= 0.0, throw_message(MessageType.Error, f"kcat value must be positive or null (reaction <code>{self.id}</code>).")
        self.kcat[direction] = kcat_value
    
    def add_km_value( self, metabolite_id: str, km_value: float ) -> None:
        """
        Add a KM value to the reaction.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the metabolite.
        km_value : float
            KM value to add.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        if metabolite_id == None or km_value == None:
            return
        if self.km == None:
            self.km = {}
        assert metabolite_id in self.metabolites, throw_message(MessageType.Error, f"Metabolite <code>{metabolite_id}</code> not in the stoichiometry of reaction <code>{self.id}</code>.")
        assert km_value >= 0.0, throw_message(MessageType.Error, f"KM value must be positive or null (reaction <code>{self.id}</code>).")
        self.km[metabolite_id] = km_value

    def enforce_kcat_irreversibility( self ) -> None:
        """
        Enforce the irreversibility of the reaction at the level of kcat values.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        if self.kcat == None:
            self.kcat = {}
        if self.direction == ReactionDirection.Forward:
            self.kcat[ReactionDirection.Backward] = 0.0
        elif self.direction == ReactionDirection.Backward:
            self.kcat[ReactionDirection.Forward] = 0.0
    
    def enforce_km_irreversibility( self ) -> None:
        """
        Enforce the irreversibility of the reaction at the level of KM values.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        if self.km == None:
            self.km = {}
        if self.direction == ReactionDirection.Forward:
            for m_id in self.products:
                self.km[m_id] = 0.0
        elif self.direction == ReactionDirection.Backward:
            for m_id in self.reactants:
                self.km[m_id] = 0.0

    def complete_kcat_values( self, kcat_value: float ) -> None:
        """
        Complete the kcat values of the reaction.

        Parameters
        ----------
        kcat_value : float
            kcat value to add.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        assert kcat_value >= 0.0, throw_message(MessageType.Error, f"kcat value must be positive or null (reaction <code>{self.id}</code>).")
        if self.kcat == None or len(self.kcat) == 0:
            self.kcat = {ReactionDirection.Backward: kcat_value, ReactionDirection.Forward: kcat_value}
        elif len(self.kcat) == 1 and ReactionDirection.Backward in self.kcat:
            self.kcat[ReactionDirection.Forward] = kcat_value
        elif len(self.kcat) == 1 and ReactionDirection.Forward in self.kcat:
            self.kcat[ReactionDirection.Backward] = kcat_value
    
    def complete_km_values( self, km_value: float ) -> None:
        """
        Complete the KM values of the reaction.

        Parameters
        ----------
        km_value : float
            KM value to add.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        assert km_value >= 0.0, throw_message(MessageType.Error, f"KM value must be positive or null (reaction <code>{self.id}</code>).")
        if self.km == None or len(self.km) == 0:
            self.km = {}
        for m_id in self.metabolites:
            if m_id not in self.km:
                self.km[m_id] = km_value
    
    def complete( self, kcat_value: float, km_value: float ) -> None:
        """
        Complete the reaction with kcat and KM values.

        Parameters
        ----------
        kcat_value : float
            kcat value to add.
        km_value : float
            KM value to add.
        """
        self.complete_kcat_values(kcat_value)
        self.complete_km_values(km_value)
    
    #~~~~~~~~~~~~~~~~~~~~~#
    # 2) General routines #
    #~~~~~~~~~~~~~~~~~~~~~#
    
    def define_direction( self ) -> None:
        """
        Define the direction of the reaction based on the lower and
        upper bounds.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        if not isinstance(self.lb, float) or not isinstance(self.ub, float):
            return
        assert self.lb <= self.ub, throw_message(MessageType.Error, f"Lower bound must be lower or equal to the upper bound for reaction <code>{self.id}</code>.")
        if self.lb < 0 and self.ub > 0:
            self.direction = ReactionDirection.Reversible
        elif self.lb >= 0 and self.ub > 0:
            self.direction = ReactionDirection.Forward
        elif self.lb < 0 and self.ub <= 0:
            self.direction = ReactionDirection.Backward
    
    def define_expression( self ) -> None:
        """
        Define the expression of the reaction based on the metabolite
        stoichiometry and the direction.
        """
        if self.metabolites == None or len(self.metabolites) == 0 or not isinstance(self.direction, ReactionDirection):
            return
        self.expression = ""
        self.expression = " + ".join([str(np.abs(self.metabolites[m_id]))+" "+m_id for m_id in self.metabolites if self.metabolites[m_id] < 0])
        if self.direction == ReactionDirection.Reversible:
            self.expression += " <=> "
        elif self.direction == ReactionDirection.Forward:
            self.expression += " --> "
        elif self.direction == ReactionDirection.Backward:
            self.expression += " <-- "
        self.expression += " + ".join([str(self.metabolites[m_id])+" "+m_id for m_id in self.metabolites if self.metabolites[m_id] > 0])
        self.expression  = self.expression.replace(" 1.0 ", "  ")
    
    def calculate_enzyme_mass( self ) -> None:
        """
        Calculate the molecular mass of the enzyme based on its
        composition in proteins.
        """
        assert self.check_no_conversion(), throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has been converted to GBA format. Consider to reset the conversion.")
        if self.proteins == None:
            return
        if self.GPR == ReactionGPR.NONE and len(self.proteins) > 1:
            throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has multiple proteins but no GPR logic.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Calculate the molecular mass of the enzyme               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.enzyme_mass          = 0.0
        self.protein_contribution = {}
        for p_id in self.proteins:
            assert p_id in self._builder.proteins, throw_message(MessageType.Error, f"Protein <code>{p_id}</code> not found in the list of proteins.")
            self.enzyme_mass += self._builder.proteins[p_id].mass*self.proteins[p_id]
        if self.GPR == ReactionGPR.OR:
            self.enzyme_mass /= len(self.proteins)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Calculate the contribution of each protein to proteomics #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # [p] = [E] * (m_p * stoic_p / m_E) (in g/L)
        self.protein_contribution = {p_id: self._builder.proteins[p_id].mass*self.proteins[p_id]/self.enzyme_mass for p_id in self.proteins}
    
    def has_missing_mass( self, verbose: Optional[bool] = False ) -> bool:
        """
        Does the reaction have a missing enzyme mass (None or zero)?

        Parameters
        ----------
        verbose : bool
            Verbosity of the output.
        """
        if self.enzyme_mass == None or self.enzyme_mass == 0.0:
            if verbose:
                throw_message(MessageType.Warning, f"Enzyme mass of reaction <code>{self.id}</code> is missing.")
            return True
        return False

    def has_missing_kcat_value( self, verbose: Optional[bool] = False ) -> bool:
        """
        Does the reaction have a missing kcat value?

        Parameters
        ----------
        verbose : bool
            Verbosity of the output.
        """
        if self.kcat == None or len(self.kcat) == 0:
            if verbose:
                throw_message(MessageType.Warning, f"No defined kcat value for reaction <code>{self.id}</code>.")
            return True
        elif len(self.kcat) == 1 and ReactionDirection.Backward in self.kcat:
            if verbose:
                throw_message(MessageType.Warning, f"Forward kcat value is missing for reaction <code>{self.id}</code>.")
            return True
        elif len(self.kcat) == 1 and ReactionDirection.Forward in self.kcat:
            if verbose:
                throw_message(MessageType.Warning, f"Backward kcat value is missing for reaction <code>{self.id}</code>.")
            return True
        elif len(self.kcat) == 2:
            return False
    
    def has_missing_km_value( self, verbose: Optional[bool] = False ) -> bool:
        """
        Does the reaction have a missing KM value?

        Parameters
        ----------
        verbose : bool
            Verbosity of the output.
        """
        if self.km == None or len(self.km) == 0:
            if verbose:
                throw_message(MessageType.Warning, f"No defined KM value for reaction <code>{self.id}</code>.")
            return True
        for m_id in self.metabolites:
            if m_id not in self.km:
                if verbose:
                    throw_message(MessageType.Warning, f"KM value is missing for the pair <code>{self.id}</code>, <code>{m_id}</code>.")
                return True
        return False
    
    #~~~~~~~~~~~~~~~~~~~~~#
    # 3) GBA routines     #
    #~~~~~~~~~~~~~~~~~~~~~#
    
    def check_mass_balance( self, verbose: Optional[bool] = False, threshold: Optional[float] = 0.1 ) -> bool:
        """
        Check the mass balance of the reaction.

        Parameters
        ----------
        verbose : bool
            Verbosity of the output.
        threshold : float
            Threshold for the mass balance (in Da).
        """
        reactants_mass = 0.0
        products_mass  = 0.0
        for m_id in self.metabolites:
            if self.metabolites[m_id] < 0:
                reactants_mass += np.abs(self.metabolites[m_id])*self._builder.metabolites[m_id].mass
            elif self.metabolites[m_id] > 0:
                products_mass += self.metabolites[m_id]*self._builder.metabolites[m_id].mass
        diff = np.abs(reactants_mass-products_mass)
        if diff > threshold:
            if verbose:
                throw_message(MessageType.Warning, f"No mass balance for reaction <code>{self.id}</code> (diff = {products_mass-reactants_mass}Da, threshold = {threshold}Da).")
            return False
        return True
    
    def check_mass_normalization( self, verbose: Optional[bool] = False, threshold: Optional[float] = 1e-8 ) -> bool:
        """
        Check if the mass stoichiometry of the reaction is normalized.

        Parameters
        ----------
        verbose : bool
            Verbosity of the output.
        threshold : float
            Threshold for the mass balance (in Da).
        """
        reactant_sum = 0.0
        product_sum  = 0.0
        for m_id in self.GBA_metabolites:
            if self.GBA_metabolites[m_id] < 0:
                reactant_sum += np.abs(self.GBA_metabolites[m_id])
            elif self.GBA_metabolites[m_id] > 0:
                product_sum += self.GBA_metabolites[m_id]
        if np.abs(reactant_sum-1.0) > threshold or np.abs(product_sum-1.0) > threshold:
            if verbose:
                throw_message(MessageType.Warning, f"Stoichiometry of reaction <code>{self.id}</code> is not normalized (threshold = {threshold}Da).")
            return False
        return True
    
    def check_no_conversion( self, verbose: Optional[bool] = False ) -> bool:
        """
        Check if the reaction has not been converted to GBA format.

        Parameters
        ----------
        verbose : bool
            Verbosity of the output.
        """
        converted = False
        if self.kcat_is_converted:
            converted = True
        if self.km_is_converted:
            converted = True
        if self.stoichiometry_is_converted:
            converted = True
        if verbose:
            throw_message(MessageType.Warning, f"A GBA conversion exists for reaction <code>{self.id}</code>. Consider to reset the conversion.")
        return not converted
    
    def check_conversion( self, verbose: Optional[bool] = False ) -> bool:
        """
        Check if the reaction has been converted to GBA format.

        Parameters
        ----------
        verbose : bool
            Verbosity of the output.
        """
        if not self.kcat_is_converted:
            if verbose:
                throw_message(MessageType.Warning, f"kcat values of reaction <code>{self.id}</code> have not been converted to GBA format.")
            return False
        if not self.km_is_converted:
            if verbose:
                throw_message(MessageType.Warning, f"KM values of reaction <code>{self.id}</code> have not been converted to GBA format.")
            return False
        if not self.stoichiometry_is_converted:
            if verbose:
                throw_message(MessageType.Warning, f"Stoichiometry of reaction <code>{self.id}</code> has not been converted to GBA format.")
            return False
        return True
    
    def convert_kcat_values( self ) -> None:
        """
        Convert the kcat values of the reaction to GBA format (mass units).
        """
        assert self._builder != None, throw_message(MessageType.Error, f"GBA builder not set for reaction <code>{self.id}</code>.")
        assert self.kcat != None and len(self.kcat) > 0, throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has no kcat values.")
        reactant_sum = 0.0
        product_sum  = 0.0
        for m_id in self.metabolites:
            w = self._builder.metabolites[m_id].mass
            if self.metabolites[m_id] < 0:
                reactant_sum += np.abs(self.metabolites[m_id])*w
            else:
                product_sum += np.abs(self.metabolites[m_id])*w
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Normalize the kcat values       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.GBA_kcat = self.kcat.copy()
        for r_dir in self.GBA_kcat:
            if r_dir == ReactionDirection.Forward:
                self.GBA_kcat[r_dir] *= product_sum/self.enzyme_mass
            elif r_dir == ReactionDirection.Backward:
                self.GBA_kcat[r_dir] *= reactant_sum/self.enzyme_mass
        self.kcat_is_converted = True
    
    def convert_km_values( self ) -> None:
        """
        Convert the KM values of the reaction to GBA format (mass units).
        """
        assert self._builder != None, throw_message(MessageType.Error, f"GBA builder not set for reaction <code>{self.id}</code>.")
        assert self.km != None and len(self.km) > 0, throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has no KM values.")
        for m_id in self.km:
            assert m_id in self.metabolites, throw_message(MessageType.Error, f"Metabolite <code>{m_id}</code> not found in the stoichiometry of reaction <code>{self.id}</code>.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Convert KM values to mass units     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.GBA_km = self.km.copy()
        for m_id in self.km:
            self.GBA_km[m_id] *= self._builder.metabolites[m_id].mass
        self.km_is_converted = True
    
    def convert_stoichiometry( self ) -> None:
        """
        Convert the stoichiometry of the reaction to GBA format (normalized mass units).
        """
        assert self._builder != None, throw_message(MessageType.Error, f"GBA builder not set for reaction <code>{self.id}</code>.")
        assert self.metabolites != None and len(self.metabolites) > 0, throw_message(MessageType.Error, f"Reaction <code>{self.id}</code> has no metabolites.")
        for m_id in self.metabolites:
            assert m_id in self._builder.metabolites, throw_message(MessageType.Error, f"Metabolite <code>{m_id}</code> not found in the list of metabolites.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Convert stoichiometry to mass units #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.GBA_metabolites = self.metabolites.copy()
        for m_id in self.GBA_metabolites:
            self.GBA_metabolites[m_id] *= self._builder.metabolites[m_id].mass
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Normalize the mass stoichiometry    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        reactant_sum = 0.0
        product_sum  = 0.0
        for m_id in self.GBA_metabolites:
            if self.GBA_metabolites[m_id] < 0:
                reactant_sum += np.abs(self.GBA_metabolites[m_id])
            elif self.GBA_metabolites[m_id] > 0:
                product_sum += self.GBA_metabolites[m_id]
        for m_id in self.GBA_metabolites:
            if self.GBA_metabolites[m_id] < 0:
                self.GBA_metabolites[m_id] /= reactant_sum
            elif self.GBA_metabolites[m_id] > 0:
                self.GBA_metabolites[m_id] /= product_sum
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Check mass normalization            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.check_mass_normalization(verbose=True)
        self.stoichiometry_is_converted = True
    
    def convert( self ) -> None:
        """
        Convert the reaction to GBA format.
        """
        self.convert_kcat_values()
        self.convert_km_values()
        self.convert_stoichiometry()
    
    def reset_conversion( self ) -> None:
        """
        Reset the conversion of the reaction to GBA format.
        """
        self.kcat_is_converted          = False
        self.km_is_converted            = False
        self.stoichiometry_is_converted = False
        self.GBA_metabolites.clear()
        self.GBA_kcat.clear()
        self.GBA_km.clear()
        self.GBA_metabolites = None
        self.GBA_kcat        = None
        self.GBA_km          = None

    #~~~~~~~~~~~~~~~~~~~~~#
    # 4) Other methods    #
    #~~~~~~~~~~~~~~~~~~~~~#

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
        Build a pandas DataFrame with reaction's data.

        Returns
        -------
        pd.DataFrame
            DataFrame with reaction's data.
        """
        df = {"Name": "-", "Type": "-", "Stoichiometry": "-",
              "Lower bound": "-", "Upper bound": "-",
              "Enzyme mass": "-", "GPR": "-", "Proteins": "-",
              "Forward kcat": "-", "Backward kcat": "-", "KM values": "-"}
        if self.name is not None:
            df["Name"] = self.name
        if self.reaction_type is not None:
            df["Type"] = ("Metabolic" if self.reaction_type == ReactionType.Metabolic else
                          "Transport" if self.reaction_type == ReactionType.Transport else
                          "Exchange" if self.reaction_type == ReactionType.Exchange else
                          "Unknown")
        if self.expression is not None:
            df["Stoichiometry"] = self.expression
        if self.lb is not None:
            df["Lower bound"] = self.lb
        if self.ub is not None:
            df["Upper bound"] = self.ub
        if self.enzyme_mass is not None:
            df["Enzyme mass"] = self.enzyme_mass
        if self.GPR is not None:
            df["GPR"] = ("AND" if self.GPR == ReactionGPR.AND else
                         "OR" if self.GPR == ReactionGPR.OR else
                         "NONE")
        if self.proteins is not None:
            df["Proteins"] = " + ".join([f"{self.proteins[p_id]} {p_id}" for p_id in self.proteins])
        if self.kcat is not None:
            df["Forward kcat"] = self.kcat[ReactionDirection.Forward] if ReactionDirection.Forward in self.kcat else "-"
            df["Backward kcat"] = self.kcat[ReactionDirection.Backward] if ReactionDirection.Backward in self.kcat else "-"
        if self.km is not None:
            df["KM values"] = " , ".join([f"{m_id}: {self.km[m_id]}" for m_id in self.km])
        return pd.DataFrame.from_dict(df, orient="index", columns=[self.id])
    
    def summary( self ) -> None:
        """
        Print a summary of the reaction.
        """
        self.define_expression()
        df       = self.build_dataframe()
        html_str = df.to_html(escape=False, index=False)
        display_html(html_str,raw=True)

#~~~~~~~~~~~~~~~~~~~#
# Utility functions #
#~~~~~~~~~~~~~~~~~~~#

def throw_message( type: MessageType, message: str ) -> None:
        """
        Throw a message to the user.

        Parameters
        ----------
        type : MessageType
            Type of message (MessageType.Info, MessageType.Warning, MessageType.Error).
        message : str
            Content of the message.
        """
        html_str  = "<table>"
        html_str += "<tr style='text-align:left'><td style='vertical-align:top'>"
        if type == MessageType.Info:
            html_str += "<td style='color:rgba(0,85,194);'><strong>&#10095; Info</strong></td>"
        elif type == MessageType.Warning:
            html_str += "<td style='color:rgba(240,147,1);'><strong>&#9888; Warning</strong></td>"
        elif type == MessageType.Error:
            html_str += "<td style='color:rgba(236,3,3);'><strong>&#10006; Error</strong></td>"
        html_str += "<td>"+message+"</td>"
        html_str += "</tr>"
        html_str += "</table>"
        display_html(html_str,raw=True)

