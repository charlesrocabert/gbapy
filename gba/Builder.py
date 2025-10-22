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
Filename: Builder.py
Author: Charles Rocabert
Date: 2024-16-12
Description:
    Builder class of the gbapy module.
License: GNU General Public License v3 (GPLv3)
Copyright: © 2024-2025 Charles Rocabert.
"""

import os
import sys
import cobra
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from IPython.display import display_html

try:
    from .Enumerations import *
    from .Species import Protein, Metabolite
    from .Reaction import Reaction
except:
    from Enumerations import *
    from Species import Protein, Metabolite
    from Reaction import Reaction


class Builder:
    """
    Class to manage the construction of models of any size.

    Attributes
    ----------
    name : str
        Name of the model build.
    info : dict[str, str]
        Dictionary of information about the model.
    proteins : dict[str, Protein]
        Dictionary of proteins.
    metabolites : dict[str, Metabolite]
        Dictionary of metabolites.
    reactions : dict[str, Reaction]
        Dictionary of reactions.
    FBA_biomass_reaction : Reaction
        Reconstructed FBA biomass reaction.
    FBA_wrapper_reactions : dict[str, Reaction]
        Reconstructed FBA wrapper reactions.
    FBA_model : cobra.Model
        Reconstructed FBA model.
    FBA_solution : cobra.Solution
        Reconstructed FBA solution.
    FBA_row_indices : dict[str, int]
        FBA row indices (metabolite ID to index map).
    FBA_external_row_indices : dict[str, int]
        FBA external row indices (metabolite ID to index map).
    FBA_internal_row_indices : dict[str, int]
        FBA internal row indices (metabolite ID to index map).
    FBA_col_indices : dict[str, int]
        FBA column indices (reaction ID to index map).
    FBA_S : numpy.ndarray
        FBA stoichiometry matrix.
    FBA_intS: numpy.ndarray
        FBA internal stoichiometry matrix.
    FBA_column_rank: int
        FBA internal stoichiometry matrix column rank.
    FBA_is_full_column_rank: bool
        Is the FBA internal stoichiometry matrix full column rank?
    FBA_dependent_reactions: list[str]
        List of linearly dependent reactions in the FBA internal stoichiometry
        matrix.
    FBA_inactive_reactions: list[str]
        List of inactive reactions in the FBA solution.
    FBA_is_built : bool
        Is the FBA model built?
    GBA_row_indices : dict[str, int]
        GBA row indices (metabolite ID to index map).
    GBA_external_row_indices : dict[str, int]
        GBA external row indices (external metabolite ID to index map).
    GBA_internal_row_indices : dict[str, int]
        GBA internal row indices (internal metabolite ID to index map).
    GBA_col_indices : dict[str, int]
        GBA column indices (reaction ID to index map).
    GBA_M : numpy.ndarray
        GBA complete mass fraction matrix.
    GBA_intM : numpy.ndarray
        GBA internal mass fraction matrix.
    GBA_kcat_f : numpy.array
        GBA forward kcat vector.
    GBA_kcat_b : numpy.array
        GBA backward kcat vector.
    GBA_KM_f : numpy.array
        GBA forward KM matrix.
    GBA_KM_b : numpy.array
        GBA backward KM matrix.
    GBA_KA : numpy.array
        GBA activation constant matrix.
    GBA_KI : numpy.array
        GBA inhibition constant matrix.
    GBA_conditions : dict[int, dict[str, float]]
        GBA conditions matrix.
    GBA_constant_rhs : dict[str, float]
        GBA metabolites with a constant RHS term for the initial solution
    GBA_constant_reactions : dict[str, float]
        GBA reactions with a constant flux value.
    GBA_protein_contributions: dict[str, float]
        Enzyme to protein mass concentration mapping.
    GBA_column_rank: int
        Internal mass fraction matrix column rank.
    GBA_is_full_column_rank: bool
        Is the internal mass fraction matrix full column rank?
    GBA_dependent_reactions: list[str]
        List of linearly dependent reaction in the internal mass fraction
        matrix.
    GBA_is_built : bool
        Is the GBA converted model built?
    GBA_initial_solution : dict[str, float]
        Initial solution of the GBA model.
    """

    def __init__( self, name ):
        """
        Constructor of the Builder class.
        
        Parameters
        ----------
        name : str
            Name of the model build.
        """
        assert name != "", throw_message(MessageType.ERROR, "Empty model build name.")
        self.name = name
        self.info = {}
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Main molecular species and reactions #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.proteins    = {}
        self.metabolites = {}
        self.reactions   = {}
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Default kinetic parameters           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.default_kcat = {ReactionType.METABOLIC: 0.0, 
                             ReactionType.TRANSPORT: 0.0, 
                             ReactionType.SPONTANEOUS: 0.0}
        self.default_km   = {ReactionType.METABOLIC: 0.0,
                             ReactionType.TRANSPORT: 0.0,
                             ReactionType.SPONTANEOUS: 0.0}
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) FBA model reconstruction             #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.FBA_biomass_reaction     = None
        self.FBA_wrapper_reactions    = {}
        self.FBA_model                = None
        self.FBA_solution             = None
        self.FBA_row_indices          = {}
        self.FBA_external_row_indices = {}
        self.FBA_internal_row_indices = {}
        self.FBA_col_indices          = {}
        self.FBA_S                    = None
        self.FBA_intS                 = None
        self.FBA_column_rank          = 0
        self.FBA_is_full_column_rank  = False
        self.FBA_dependent_reactions  = []
        self.FBA_inactive_reactions   = []
        self.FBA_is_built             = False
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) GBA reconstruction                   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.GBA_row_indices           = {}
        self.GBA_external_row_indices  = {}
        self.GBA_internal_row_indices  = {}
        self.GBA_col_indices           = {}
        self.GBA_M                     = None
        self.GBA_intM                  = None
        self.GBA_kcat_f                = None
        self.GBA_kcat_b                = None
        self.GBA_KM_f                  = None
        self.GBA_KM_b                  = None
        self.GBA_KA                    = None
        self.GBA_KI                    = None
        self.GBA_rho                   = 0.0
        self.GBA_conditions            = {}
        self.GBA_constant_rhs          = {}
        self.GBA_constant_reactions    = {}
        self.GBA_protein_contributions = {}
        self.GBA_column_rank           = 0
        self.GBA_is_full_column_rank   = False
        self.GBA_dependent_reactions   = []
        self.GBA_is_built              = False
        self.GBA_initial_solution      = {}

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Getters                  #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def which_reaction( self, metabolite_id: str ) -> list[str]:
        """
        Get the reactions in which a metabolite participates.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the metabolite.
        """
        assert metabolite_id in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite_id}</code> does not exist.")
        return [r_id for r_id, reaction in self.reactions.items() if metabolite_id in reaction.metabolites]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Setters                  #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def add_info( self, category: str, key: str, content: str ) -> None:
        """
        Add a piece of information to the model.

        Parameters
        ----------
        category : str
            Category of the information (e.g., "Description", "Units", "Sheets",
            "Authors", ...)
        key : str
            Key of the information.
        content : str
            Content of the information.
        """
        if category not in self.info:
            self.info[category] = {}
        assert key not in self.info[category], throw_message(MessageType.ERROR, f"Info key <code>{key}</code> already exists in info category <code>{category}</code>.")
        self.info[category][key] = content
    
    def add_protein( self, protein: Protein ) -> None:
        """
        Add a protein to the model.

        Parameters
        ----------
        protein : Protein
            Protein object to add to the model.
        """
        #assert isinstance(protein, Protein), throw_message(MessageType.ERROR, f"Expected <code>protein</code> to be a Protein, but got <code>{type(protein).__name__}</code>.")
        assert protein.id not in self.proteins, throw_message(MessageType.ERROR, f"Protein <code>{protein.id}</code> already exists.") 
        protein.set_builder(self)
        self.proteins[protein.id] = protein
    
    def add_proteins( self, proteins_list: list[Protein] ) -> None:
        """
        Add a list of proteins to the model.

        Parameters
        ----------
        proteins_list : list[Protein]
            List of Protein objects to add to the model.
        """
        for protein in proteins_list:
            self.add_protein(protein)
    
    def add_metabolite( self, metabolite: Metabolite ) -> None:
        """
        Add a metabolite to the model.

        Parameters
        ----------
        metabolite : Metabolite
            Metabolite object to add to the model.
        """
        #assert isinstance(metabolite, Metabolite), throw_message(MessageType.ERROR, f"Expected <code>metabolite</code> to be a Metabolite, but got <code>{type(metabolite).__name__}</code>.") 
        assert metabolite.id not in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite.id}</code> already exists.")
        metabolite.set_builder(self)
        self.metabolites[metabolite.id] = metabolite
    
    def add_metabolites( self, metabolites_list: list[Metabolite] ) -> None:
        """
        Add a list of metabolites to the model.

        Parameters
        ----------
        metabolites_list : list[Metabolite]
            List of Metabolite objects to add to the model.
        """
        for metabolite in metabolites_list:
            self.add_metabolite(metabolite)
    
    def add_reaction( self, reaction: Reaction ) -> None:
        """
        Add a reaction to the model.

        Parameters
        ----------
        reaction : Reaction
            Reaction object to add to the model.
        """
        #assert isinstance(reaction, Reaction), throw_message(MessageType.ERROR, f"Expected <code>reaction</code> to be a Reaction, but got <code>{type(reaction).__name__}</code>.") 
        assert reaction.id not in self.reactions, throw_message(MessageType.ERROR, f"Reaction <code>{reaction.id}</code> already exists.")
        reaction.set_builder(self)
        self.reactions[reaction.id] = reaction
        if reaction.proteins not in [None, {}]:# and reaction.GPR is not None:
            reaction.calculate_enzyme_mass()
    
    def add_reactions( self, reactions_list: list[Reaction] ) -> None:
        """
        Add a list of reactions to the model.

        Parameters
        ----------
        reactions_list : list[Reaction]
            List of Reaction objects to add to the model.
        """
        for reaction in reactions_list:
            self.add_reaction(reaction)
    
    def remove_protein( self, protein_id: str ) -> None:
        """
        Remove a protein from the model.

        Parameters
        ----------
        protein_id : str
            Identifier of the protein to remove.
        """
        assert protein_id in self.proteins, throw_message(MessageType.ERROR, f"Protein <code>{protein_id}</code> does not exist.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Remove the protein from the main dictionary #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        del self.proteins[protein_id]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Remove the protein from reactions           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for reaction in self.reactions.values():
            if protein_id in reaction.proteins:
                reaction.remove_protein(protein_id)
    
    def remove_proteins( self, proteins_list: list[str] ) -> None:
        """
        Remove a list of proteins from the model.

        Parameters
        ----------
        proteins_list : list[str]
            List of protein identifiers to remove.
        """
        for p_id in proteins_list:
            self.remove_protein(p_id)
    
    def remove_metabolite( self, metabolite_id: str ) -> None:
        """
        Remove a metabolite from the model.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the metabolite to remove.
        """
        assert metabolite_id in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite_id}</code> does not exist.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Remove the metabolite from the main dictionary #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        del self.metabolites[metabolite_id]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Remove the metabolite from reactions           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for reaction in self.reactions.values():
            if metabolite_id in reaction.metabolites:
                reaction.remove_metabolite(metabolite_id)
    
    def remove_metabolites( self, metabolites_list: list[str] ) -> None:
        """
        Remove a list of metabolites from the model.

        Parameters
        ----------
        metabolites_list : list[str]
            List of metabolite identifiers to remove.
        """
        for m_id in metabolites_list:
            self.remove_metabolite(m_id)
    
    def remove_reaction( self, reaction_id: str ) -> None:
        """
        Remove a reaction from the model.

        Parameters
        ----------
        reaction_id : str
            Identifier of the reaction to remove.
        """
        assert reaction_id in self.reactions, throw_message(MessageType.ERROR, f"Reaction <code>{reaction_id}</code> does not exist.")
        del self.reactions[reaction_id]
    
    def remove_reactions( self, reactions_list: list[str] ) -> None:
        """
        Remove a list of reactions from the model.

        Parameters
        ----------
        reactions_list : list[str]
            List of reaction identifiers to remove.
        """
        for r_id in reactions_list:
            self.remove_reaction(r_id)
    
    def rename_metabolite( self, previous_id: str, new_id: str ) -> None:
        """
        Rename a metabolite in the model.

        Parameters
        ----------
        previous_id : str
            Previous identifier of the metabolite.
        new_id : str
            New identifier of the metabolite.
        """
        assert previous_id in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{previous_id}</code> does not exist.")
        assert new_id not in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{new_id}</code> already exists.")
        #~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Manage reactions   #
        #~~~~~~~~~~~~~~~~~~~~~~~#
        reactions = self.which_reaction(previous_id)
        for r_id in reactions:
            self.reactions[r_id].rename_metabolite(previous_id, new_id)
        if previous_id in self.FBA_biomass_reaction.metabolites:
            self.FBA_biomass_reaction.rename_metabolite(previous_id, new_id)
        #~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Manage metabolites #
        #~~~~~~~~~~~~~~~~~~~~~~~#
        self.metabolites[new_id]    = self.metabolites.pop(previous_id)
        self.metabolites[new_id].id = new_id

    def rename_reaction( self, previous_id: str, new_id: str ) -> None:
        """
        Rename a reaction in the model.

        Parameters
        ----------
        previous_id : str
            Previous identifier of the reaction.
        new_id : str
            New identifier of the reaction.
        """
        assert previous_id in self.reactions, throw_message(MessageType.ERROR, f"Reaction <code>{previous_id}</code> does not exist.")
        assert new_id not in self.reactions, throw_message(MessageType.ERROR, f"Reaction <code>{new_id}</code> already exists.")
        self.reactions[new_id]    = self.reactions.pop(previous_id)
        self.reactions[new_id].id = new_id
    
    def create_average_protein( self, protein_id: str, protein_name: str, proteins_list: list[str] ) -> None:
        """
        Create an average protein from a list of proteins.
        
        Parameters
        ----------
        protein_id : str
            Identifier of the average protein.
        protein_name : str
            Name of the average protein.
        proteins_list : list[str]
            List of protein identifiers to average.    
        """
        assert protein_id != "", throw_message(MessageType.ERROR, "Empty protein identifier.")
        assert protein_id not in self.proteins, throw_message(MessageType.ERROR, f"Protein <code>{protein_id}</code> already exists.")
        assert len(proteins_list) > 0, throw_message(MessageType.ERROR, "Empty list of proteins.")
        for p_id in proteins_list:
            assert p_id in self.proteins, throw_message(MessageType.ERROR, f"Protein <code>{p_id}</code> does not exist.")
        avg_protein      = Protein(id=protein_id, name=protein_name, sequence="", mass=0.0)
        avg_protein.mass = np.sum([self.proteins[p_id].mass for p_id in proteins_list])/len(proteins_list)
        self.add_protein(avg_protein)
        throw_message(MessageType.INFO, f"Created average protein <code>{protein_id}</code> ({round(avg_protein.mass,2)} Da).")
    
    def create_sum_protein( self, protein_id: str, protein_name: str, proteins_list: list[str] ) -> None:
        """
        Create a sum protein from a list of proteins.

        Parameters
        ----------
        protein_id : str
            Identifier of the sum protein.
        protein_name : str
            Name of the sum protein.
        proteins_list : list[str]
            List of protein identifiers to sum.
        """
        assert protein_id != "", throw_message(MessageType.ERROR, "Empty protein identifier.")
        assert protein_id not in self.proteins, throw_message(MessageType.ERROR, f"Protein <code>{protein_id}</code> already exists.")
        assert len(proteins_list) > 0, throw_message(MessageType.ERROR, "Empty list of proteins.")
        for p_id in proteins_list:
            assert p_id in self.proteins, throw_message(MessageType.ERROR, f"Protein <code>{p_id}</code> does not exist.")
        sum_protein      = Protein(id=protein_id, name=protein_name, sequence="", mass=0.0)
        sum_protein.mass = np.sum([self.proteins[p_id].mass for p_id in proteins_list])
        self.add_protein(sum_protein)
        throw_message(MessageType.INFO, f"Created sum protein <code>{protein_id}</code> ({round(sum_protein.mass,2)} Da).")
    
    def create_dummy_protein( self, protein_id: str, protein_name: str, protein_mass: float ) -> None:
        """
        Create a dummy protein.

        Parameters
        ----------
        protein_id : str
            Identifier of the dummy protein.
        protein_name : str
            Name of the dummy protein.
        protein_mass : float
            Mass of the dummy protein.
        """
        assert protein_id != "", throw_message(MessageType.ERROR, "Empty protein identifier.")
        assert protein_id not in self.proteins, throw_message(MessageType.ERROR, f"Protein <code>{protein_id}</code> already exists.")
        assert protein_mass > 0.0, throw_message(MessageType.ERROR, "Invalid protein mass.")
        dummy_protein = Protein(id=protein_id, name=protein_name, sequence="", mass=protein_mass)
        self.add_protein(dummy_protein)
        throw_message(MessageType.INFO, f"Created dummy protein <code>{protein_id}</code> ({round(dummy_protein.mass,2)} Da).")

    def create_average_metabolite( self, metabolite_id: str, metabolite_name: str, metabolites_list: list[str] ) -> None:
        """
        Create an average metabolite from a list of metabolites.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the average metabolite.
        metabolite_name : str
            Name of the average metabolite.
        metabolites_list : list[str]
            List of metabolite identifiers to average.
        """
        assert metabolite_id != "", throw_message(MessageType.ERROR, "Empty metabolite identifier.")
        assert metabolite_id not in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite_id}</code> already exists.")
        assert len(metabolites_list) > 0, throw_message(MessageType.ERROR, "Empty list of metabolites.")
        for m_id in metabolites_list:
            assert m_id in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{m_id}</code> does not exist.")
        avg_metabolite      = Metabolite(id=metabolite_id, name=metabolite_name, formula="", mass=0.0)
        avg_metabolite.mass = np.sum([self.metabolites[m_id].mass for m_id in metabolites_list])/len(metabolites_list)
        self.add_metabolite(avg_metabolite)
        throw_message(MessageType.INFO, f"Created average metabolite <code>{metabolite_id}</code> ({round(avg_metabolite.mass,2)} Da).")
    
    def create_sum_metabolite( self, metabolite_id: str, metabolite_name: str, metabolites_list: list[str] ) -> None:
        """
        Create a sum metabolite from a list of metabolites.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the sum metabolite.
        metabolite_name : str
            Name of the sum metabolite.
        metabolites_list : list[str]
            List of metabolite identifiers to sum.
        """
        assert metabolite_id != "", throw_message(MessageType.ERROR, "Empty metabolite identifier.")
        assert metabolite_id not in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite_id}</code> already exists.")
        assert len(metabolites_list) > 0, throw_message(MessageType.ERROR, "Empty list of metabolites.")
        for m_id in metabolites_list:
            assert m_id in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{m_id}</code> does not exist.")
        sum_metabolite      = Metabolite(id=metabolite_id, name=metabolite_name, formula="", mass=0.0)
        sum_metabolite.mass = np.sum([self.metabolites[m_id].mass for m_id in metabolites_list])
        self.add_metabolite(sum_metabolite)
        throw_message(MessageType.INFO, f"Created sum metabolite <code>{metabolite_id}</code> ({round(sum_metabolite.mass,2)} Da).")
    
    def create_dummy_metabolite( self, metabolite_id: str, metabolite_name: str, metabolite_mass: float ) -> None:
        """
        Create a dummy metabolite.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the dummy metabolite.
        metabolite_name : str
            Name of the dummy metabolite.
        metabolite_mass : float
            Mass of the dummy metabolite.
        """
        assert metabolite_id != "", throw_message(MessageType.ERROR, "Empty metabolite identifier.")
        assert metabolite_id not in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite_id}</code> already exists.")
        assert metabolite_mass > 0.0, throw_message(MessageType.ERROR, "Invalid metabolite mass.")
        dummy_metabolite = Metabolite(id=metabolite_id, name=metabolite_name, formula="", mass=metabolite_mass)
        self.add_metabolite(dummy_metabolite)
        throw_message(MessageType.INFO, f"Created dummy metabolite <code>{metabolite_id}</code> ({round(dummy_metabolite.mass,2)} Da).")
    
    def enforce_kcat_irreversibility( self ) -> None:
        """
        Enforce the irreversibility of all reactions at the level of kcat
        values.
        """
        for reaction in self.reactions.values():
            reaction.enforce_kcat_irreversibility()

    def enforce_km_irreversibility( self ) -> None:
        """
        Enforce the irreversibility of all reactions at the level of KM values.
        """
        for reaction in self.reactions.values():
            reaction.enforce_km_irreversibility()
    
    def add_activation_constant( self, metabolite_id: str, reaction_id: str, value: float ) -> None:
        """
        Add an activation constant to a reaction.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the metabolite.
        reaction_id : str
            Identifier of the reaction.
        value : float
            Activation constant value.
        """
        assert self.GBA_is_built, throw_message(MessageType.ERROR, "GBA converted model is not built.")
        assert self.GBA_KA is not None, throw_message(MessageType.ERROR, "Activation constant matrix is not initialized.")
        assert self.GBA_KA.shape == (len(self.metabolites), len(self.reactions)), throw_message(MessageType.ERROR, "Invalid activation constant matrix shape.")
        assert metabolite_id in self.GBA_row_indices, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite_id}</code> is not listed in the GBA converted model.")
        assert reaction_id in self.GBA_col_indices, throw_message(MessageType.ERROR, f"Reaction <code>{reaction_id}</code> is not listed in the GBA converted model.")
        assert metabolite_id in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite_id}</code> does not exist.")
        assert reaction_id in self.reactions, throw_message(MessageType.ERROR, f"Reaction <code>{reaction_id}</code> does not exist.")
        assert value > 0.0, throw_message(MessageType.ERROR, f"The activation constant value must be positive (<code>{metabolite_id}</code>, <code>{reaction_id}</code>).")
        m_index                       = self.GBA_row_indices[metabolite_id]
        r_index                       = self.GBA_col_indices[reaction_id]
        self.GBA_KA[m_index, r_index] = value

    def add_inhibition_constant( self, metabolite_id: str, reaction_id: str, value: float ) -> None:
        """
        Add an inhibition constant to a reaction.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the metabolite.
        reaction_id : str
            Identifier of the reaction.
        value : float
            Inhibition constant value.
        """
        assert self.GBA_is_built, throw_message(MessageType.ERROR, "GBA converted model is not built.")
        assert self.GBA_KI is not None, throw_message(MessageType.ERROR, "Inhibition constant matrix is not initialized.")
        assert self.GBA_KI.shape == (len(self.metabolites), len(self.reactions)), throw_message(MessageType.ERROR, "Invalid inhibition constant matrix shape.")
        assert metabolite_id in self.GBA_row_indices, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite_id}</code> is not listed in the GBA converted model.")
        assert reaction_id in self.GBA_col_indices, throw_message(MessageType.ERROR, f"Reaction <code>{reaction_id}</code> is not listed in the GBA converted model.")
        assert metabolite_id in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite_id}</code> does not exist.")
        assert reaction_id in self.reactions, throw_message(MessageType.ERROR, f"Reaction <code>{reaction_id}</code> does not exist.")
        assert value > 0.0, throw_message(MessageType.ERROR, f"The inhibition constant value must be positive (<code>{metabolite_id}</code>, <code>{reaction_id}</code>).")
        m_index                       = self.GBA_row_indices[metabolite_id]
        r_index                       = self.GBA_col_indices[reaction_id]
        self.GBA_KI[m_index, r_index] = value

    def set_rho( self, rho: float ) -> None:
        """
        Set the total density of the cell in the GBA converted model.

        Parameters
        ----------
        rho : float
            Total density of the cell (g/L).
        """
        assert self.GBA_is_built, throw_message(MessageType.ERROR, "GBA converted model is not built.")
        assert rho > 0.0, throw_message(MessageType.ERROR, "The total density must be positive.")
        self.GBA_rho = rho
    
    def clear_conditions( self ) -> None:
        """
        Clear all external conditions from the GBA converted model.
        """
        self.GBA_conditions = {}
    
    def add_condition( self, condition_id: str, default_concentration: Optional[float] = 1.0, metabolites: Optional[dict[str, float]] = None ) -> None:
        """
        Add an external condition to the GBA converted model.

        Parameters
        ----------
        condition_id : str
            Identifier of the condition.
        default_concentration : float
            Default concentration of metabolites (g/L).
        metabolites : dict[str, float]
            Dictionary of metabolite concentrations (g/L).
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Assertions                             #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        assert self.GBA_is_built, throw_message(MessageType.ERROR, "GBA converted model is not built.")
        assert condition_id not in self.GBA_conditions, throw_message(MessageType.ERROR, f"Condition <code>{condition_id}</code> already exists.")
        assert self.GBA_rho > 0.0, throw_message(MessageType.ERROR, "The total density must be positive.")
        assert default_concentration >= 0.0, throw_message(MessageType.ERROR, "The default concentration must be positive.")
        if metabolites is not None:
            for m_id, concentration in metabolites.items():
                assert m_id in self.metabolites, throw_message(MessageType.ERROR, f"Metabolite <code>{m_id}</code> does not exist.")
                assert concentration >= 0.0, throw_message(MessageType.ERROR, f"The concentration of metabolite <code>{m_id}</code> must be positive.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Set the condition                      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.GBA_conditions[condition_id] = {"rho": self.GBA_rho}
        self.GBA_conditions[condition_id].update({m_id: default_concentration for m_id in self.metabolites if self.metabolites[m_id].species_location == SpeciesLocation.EXTERNAL})
        if metabolites is not None:
            for m_id, concentration in metabolites.items():
                self.GBA_conditions[condition_id][m_id] = concentration

    def clear_constant_rhs( self ) -> None:
        """
        Clear all constant RHS terms from the GBA converted model.
        """
        self.GBA_constant_rhs = {}
    
    def add_constant_rhs( self, metabolite_id: str, value: float ) -> None:
        """
        Make a GBA metabolite constant in the RHS term for the initial solution.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the metabolite.
        value : float
            Flux value.
        """
        assert metabolite_id not in self.GBA_constant_rhs, throw_message(MessageType.ERROR, f"Metabolite <code>{metabolite_id}</code> is already constant.")
        assert value > 0.0, throw_message(MessageType.ERROR, f"The constant value must be positive (<code>{metabolite_id}</code>).")
        self.GBA_constant_rhs[metabolite_id] = value

    def clear_constant_reactions( self ) -> None:
        """
        Clear all constant reactions from the GBA converted model.
        """
        self.GBA_constant_reactions = {}

    def add_constant_reaction( self, reaction_id: str, value: float ) -> None:
        """
        Make a GBA reaction constant to a given flux value.

        Parameters
        ----------
        reaction_id : str
            Identifier of the reaction.
        value : float
            Flux value.
        """
        assert reaction_id not in self.GBA_constant_reactions, throw_message(MessageType.ERROR, f"Reaction <code>{reaction_id}</code> is already constant.")
        self.GBA_constant_reactions[reaction_id] = value

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 3) Model consistency tests  #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    def check_model( self, test_structure: Optional[bool] = False ) -> None:
        """
        Detect missing molecular masses, kinetic parameters and connectivity
        issues in the model.

        Parameters
        ----------
        test_structure : bool
            Test the structure of the metabolic network.
        """
        self.detect_missing_mass(True)
        self.detect_missing_kinetic_parameters(True)
        self.detect_missing_connections(True)
        if test_structure:
            self.detect_unproduced_metabolites(True)
            self.detect_infeasible_loops(True)
            self.detect_dead_end_metabolites(True)
            self.detect_isolated_transporters(True)
    
    def detect_missing_mass( self, verbose: Optional[bool] = False ) -> dict[str, list[str]]:
        """
        Detect objects with missing molecular masses.

        Parameters
        ----------
        verbose : bool
            Display messages if True.
        """
        v_proteins    = [x.id for x in self.proteins.values() if x.has_missing_mass()]
        v_metabolites = [x.id for x in self.metabolites.values() if x.has_missing_mass()]
        v_enzymes     = [x.id for x in self.reactions.values() if x.has_missing_mass()]
        if verbose:
            if len(v_proteins) > 0:
                perc = len(v_proteins)/len(self.proteins)*100
                throw_message(MessageType.WARNING, f"{perc:.2f}% of proteins with missing mass.")
            if len(v_metabolites) > 0:
                perc = len(v_metabolites)/len(self.metabolites)*100
                throw_message(MessageType.WARNING, f"{perc:.2f}% of metabolites with missing mass.")
            if len(v_enzymes) > 0:
                perc = len(v_enzymes)/len(self.reactions)*100
                throw_message(MessageType.WARNING, f"{perc:.2f}% of enzymes with missing mass.")
            if len(v_proteins)==0 and len(v_metabolites)==0 and len(v_enzymes)==0:
                throw_message(MessageType.INFO, "No missing mass in the model.")
        return {"proteins": v_proteins, "metabolites": v_metabolites, "enzymes": v_enzymes}
    
    def detect_missing_kinetic_parameters( self, verbose: Optional[bool] = False ) -> dict[str, list[str]]:
        """
        Detect reactions with missing kinetic parameters.

        Parameters
        ----------
        verbose : bool
            Display messages if True.
        """
        missing_kcat = [r.id for r in self.reactions.values() if r.has_missing_kcat_value()]
        missing_km   = [r.id for r in self.reactions.values() if r.has_missing_km_value()]
        if verbose:
            if len(missing_kcat) > 0:
                transporter_count = len([r_id for r_id in missing_kcat if self.reactions[r_id].reaction_type == ReactionType.TRANSPORT])
                spontaneous_count = len([r_id for r_id in missing_kcat if self.reactions[r_id].reaction_type == ReactionType.SPONTANEOUS])
                metabolic_count   = len([r_id for r_id in missing_kcat if self.reactions[r_id].reaction_type == ReactionType.METABOLIC])
                perc              = len(missing_kcat)/len(self.reactions)*100
                transporter_perc  = transporter_count/len([r.id for r in self.reactions.values() if r.reaction_type == ReactionType.TRANSPORT])*100
                spontaneous_perc  = spontaneous_count/len([r.id for r in self.reactions.values() if r.reaction_type == ReactionType.SPONTANEOUS])*100
                metabolic_perc    = metabolic_count/len([r.id for r in self.reactions.values() if r.reaction_type == ReactionType.METABOLIC])*100
                throw_message(MessageType.WARNING, f"{perc:.2f}% of reactions with missing kcat values ({transporter_perc:.2f}% transporters, {spontaneous_perc:.2f}% spontaneous, {metabolic_perc:.2f}% metabolic).")
            if len(missing_km) > 0:
                perc = len(missing_km)/len(self.reactions)*100
                throw_message(MessageType.WARNING, f"{perc:.2f}% of reactions with missing KM values.")
            if len(missing_kcat)==0 and len(missing_km)==0:
                throw_message(MessageType.INFO, "No missing kinetic parameters in the model.")
        return {"kcat": missing_kcat, "km": missing_km}
    
    def detect_missing_connections( self, verbose: Optional[bool] = False ) -> dict[str, list[str]]:
        """
        Detect connectivity issues in the model.

        Parameters
        ----------
        verbose : bool
            Display messages if True.
        """
        p_to_r_vec = m_to_r_vec = r_to_p_vec = r_to_m_vec = []
        connectivity_error = False
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize mappings                  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        protein_to_reaction_map    = {p_id: [] for p_id in self.proteins}
        metabolite_to_reaction_map = {m_id: [] for m_id in self.metabolites}
        reaction_to_protein_map    = {r_id: [] for r_id in self.reactions}
        reaction_to_metabolite_map = {r_id: [] for r_id in self.reactions}
        for r in self.reactions.values():
            for p_id in r.proteins:
                protein_to_reaction_map[p_id].append(r.id)
                reaction_to_protein_map[r.id].append(p_id)
            for m_id in r.metabolites:
                metabolite_to_reaction_map[m_id].append(r.id)
                reaction_to_metabolite_map[r.id].append(m_id)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Check the consistency of the mapping #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for p_id in protein_to_reaction_map:
            if len(protein_to_reaction_map[p_id]) == 0:
                p_to_r_vec.append(p_id)
                # connectivity_error = True
                #if verbose:
                #    throw_message(MessageType.WARNING, f"Protein <code>{p_id}</code> has no associated reaction.")
        for m_id in metabolite_to_reaction_map:
            if len(metabolite_to_reaction_map[m_id]) == 0:
                m_to_r_vec.append(m_id)
                connectivity_error = True
                if verbose:
                    throw_message(MessageType.WARNING, f"Metabolite <code>{m_id}</code> has no associated reaction.")
        for r_id in reaction_to_protein_map:
            if len(reaction_to_protein_map[r_id]) == 0:
                r_to_p_vec.append(r_id)
                if verbose:
                    connectivity_error = True
                    throw_message(MessageType.WARNING, f"Reaction <code>{r_id}</code> has no associated protein.")
        for r_id in reaction_to_metabolite_map:
            if len(reaction_to_metabolite_map[r_id]) == 0:
                r_to_m_vec.append(r_id)
                if verbose:
                    connectivity_error = True
                    throw_message(MessageType.WARNING, f"Reaction <code>{r_id}</code> has no associated metabolite.")
        if not connectivity_error and verbose:
            throw_message(MessageType.INFO, "No connectivity issues in the model.")
        return {"protein_to_reaction":    p_to_r_vec,
                "metabolite_to_reaction": m_to_r_vec,
                "reaction_to_protein":    r_to_p_vec,
                "reaction_to_metabolite": r_to_m_vec}
    
    def detect_unproduced_metabolites( self, verbose: Optional[bool] = False ) -> list[str]:
        """
        Detect unproduced metabolites in the model.

        Parameters
        ----------
        verbose : bool
            Display messages if True.
        """
        met_to_met_connectivity = {m_id: {"previous": [], "next": []} for m_id in self.metabolites}
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Build the connectivity dictionaries #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for reaction in self.reactions.values():
            ### 1.1) If the reaction is forward irreversible ###
            if ReactionDirection.FORWARD:
                ### Classify reactants
                for m_id in reaction.reactants:
                    current_list = met_to_met_connectivity[m_id]["next"].copy()
                    met_to_met_connectivity[m_id]["next"] += [m_id for m_id in reaction.products if m_id not in current_list]
                ### Classify products
                for m_id in reaction.products:
                    current_list = met_to_met_connectivity[m_id]["previous"].copy()
                    met_to_met_connectivity[m_id]["previous"] += [m_id for m_id in reaction.reactants if m_id not in current_list]
            ### 1.2) If the reaction is backward irreversible ###
            elif ReactionDirection.BACKWARD:
                ### Classify reactants
                for m_id in reaction.products:
                    current_list = met_to_met_connectivity[m_id]["next"].copy()
                    met_to_met_connectivity[m_id]["next"] += [m_id for m_id in reaction.reactants if m_id not in current_list]
                ### Classify products
                for m_id in reaction.reactants:
                    current_list = met_to_met_connectivity[m_id]["previous"].copy()
                    met_to_met_connectivity[m_id]["previous"] += [m_id for m_id in reaction.products if m_id not in current_list]
            ### 1.3) If the reaction is reversible ###
            elif ReactionDirection.REVERSIBLE:
                ### Classify reactants
                for m_id in reaction.reactants:
                    current_list = met_to_met_connectivity[m_id]["previous"].copy()
                    met_to_met_connectivity[m_id]["previous"] += [m_id for m_id in reaction.products if m_id not in current_list]
                    current_list = met_to_met_connectivity[m_id]["next"].copy()
                    met_to_met_connectivity[m_id]["next"] += [m_id for m_id in reaction.products if m_id not in current_list]
                ### Classify products
                for m_id in reaction.products:
                    current_list = met_to_met_connectivity[m_id]["previous"].copy()
                    met_to_met_connectivity[m_id]["previous"] += [m_id for m_id in reaction.reactants if m_id not in current_list]
                    current_list = met_to_met_connectivity[m_id]["next"].copy()
                    met_to_met_connectivity[m_id]["next"] += [m_id for m_id in reaction.reactants if m_id not in current_list]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Detect unproduced metabolites       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        not_produced = [m_id for m_id in met_to_met_connectivity if len(met_to_met_connectivity[m_id]["previous"]) == 0 and self.metabolites[m_id].species_location == SpeciesLocation.INTERNAL]
        if verbose:
            if len(not_produced) == 0:
                throw_message(MessageType.INFO, "No unproduced metabolites in the model.")
            for m_id in not_produced:
                throw_message(MessageType.WARNING, f"Metabolite <code>{m_id}</code> is not produced by any reaction.")
        return not_produced
    
    def detect_infeasible_loops( self, verbose: Optional[bool] = False ) -> list[list[str]]:
        """
        Detect infeasible loops breaking the mass balance in the model.

        Parameters
        ----------
        verbose : bool
            Display messages if True.
        """
        met_to_met_connectivity = {m_id: {"previous": [], "next": []} for m_id in self.metabolites}
        met_to_rea_connectivity = {m_id: {"reactant": [], "product": []} for m_id in self.metabolites}
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Build the connectivity dictionaries #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for reaction in self.reactions.values():
            ### 1.1) If the reaction is forward irreversible ###
            if ReactionDirection.FORWARD:
                ### Classify reactants
                for m_id in reaction.reactants:
                    current_list = met_to_met_connectivity[m_id]["next"].copy()
                    met_to_met_connectivity[m_id]["next"] += [m_id for m_id in reaction.products if m_id not in current_list]
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
                ### Classify products
                for m_id in reaction.products:
                    current_list = met_to_met_connectivity[m_id]["previous"].copy()
                    met_to_met_connectivity[m_id]["previous"] += [m_id for m_id in reaction.reactants if m_id not in current_list]
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
            ### 1.2) If the reaction is backward irreversible ###
            elif ReactionDirection.BACKWARD:
                ### Classify reactants
                for m_id in reaction.products:
                    current_list = met_to_met_connectivity[m_id]["next"].copy()
                    met_to_met_connectivity[m_id]["next"] += [m_id for m_id in reaction.reactants if m_id not in current_list]
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
                ### Classify products
                for m_id in reaction.reactants:
                    current_list = met_to_met_connectivity[m_id]["previous"].copy()
                    met_to_met_connectivity[m_id]["previous"] += [m_id for m_id in reaction.products if m_id not in current_list]
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
            ### 1.3) If the reaction is reversible ###
            elif ReactionDirection.REVERSIBLE:
                ### Classify reactants
                for m_id in reaction.reactants:
                    current_list = met_to_met_connectivity[m_id]["previous"].copy()
                    met_to_met_connectivity[m_id]["previous"] += [m_id for m_id in reaction.products if m_id not in current_list]
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
                    current_list = met_to_met_connectivity[m_id]["next"].copy()
                    met_to_met_connectivity[m_id]["next"] += [m_id for m_id in reaction.products if m_id not in current_list]
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
                ### Classify products
                for m_id in reaction.products:
                    current_list = met_to_met_connectivity[m_id]["previous"].copy()
                    met_to_met_connectivity[m_id]["previous"] += [m_id for m_id in reaction.reactants if m_id not in current_list]
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
                    current_list = met_to_met_connectivity[m_id]["next"].copy()
                    met_to_met_connectivity[m_id]["next"] += [m_id for m_id in reaction.reactants if m_id not in current_list]
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Detect infeasible loops             #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        in_a_loop = [m_id for m_id in met_to_met_connectivity if len(list(set(met_to_met_connectivity[m_id]["previous"]).intersection(met_to_met_connectivity[m_id]["next"]))) > 0 and self.metabolites[m_id].species_location == SpeciesLocation.INTERNAL]
        pairs     = []
        for m_id in in_a_loop:
            common_ids = list(set(met_to_met_connectivity[m_id]["previous"]).intersection(met_to_met_connectivity[m_id]["next"]))
            common_ids = [id for id in common_ids if self.metabolites[id].species_location == SpeciesLocation.INTERNAL]
            for c_id in common_ids:
                m_reactant = met_to_rea_connectivity[m_id]["reactant"]
                c_reactant = met_to_rea_connectivity[c_id]["reactant"]
                m_product  = met_to_rea_connectivity[m_id]["product"]
                c_product  = met_to_rea_connectivity[c_id]["product"]
                if m_reactant == c_product and m_product == c_reactant:
                    if [m_id, c_id] not in pairs and [c_id, m_id] not in pairs:
                        pairs.append([m_id, c_id])
        if verbose:
            if len(pairs) == 0:
                throw_message(MessageType.INFO, "No infeasible loops in the model.")
            for m_id, c_id in pairs:
                throw_message(MessageType.WARNING, f"Infeasible loop between <code>{m_id}</code> and <code>{c_id}</code>.")
        return pairs

    def detect_dead_end_metabolites( self, verbose: Optional[bool] = False ) -> list[str]:
        """
        Detect dead-end metabolites in the model.

        Parameters
        ----------
        verbose : bool
            Display messages if True.
        """
        met_to_rea_connectivity = {m_id: {"reactant": [], "product": []} for m_id in self.metabolites}
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Build the connectivity dictionary #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for reaction in self.reactions.values():
            ### 1.1) If the reaction is forward irreversible ###
            if ReactionDirection.FORWARD:
                ### Classify reactants
                for m_id in reaction.reactants:
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
                ### Classify products
                for m_id in reaction.products:
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
            ### 1.2) If the reaction is backward irreversible ###
            elif ReactionDirection.BACKWARD:
                ### Classify reactants
                for m_id in reaction.products:
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
                ### Classify products
                for m_id in reaction.reactants:
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
            ### 1.3) If the reaction is reversible ###
            elif ReactionDirection.REVERSIBLE:
                ### Classify reactants
                for m_id in reaction.reactants:
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
                ### Classify products
                for m_id in reaction.products:
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Detect dead-end metabolites       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        dead_end = []
        for m_id in met_to_rea_connectivity:
            if self.metabolites[m_id].species_location == SpeciesLocation.EXTERNAL:
                continue
            if len(met_to_rea_connectivity[m_id]["reactant"]) == 0 and len(met_to_rea_connectivity[m_id]["product"]) == 1:
                r_id = met_to_rea_connectivity[m_id]["product"][0]
                if self.reactions[r_id].reaction_type == ReactionType.METABOLIC:
                    dead_end.append(m_id)
                    if verbose:
                        throw_message(MessageType.WARNING, f"Metabolite <code>{m_id}</code> is a dead-end metabolite.")
        if verbose and len(dead_end) == 0:
            throw_message(MessageType.INFO, "No dead-end metabolites in the model.")
        return dead_end

    def detect_isolated_transporters( self, verbose: Optional[bool] = False ) -> list[str]:
        """
        Detect isolated transporters in the model (imported but not used).

        Parameters
        ----------
        verbose : bool
            Display messages if True.
        """
        met_to_rea_connectivity = {m_id: {"reactant": [], "product": []} for m_id in self.metabolites}
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Build the connectivity dictionary #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for reaction in self.reactions.values():
            ### 1.1) If the reaction is forward irreversible ###
            if ReactionDirection.FORWARD:
                ### Classify reactants
                for m_id in reaction.reactants:
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
                ### Classify products
                for m_id in reaction.products:
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
            ### 1.2) If the reaction is backward irreversible ###
            elif ReactionDirection.BACKWARD:
                ### Classify reactants
                for m_id in reaction.products:
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
                ### Classify products
                for m_id in reaction.reactants:
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
            ### 1.3) If the reaction is reversible ###
            elif ReactionDirection.REVERSIBLE:
                ### Classify reactants
                for m_id in reaction.reactants:
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
                ### Classify products
                for m_id in reaction.products:
                    met_to_rea_connectivity[m_id]["product"].append(reaction.id)
                    met_to_rea_connectivity[m_id]["reactant"].append(reaction.id)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Detect isolated metabolites       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        isolated = []
        for m_id in met_to_rea_connectivity:
            if self.metabolites[m_id].species_location == SpeciesLocation.EXTERNAL:
                continue
            if len(met_to_rea_connectivity[m_id]["reactant"]) == 0 and len(met_to_rea_connectivity[m_id]["product"]) == 1:
                r_id = met_to_rea_connectivity[m_id]["product"][0]
                if self.reactions[r_id].reaction_type in [ReactionType.TRANSPORT, ReactionType.SPONTANEOUS]:
                    isolated.append(m_id)
                    if verbose:
                        throw_message(MessageType.WARNING, f"Metabolite <code>{m_id}</code> is solely imported.")
        if verbose and len(isolated) == 0:
            throw_message(MessageType.INFO, "No isolated transporters in the model.")
        return isolated

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 4) FBA model reconstruction #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def create_FBA_biomass_reaction( self, metabolites: dict[str, float] ) -> None:
        """
        Create FBA biomass function.

        Parameters
        ----------
        metabolites : dict[str, float]
            Dictionary of metabolites and their coefficients.
        """
        self.FBA_biomass_reaction = Reaction(id="BIOMASS", name="Biomass function",
                                             reaction_type=ReactionType.METABOLIC,
                                             lb=0.0, ub=1000.0,
                                             GPR=ReactionGPR.NONE)
        self.FBA_biomass_reaction.set_builder(self)
        for m_id, coeff in metabolites.items():
            if m_id in self.metabolites:
                self.FBA_biomass_reaction.add_metabolites({m_id: coeff})
        self.FBA_biomass_reaction.add_proteins({"average_protein": 1.0})
        self.FBA_biomass_reaction.define_direction()
        self.FBA_biomass_reaction.define_expression()
        self.FBA_biomass_reaction.calculate_enzyme_mass()
    
    def build_FBA_indices( self ) -> None:
        """
        Build metabolite and reaction indices for the FBA model.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Build metabolite indices #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.FBA_row_indices.clear()
        self.FBA_external_row_indices.clear()
        self.FBA_internal_row_indices.clear()
        index = 0
        for m in self.metabolites.values():
            if m.species_location == SpeciesLocation.EXTERNAL:
                self.FBA_row_indices[m.id]          = index
                self.FBA_external_row_indices[m.id] = index
                index += 1
        for m in self.metabolites.values():
            if m.species_location == SpeciesLocation.INTERNAL:
                self.FBA_row_indices[m.id]          = index
                self.FBA_internal_row_indices[m.id] = index
                index += 1
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Build reaction indices   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.FBA_col_indices.clear()
        index = 0
        for r in self.reactions.values():
            if r.reaction_type == ReactionType.TRANSPORT:
                self.FBA_col_indices[r.id] = index
                index += 1
        for r in self.reactions.values():
            if r.reaction_type == ReactionType.SPONTANEOUS:
                self.FBA_col_indices[r.id] = index
                index += 1
        for r in self.reactions.values():
            if r.reaction_type == ReactionType.METABOLIC:
                self.FBA_col_indices[r.id] = index
                index += 1
    
    def build_FBA_stoichiometric_matrix( self ) -> None:
        """
        Build the FBA stoichiometric matrix.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Build the complete stoichiometric matrix #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.FBA_S = np.zeros((len(self.FBA_row_indices), len(self.FBA_col_indices)))
        for r in self.reactions.values():
            for m_id, stoich in r.metabolites.items():
                m_index = self.FBA_row_indices[m_id]
                r_index = self.FBA_col_indices[r.id]
                self.FBA_S[m_index, r_index] = stoich
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Build the internal stoichiometric matrix #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.FBA_intS = np.zeros((len(self.FBA_internal_row_indices), len(self.FBA_col_indices)))
        for i in range(len(self.FBA_internal_row_indices)):
            m_id = list(self.FBA_internal_row_indices.keys())[i]
            for j in range(len(self.FBA_col_indices)):
                self.FBA_intS[i, j] = self.FBA_S[self.FBA_internal_row_indices[m_id], j]
    
    def compute_stoichiometric_matrix_metrics( self ) -> None:
        """
        Compute the mass fraction matrix metrics:
        - Rank of the mass fraction matrix
        - List of dependent reactions
        """
        self.FBA_column_rank = np.linalg.matrix_rank(self.FBA_intS)
        if self.FBA_column_rank == self.FBA_intS.shape[1]:
            self.FBA_is_full_column_rank = True
        lambdas, V                   =  np.linalg.qr(self.FBA_intS)
        linearly_independent_indices = np.abs(np.diag(V)) >= 1e-10
        indices                      = np.where(linearly_independent_indices == False)[0]
        r_ids                        = list(self.FBA_col_indices.keys())
        self.FBA_dependent_reactions = [r_id for i, r_id in enumerate(r_ids) if i in indices]
    
    def build_FBA_model( self, enforced_reactions: Optional[dict[str, float]] = None ) -> None:
        """
        Build the FBA model.

        Parameters
        ----------
        enforced_reactions : dict[str, float]
            Dictionary of enforced reactions.
        """
        if enforced_reactions is not None:
            for r_id, value in enforced_reactions.items():
                if value > 0.0:
                    self.reactions[r_id].lb = value
                else:
                    self.reactions[r_id].ub = value
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Create the cobra model            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.FBA_model = cobra.Model(self.name)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Add metabolites                   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        cobra_metabolites = {}
        for m_item in self.metabolites.items():
            if m_item[1].species_location == SpeciesLocation.EXTERNAL:
                m = cobra.Metabolite(m_item[0], compartment="e")
                self.FBA_model.add_metabolites([m])
                cobra_metabolites[m_item[0]] = m
            elif m_item[1].species_location == SpeciesLocation.INTERNAL:
                m = cobra.Metabolite(m_item[0], compartment="c")
                self.FBA_model.add_metabolites([m])
                cobra_metabolites[m_item[0]] = m
            else:
                throw_message(MessageType.ERROR, "Unknown species location for metabolite <code>{m_id}</code>.")
                sys.exit(1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Add reactions                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for r_item in self.reactions.items():
            r             = cobra.Reaction(r_item[0])
            r.lower_bound = r_item[1].lb
            r.upper_bound = r_item[1].ub
            for m_item in r_item[1].metabolites.items():
                r.add_metabolites({cobra_metabolites[m_item[0]]: m_item[1]})
            self.FBA_model.add_reactions([r])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Add the biomass reaction          #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        r             = cobra.Reaction(self.FBA_biomass_reaction.id)
        r.lower_bound = self.FBA_biomass_reaction.lb
        r.upper_bound = self.FBA_biomass_reaction.ub
        for m_item in self.FBA_biomass_reaction.metabolites.items():
            if m_item[0] in self.metabolites:
                r.add_metabolites({cobra_metabolites[m_item[0]]: m_item[1]})
        self.FBA_model.add_reactions([r])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Add the FBA wrapper reactions     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for m_item in self.metabolites.items():
            if m_item[1].species_location == SpeciesLocation.EXTERNAL:
                self.FBA_model.add_boundary(cobra_metabolites[m_item[0]], type="exchange")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Solve the FBA models              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.FBA_model.solver    = "gurobi"
        self.FBA_model.objective = "BIOMASS"
        self.FBA_solution        = self.FBA_model.optimize()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 7) Build the stoichiometric matrix   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.build_FBA_indices()
        self.build_FBA_stoichiometric_matrix()
        self.compute_stoichiometric_matrix_metrics()
        self.FBA_is_built = True
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 8) Collect active/inactive reactions #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        sol = self.FBA_solution.fluxes.to_dict()
        self.inactive_reactions = [r_id for r_id in sol if sol[r_id] == 0.0 and not r_id.startswith("EX_")]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 5) Model reconstruction     #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def adjust_masses( self, metabolites: dict[str, float] ) -> None:
        """
        Adjust the masses of a list of metabolites.

        Parameters
        ----------
        metabolites : dict[str, float]
            Dictionary of metabolites and their mass adjustments.
        """
        for m_id, mass_adjust in metabolites.items():
            if m_id in self.metabolites:
                self.metabolites[m_id].mass += mass_adjust
    
    def check_mass_balance( self, verbose: Optional[bool] = False ) -> None:
        """
        Check the mass balance of the model.

        Parameters
        ----------
        verbose : bool
            Display messages if True.
        """
        is_balanced = True
        for reaction in self.reactions.values():
            if not reaction.check_mass_balance(verbose):
                is_balanced = False
        if is_balanced:
            throw_message(MessageType.INFO, f"Model build <code>{self.name}</code> is mass balanced.")

    def check_mass_normalization( self, verbose: Optional[bool] = False ) -> None:
        """
        Check the mass normalization of the model.

        Parameters
        ----------
        verbose : bool
            Display messages if True.
        """
        is_normalized = True
        for reaction in self.reactions.values():
            if not reaction.check_mass_normalization(verbose):
                is_normalized = False
        if is_normalized:
            throw_message(MessageType.INFO, f"Model build <code>{self.name}</code> is mass normalized")
    
    def check_ribosomal_reaction_consistency( self ) -> None:
        """
        Check the ribosomal reaction consistency of the model.
        """
        assert "Ribosome" in self.reactions, throw_message(MessageType.ERROR, "Ribosomal reaction not found in the model.")
        assert "Protein" in self.metabolites, throw_message(MessageType.ERROR, "Protein metabolite not found in the reaction.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Check the ribosomal reaction itself #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        r_ribosome = self.reactions["Ribosome"]
        assert "Protein" in r_ribosome.products, throw_message(MessageType.ERROR, "Protein metabolite not found in the ribosomal reaction.")
        assert r_ribosome.kcat[ReactionDirection.BACKWARD] == 0.0, throw_message(MessageType.ERROR, "Ribosomal reaction should be forward irreversible.")
        assert r_ribosome.direction == ReactionDirection.FORWARD, throw_message(MessageType.ERROR, "Ribosomal reaction should be forward irreversible.")
        assert r_ribosome.lb == 0.0, throw_message(MessageType.ERROR, "Ribosomal reaction should have a lower bound of 0.")
        assert r_ribosome.ub > 0.0, throw_message(MessageType.ERROR, "Ribosomal reaction should have a positive upper bound.")
        if len(r_ribosome.products) == 1:
            assert r_ribosome.GBA_metabolites["Protein"] == 1.0, throw_message(MessageType.ERROR, "Protein coefficient should be 1 in the ribosomal reaction.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Check other reactions               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for r in self.reactions.values():
            if not r.id == "Ribosome":
                assert "Protein" not in r.GBA_metabolites, throw_message(MessageType.ERROR, f"Protein metabolite found in reaction <code>{r.id}</code>. Protein metabolite should only be a ribosomal product.")

    def check_conversion( self, verbose: Optional[bool] = False ) -> None:
        """
        Check the conversion of the model to GBA format.

        Parameters
        ----------
        verbose : bool
            Display messages if True.
        """
        is_converted = True
        for r in self.reactions.values():
            if not r.check_conversion(verbose):
                is_converted = False
        if is_converted:
            if verbose:
                throw_message(MessageType.INFO, f"Model build <code>{self.name}</code> is converted to GBA format.")
            return True
        return False
    
    def convert( self, ribosome_byproducts: Optional[bool] = False,
                 ribosome_mass_kcat: Optional[float] = 4.55,
                 ribosome_mass_km: Optional[float] = 8.3 ) -> None:
        """
        Convert the model to GBA format.

        Parameters
        ----------
        ribosome_byproducts : bool
            Add ribosome byproducts if True.
        ribosome_mass_kcat : float
            Value of the mass normalized kcat value.
        ribosome_mass_km : float
            Value of the mass normalized KM value.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Edit the ribosomal reaction if needed #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if not ribosome_byproducts:
            products = [p for p in self.reactions["Ribosome"].products if p != "Protein"]
            for p in products:
                self.reactions["Ribosome"].remove_metabolite(p)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Convert every reactions               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for r in self.reactions.values():
            r.convert()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Check the ribosomal reaction          #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.check_ribosomal_reaction_consistency()            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Set up ribosomal kinetic parameters   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if ribosome_mass_kcat is not None:
            self.reactions["Ribosome"].GBA_kcat[ReactionDirection.FORWARD] = ribosome_mass_kcat
        if ribosome_mass_km is not None:
            for m_id in self.reactions["Ribosome"].reactants:
                self.reactions["Ribosome"].GBA_km[m_id] = ribosome_mass_km

    def reset_conversion( self ) -> None:
        """
        Reset the conversion of the model to GBA format.
        """
        for r in self.reactions.values():
            r.reset_conversion()
    
    def build_GBA_indices( self ) -> None:
        """
        Build metabolite and reaction indices for the GBA converted model
        (Protein metabolite is always the last metabolite,
         and Ribosome reaction is always the last reaction).
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Build metabolite indices #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.GBA_row_indices.clear()
        self.GBA_external_row_indices.clear()
        self.GBA_internal_row_indices.clear()
        index = 0
        for m in self.metabolites.values():
            if m.species_location == SpeciesLocation.EXTERNAL and m.id != "Protein":
                self.GBA_row_indices[m.id]          = index
                self.GBA_external_row_indices[m.id] = index
                index += 1
        for m in self.metabolites.values():
            if m.species_location == SpeciesLocation.INTERNAL and m.id != "Protein":
                self.GBA_row_indices[m.id]          = index
                self.GBA_internal_row_indices[m.id] = index
                index += 1
        self.GBA_row_indices["Protein"] = index
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Build reaction indices   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.GBA_col_indices.clear()
        index = 0
        for r in self.reactions.values():
            if r.reaction_type == ReactionType.TRANSPORT and r.id != "Ribosome":
                self.GBA_col_indices[r.id] = index
                index += 1
        for r in self.reactions.values():
            if r.reaction_type == ReactionType.SPONTANEOUS and r.id != "Ribosome":
                self.GBA_col_indices[r.id] = index
                index += 1
        for r in self.reactions.values():
            if r.reaction_type == ReactionType.METABOLIC and r.id != "Ribosome":
                self.GBA_col_indices[r.id] = index
                index += 1
        self.GBA_col_indices["Ribosome"] = index

    def build_GBA_mass_fraction_matrix( self ) -> None:
        """
        Build the model mass fraction matrix.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Build the complete mass fraction matrix #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.GBA_M = np.zeros((len(self.GBA_row_indices), len(self.GBA_col_indices)))
        for r in self.reactions.values():
            for m_id, stoich in r.GBA_metabolites.items():
                m_index                      = self.GBA_row_indices[m_id]
                r_index                      = self.GBA_col_indices[r.id]
                self.GBA_M[m_index, r_index] = stoich
                if stoich == -0.0:
                    self.GBA_M[m_index, r_index] = 0.0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Build the internal mass fraction matrix #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.GBA_intM = np.zeros((len(self.GBA_internal_row_indices), len(self.GBA_col_indices)))
        for i in range(len(self.GBA_internal_row_indices)):
            m_id = list(self.GBA_internal_row_indices.keys())[i]
            for j in range(len(self.GBA_col_indices)):
                self.GBA_intM[i, j] = self.GBA_M[self.GBA_internal_row_indices[m_id], j]

    def build_GBA_kcat_vectors( self ) -> None:
        """
        Build the GBA kcat vectors.
        """
        self.GBA_kcat_f = np.zeros(len(self.GBA_col_indices))
        self.GBA_kcat_b = np.zeros(len(self.GBA_col_indices))
        for r in self.reactions.values():
            r_index                  = self.GBA_col_indices[r.id]
            self.GBA_kcat_f[r_index] = r.GBA_kcat[ReactionDirection.FORWARD]
            self.GBA_kcat_b[r_index] = r.GBA_kcat[ReactionDirection.BACKWARD]

    def build_GBA_KM_matrices( self ) -> None:
        """
        Build the GBA KM matrices.
        """
        self.GBA_KM_f = np.zeros((len(self.GBA_row_indices), len(self.GBA_col_indices)))
        self.GBA_KM_b = np.zeros((len(self.GBA_row_indices), len(self.GBA_col_indices)))
        for r in self.reactions.values():
            for m_id, stoic in r.GBA_metabolites.items():
                if stoic < 0.0:
                    m_index                         = self.GBA_row_indices[m_id]
                    r_index                         = self.GBA_col_indices[r.id]
                    self.GBA_KM_f[m_index, r_index] = r.GBA_km[m_id]
                elif stoic > 0.0:
                    m_index                         = self.GBA_row_indices[m_id]
                    r_index                         = self.GBA_col_indices[r.id]
                    self.GBA_KM_b[m_index, r_index] = r.GBA_km[m_id]
    
    def build_GBA_KA_KI_matrices( self ) -> None:
        """
        Build the GBA activation and inhibition matrices.
        """
        self.GBA_KA = np.zeros((len(self.GBA_row_indices), len(self.GBA_col_indices)))
        self.GBA_KI = np.zeros((len(self.GBA_row_indices), len(self.GBA_col_indices)))

    def compile_protein_contributions( self ) -> None:
        """
        Compile the protein contributions from the reactions.
        """
        self.GBA_protein_contributions.clear()
        for r in self.reactions.values():
            if not r.protein_contributions is None:
                self.GBA_protein_contributions[r.id] = r.protein_contributions

    def compute_mass_fraction_matrix_metrics( self ) -> None:
        """
        Compute the mass fraction matrix metrics:
        - Rank of the mass fraction matrix
        - List of dependent reactions
        """
        self.GBA_column_rank = np.linalg.matrix_rank(self.GBA_intM)
        if self.GBA_column_rank == self.GBA_intM.shape[1]:
            self.GBA_is_full_column_rank = True
        lambdas, V                   =  np.linalg.qr(self.GBA_intM)
        linearly_independent_indices = np.abs(np.diag(V)) >= 1e-10
        indices                      = np.where(linearly_independent_indices == False)[0]
        r_ids                        = list(self.GBA_col_indices.keys())
        self.GBA_dependent_reactions = [r_id for i, r_id in enumerate(r_ids) if i in indices]
    
    def build_GBA_model( self ) -> None:
        """
        Build the GBA converted model.
        """
        if not self.check_conversion():
            self.convert()
        self.build_GBA_indices()
        self.build_GBA_mass_fraction_matrix()
        self.build_GBA_kcat_vectors()
        self.build_GBA_KM_matrices()
        self.build_GBA_KA_KI_matrices()
        self.compile_protein_contributions()
        self.compute_mass_fraction_matrix_metrics()
        self.GBA_is_built = True

    def convert_GBA_reaction_to_forward_irreversible( self, reaction_id: str, direction: ReactionDirection ) -> None:
        """
        Convert the GBA reaction to a forward irreversible reaction.

        Parameters
        ----------
        reaction_id : str
            Identifier of the reaction.
        direction : ReactionDirection
            Wanted direction of the reaction.
        """
        assert self.check_conversion(), throw_message(MessageType.ERROR, "The model is not converted to GBA units. Convert the model before building GBA variables.")
        assert self.GBA_is_built, throw_message(MessageType.ERROR, f"The GBA converted model <code>{self.name}</code> is not built")
        assert reaction_id in self.GBA_col_indices, throw_message(MessageType.ERROR, f"Reaction <code>{reaction_id}</code> does not exist")
        assert direction != ReactionDirection.REVERSIBLE, throw_message(MessageType.ERROR, "The wanted direction should be irreversible")
        j = self.GBA_col_indices[reaction_id]
        if direction == ReactionDirection.FORWARD:
            self.GBA_kcat_b[j] = 0.0
            self.GBA_KM_b[:,j] = 0.0
        elif direction == ReactionDirection.BACKWARD:
            self.GBA_M[:,j]    = -self.GBA_M[:,j]
            self.GBA_kcat_f[j] = self.GBA_kcat_b[j]
            self.GBA_kcat_b[j] = 0.0
            self.GBA_KM_f[:,j] = self.GBA_KM_b[:,j]
            self.GBA_KM_b[:,j] = 0.0

    def enforce_directionality( self, fluxes: dict[str, float] ) -> None:
        """
        Enforce the directionality of the GBA reactions based on a list of
        fluxes.

        Parameters
        ----------
        fluxes : dict[str, float]
            Dictionary of reaction identifiers and their flux values.
        """
        for item in fluxes.items():
            r_id  = item[0]
            f_val = item[1]
            if r_id in self.reactions and f_val >= 0.0:
                self.convert_GBA_reaction_to_forward_irreversible(r_id, ReactionDirection.FORWARD)
            elif r_id in self.reactions and f_val < 0.0:
                self.convert_GBA_reaction_to_forward_irreversible(r_id, ReactionDirection.BACKWARD)
    
    def column_rank( self ) -> None:
        """
        Display the column rank of the GBA mass fraction matrix.
        """
        x1 = self.GBA_column_rank
        x2 = self.GBA_intM.shape[1]
        throw_message(MessageType.PLAIN, "internal M shape : "+str(self.GBA_intM.shape))
        if self.GBA_is_full_column_rank:
            throw_message(MessageType.PLAIN, "Internal M is full column rank")
        else:
            throw_message(MessageType.PLAIN, "Internal M column rank {}/{} (diff = {})".format(x1, x2, x2-x1))
            throw_message(MessageType.PLAIN, "Dependent reactions : "+str(self.GBA_dependent_reactions))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 6) Export functions         #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def export_to_csv( self, name: Optional[str] = "", path: Optional[str] = "." ) -> None:
        """
        Export the model to a folder in CSV format.

        Parameters
        ----------
        name : Optional[str], default=""
            Name of the model. If empty, the model name is used.
        path : Optional[str], default="."
            Path to the folder.
        """
        assert self.check_conversion(), throw_message(MessageType.ERROR, "The model is not converted to GBA units. Convert the model before building GBA variables.")
        assert self.GBA_is_built, throw_message(MessageType.ERROR, f"The model <code>{self.name}</code> is not built")
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Check the existence of the folder #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        model_path = path+"/"+(name if name != "" else self.name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            files = ["Info.csv",
                     "M.csv", "kcat.csv", "K.csv",
                     "KA.csv", "KI.csv",
                     "rho.csv", "conditions.csv",
                     "constant_reactions.csv", "constant_rhs.csv", 
                     "protein_contributions.csv"]
            for f in files:
                if os.path.exists(model_path+"/"+f):
                    os.system(f"rm {model_path}/{f}")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Write the information             #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(self.info) > 0:
            rows = []
            for key, value in self.info.items():
                rows.append([key, value if isinstance(value, str) else ""])
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        rows.append(["", subkey, subvalue])
            Info_df         = pd.DataFrame(rows)
            Info_df.columns = ["", "", ""]
            Info_df.to_csv(model_path+"/Info.csv", sep=";", index=False, header=False)
            del(Info_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Write the mass fraction matrix    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        M_df = pd.DataFrame(self.GBA_M, index=self.GBA_row_indices.keys(), columns=self.GBA_col_indices.keys())
        M_df.replace(-0.0, 0.0, inplace=True)
        M_df.to_csv(model_path+"/M.csv", sep=";")
        del(M_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Write the kcat vectors            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        kcat_df           = pd.DataFrame(self.GBA_kcat_f, index=self.GBA_col_indices.keys(), columns=["kcat_f"])
        kcat_df["kcat_b"] = self.GBA_kcat_b
        kcat_df           = kcat_df.transpose()
        kcat_df.replace(-0.0, 0.0, inplace=True)
        kcat_df.to_csv(model_path+"/kcat.csv", sep=";")
        del(kcat_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Write the K matrix                #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for i in self.GBA_row_indices.values():
            for j in self.GBA_col_indices.values():
                if self.GBA_KM_f[i, j] != 0.0:
                    assert self.GBA_KM_b[i, j] == 0.0, throw_message(MessageType.ERROR, f"Backward KM value should be zero for metabolite <code>{list(self.GBA_row_indices.keys())[i]}</code> and reaction <code>{list(self.GBA_col_indices.keys())[j]}</code>.")
                if self.GBA_KM_b[i, j] != 0.0:
                    assert self.GBA_KM_f[i, j] == 0.0, throw_message(MessageType.ERROR, f"Forward KM value should be zero for metabolite <code>{list(self.GBA_row_indices.keys())[i]}</code> and reaction <code>{list(self.GBA_col_indices.keys())[j]}</code>.")
        K_df = pd.DataFrame(self.GBA_KM_f+self.GBA_KM_b, index=self.GBA_row_indices.keys(), columns=self.GBA_col_indices.keys())
        K_df.replace(-0.0, 0.0, inplace=True)
        K_df.to_csv(model_path+"/K.csv", sep=";")
        del(K_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Write the KA and KI matrices      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if np.any(self.GBA_KA):
            KA_df = pd.DataFrame(self.GBA_KA, index=self.GBA_row_indices.keys(), columns=self.GBA_col_indices.keys())
            KA_df.replace(-0.0, 0.0, inplace=True)
            KA_df.to_csv(model_path+"/KA.csv", sep=";")
            del(KA_df)
        if np.any(self.GBA_KI):
            KI_df = pd.DataFrame(self.GBA_KI, index=self.GBA_row_indices.keys(), columns=self.GBA_col_indices.keys())
            KI_df.replace(-0.0, 0.0, inplace=True)
            KI_df.to_csv(model_path+"/KI.csv", sep=";")
            del(KI_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 7) Write rho                         #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open(model_path+"/rho.csv", "w")
        f.write(";(g/L)\n")
        f.write("rho;"+str(self.GBA_rho)+"\n")
        f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 8) Write the conditions              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        conditions_df = pd.DataFrame(self.GBA_conditions)
        conditions_df.replace(-0.0, 0.0, inplace=True)
        conditions_df.to_csv(model_path+"/conditions.csv", sep=";")
        del(conditions_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 9) Write the constant RHS terms      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(self.GBA_constant_rhs) > 0:
            f = open(model_path+"/constant_rhs.csv", "w")
            f.write("metabolite;value\n")
            for item in self.GBA_constant_rhs.items():
                f.write(item[0]+";"+str(item[1])+"\n")
            f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 10) Write the constant reactions     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(self.GBA_constant_reactions) > 0:
            f = open(model_path+"/constant_reactions.csv", "w")
            f.write("reaction;value\n")
            for item in self.GBA_constant_reactions.items():
                f.write(item[0]+";"+str(item[1])+"\n")
            f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 11) Save protein contributions       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open(model_path+"/protein_contributions.csv", "w")
        f.write("reaction;protein;contribution\n")
        for r_id, contributions in self.GBA_protein_contributions.items():
            for p_id, contribution in contributions.items():
                f.write(r_id+";"+p_id+";"+str(contribution)+"\n")
        f.close()

    def export_to_ods( self, name: Optional[str] = "", path: Optional[str] = "." ) -> None:
        """
        Export the model to a folder in ODS format.

        Parameters
        ----------
        name : Optional[str], default=""
            Name of the model. If empty, the model name is used.
        path : Optional[str], default="."
            Path to the folder.
        """
        assert self.check_conversion(), throw_message(MessageType.ERROR, "The model is not converted to GBA units. Convert the model before building GBA variables.")
        assert self.GBA_is_built, throw_message(MessageType.ERROR, f"The model <code>{self.name}</code> is not built")
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Write the information           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        Info_df = None
        if len(self.info) > 0:
            rows = []
            for key, value in self.info.items():
                rows.append([key, value if isinstance(value, str) else ""])
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        rows.append(["", subkey, subvalue])
            Info_df         = pd.DataFrame(rows)
            Info_df.columns = ["", "", ""]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Write the mass fraction matrix  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        M_df = pd.DataFrame(self.GBA_M, index=self.GBA_row_indices.keys(), columns=self.GBA_col_indices.keys())
        M_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Write the kcat vectors          #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        kcat_df           = pd.DataFrame(self.GBA_kcat_f, index=self.GBA_col_indices.keys(), columns=["kcat_f"])
        kcat_df["kcat_b"] = self.GBA_kcat_b
        kcat_df           = kcat_df.transpose()
        kcat_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Write the K matrix              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for i in self.GBA_row_indices.values():
            for j in self.GBA_col_indices.values():
                if self.GBA_KM_f[i, j] != 0.0:
                    assert self.GBA_KM_b[i, j] == 0.0, throw_message(MessageType.ERROR, f"Backward KM value should be zero for metabolite <code>{list(self.GBA_row_indices.keys())[i]}</code> and reaction <code>{list(self.GBA_col_indices.keys())[j]}</code>.")
                if self.GBA_KM_b[i, j] != 0.0:
                    assert self.GBA_KM_f[i, j] == 0.0, throw_message(MessageType.ERROR, f"Forward KM value should be zero for metabolite <code>{list(self.GBA_row_indices.keys())[i]}</code> and reaction <code>{list(self.GBA_col_indices.keys())[j]}</code>.")
        K_df = pd.DataFrame(self.GBA_KM_f+self.GBA_KM_b, index=self.GBA_row_indices.keys(), columns=self.GBA_col_indices.keys())
        K_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Write the KA and KI matrices    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        KA_df = None
        KI_df = None
        if np.any(self.GBA_KA):
            KA_df = pd.DataFrame(self.GBA_KA, index=self.GBA_row_indices.keys(), columns=self.GBA_col_indices.keys())
            KA_df.replace(-0.0, 0.0, inplace=True)
        if np.any(self.GBA_KI):
            KI_df = pd.DataFrame(self.GBA_KI, index=self.GBA_row_indices.keys(), columns=self.GBA_col_indices.keys())
            KI_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Write rho                         #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        rho_df = pd.DataFrame([["rho", self.GBA_rho]], columns=["", "(g/L)"])
        rho_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 7) Write the conditions              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        conditions_df = pd.DataFrame(self.GBA_conditions)
        conditions_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 8) Write the constant terms          #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        constant_rhs_df       = None
        constant_reactions_df = None
        if len(self.GBA_constant_rhs) > 0:
            constant_rhs_df = pd.DataFrame(list(self.GBA_constant_rhs.items()), columns=["metabolite", "value"])
            constant_rhs_df.replace(-0.0, 0.0, inplace=True)
        if len(self.GBA_constant_reactions) > 0:
            constant_reactions_df = pd.DataFrame(list(self.GBA_constant_reactions.items()), columns=["reaction", "value"])
            constant_reactions_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 9) Write the protein contributions #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        protein_contributions_df = None
        if len(self.GBA_protein_contributions) > 0:
            rows = []
            for r_id, contributions in self.GBA_protein_contributions.items():
                for p_id, contribution in contributions.items():
                    rows.append([r_id, p_id, contribution])
            protein_contributions_df = pd.DataFrame(rows, columns=["reaction", "protein", "contribution"])
            protein_contributions_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 10) Write the variables in ods       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ods_path = path+"/"+(name if name != "" else self.name)+".ods"
        with pd.ExcelWriter(ods_path, engine="odf") as writer:
            if Info_df is not None:
                Info_df.to_excel(writer, sheet_name="Info", index=False, header=False)
            M_df.to_excel(writer, sheet_name="M")
            kcat_df.to_excel(writer, sheet_name="kcat")
            K_df.to_excel(writer, sheet_name="K")
            if KA_df is not None:
                KA_df.to_excel(writer, sheet_name="KA")
            if KI_df is not None:
                KI_df.to_excel(writer, sheet_name="KI")
            rho_df.to_excel(writer, sheet_name="rho", index=False, header=True)
            conditions_df.to_excel(writer, sheet_name="conditions")
            if constant_rhs_df is not None:
                constant_rhs_df.to_excel(writer, sheet_name="constant_rhs", index=False)
            if constant_reactions_df is not None:
                constant_reactions_df.to_excel(writer, sheet_name="constant_reactions", index=False)
            if protein_contributions_df is not None:
                protein_contributions_df.to_excel(writer, sheet_name="protein_contributions", index=False)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 12) Free memory                      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        del(Info_df)
        del(M_df)
        del(kcat_df)
        del(K_df)
        del(KA_df)
        del(KI_df)
        del(rho_df)
        del(conditions_df)
        del(constant_rhs_df)
        del(constant_reactions_df)
        del(protein_contributions_df)

    def write_proteins_list( self, path: Optional[str] = ".", name: Optional[str] = "" ) -> None:
        """
        Write list of proteins into CSV format.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to the folder.
        name : Optional[str], default=""
            Name of the folder.
        """
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        filename = path+"/"+(self.name if name == "" else name)+"_proteins.csv"
        f        = open(filename, "w")
        f.write("id;name;mass;sequence;length;gene;product;essentiality\n")
        for p in self.proteins.values():
            name         = ("" if p.name is None else p.name)
            gene         = ("" if p.gene is None else p.gene)
            product      = ("" if p.product is None else p.product)
            essentiality = ""
            if p.essentiality is not None and p.essentiality == GeneEssentiality.ESSENTIAL:
                essentiality = "Essential"
            elif p.essentiality is not None and p.essentiality == GeneEssentiality.QUASI_ESSENTIAL:
                essentiality = "Quasi-essential"
            elif p.essentiality is not None and p.essentiality == GeneEssentiality.NON_ESSENTIAL:
                essentiality = "Non-essential"
            else:
                essentiality = "Unknown"
            f.write(p.id+";"+name+";"+str(p.mass)+";"+p.formula+";"+str(len(p.formula))+";"+gene+";"+product+";"+essentiality+"\n")
        f.close()
    
    def write_ribosomal_proteins_list( self, path: Optional[str] = ".", name: Optional[str] = "" ) -> None:
        """
        Write list of ribosomal proteins into CSV format.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to the folder.
        name : Optional[str], default=""
            Name of the folder.
        """
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        filename = path+"/"+(self.name if name == "" else name)+"_ribosomal_proteins.csv"
        f        = open(filename, "w")
        f.write("id;contribution\n")
        for p_id, contrib in self.reactions["Ribosome"].protein_contributions.items():
            f.write(p_id+";"+str(contrib)+"\n")
        f.close()
    
    def write_metabolites_list( self, path: Optional[str] = ".", name: Optional[str] = "" ) -> None:
        """
        Write list of metabolites into CSV format.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to the folder.
        name : Optional[str], default=""
            Name of the folder.
        """
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        filename = path+"/"+(self.name if name == "" else name)+"_metabolites.csv"
        f        = open(filename, "w")
        f.write("id;name;location;category;mass;formula;kegg_id\n")
        for m in self.metabolites.values():
            location = ""
            if m.species_location == SpeciesLocation.INTERNAL:
                location = "internal"
            elif m.species_location == SpeciesLocation.EXTERNAL:
                location = "external"
            elif m.species_location == SpeciesLocation.UNKNOWN:
                location = "unknown"
            category = ""
            if m.species_type == SpeciesType.DNA:
                category = "DNA"
            elif m.species_type == SpeciesType.RNA:
                category = "RNA"
            elif m.species_type == SpeciesType.PROTEIN:
                category = "protein"
            elif m.species_type == SpeciesType.SMALLMOLECULE:
                category = "small molecule"
            elif m.species_type == SpeciesType.MACROMOLECULE:
                category = "large molecule"
            elif m.species_type == SpeciesType.UNKNOWN:
                category = "unknown"
            formula = m.formula
            kegg_id = ""
            if "kegg.compound" in m.annotation:
                kegg_id = m.annotation["kegg.compound"]
            f.write(m.id+";"+m.name+";"+location+";"+category+";"+str(m.mass)+";"+formula+";"+kegg_id+"\n")
        f.close()
    
    def write_reactions_list( self, path: Optional[str] = ".", name: Optional[str] = "" ) -> None:
        """
        Write list of reactions into CSV format.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to the folder.
        name : Optional[str], default=""
            Name of the folder.
        """
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        filename = path+"/"+(self.name if name == "" else name)+"_reactions.csv"
        f        = open(filename, "w")
        f.write("id;name;type;lb;ub;expression;proteins;GPR;enzyme_mass\n")
        for r in self.reactions.values():
            r_type = ""
            if r.reaction_type == ReactionType.METABOLIC:
                r_type = "metabolic"
            elif r.reaction_type == ReactionType.TRANSPORT:
                r_type = "transporter"
            elif r.reaction_type == ReactionType.SPONTANEOUS:
                r_type = "spontaneous"
            elif r.reaction_type == ReactionType.EXCHANGE:
                r_type = "exchange"
            proteins = " + ".join([f"{r.proteins[p_id]} {p_id}" for p_id in r.proteins])
            GPR      = ("and" if r.GPR == ReactionGPR.AND else "or" if r.GPR == ReactionGPR.OR else "none")
            f.write(r.id+";"+r.name+";"+r_type+";"+str(r.lb)+";"+str(r.ub)+";"+r.expression+";"+proteins+";"+GPR+";"+str(r.enzyme_mass)+"\n")
        f.close()
    
    def write_kinetic_parameters_list( self, path: Optional[str] = ".", name: Optional[str] = "" ) -> None:
        """
        Write list of kinetic parameters into CSV format.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to the folder.
        name : Optional[str], default=""
            Name of the folder.
        """
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        #~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Write kcat values #
        #~~~~~~~~~~~~~~~~~~~~~~#
        filename = path+"/"+(self.name if name == "" else name)+"_kcat.csv"
        f        = open(filename, "w")
        f.write("reaction_id;direction;kcat\n")
        for r in self.reactions.values():
            f.write(r.id+";forward;"+str(r.kcat[ReactionDirection.FORWARD])+"\n")
            f.write(r.id+";backward;"+str(r.kcat[ReactionDirection.BACKWARD])+"\n")
        f.close()
        #~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Write KM values   #
        #~~~~~~~~~~~~~~~~~~~~~~#
        filename = path+"/"+(self.name if name == "" else name)+"_km.csv"
        f        = open(filename, "w")
        f.write("reaction_id;metabolite_id;km\n")
        for r in self.reactions.values():
            for m_id, km in r.km.items():
                f.write(r.id+";"+m_id+";"+str(km)+"\n")
        f.close()

    def write_protein_contributions_list( self, path: Optional[str] = ".", name: Optional[str] = "" ) -> None:
        """
        Write the list of protein contributions into a CSV file.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to the folder.
        name : Optional[str], default=""
            Name of the folder.
        """
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        filename = path+"/"+(self.name if name == "" else name)+"_protein_contributions.csv"
        f        = open(filename, "w")
        f.write("reaction;protein;contribution\n")
        for r in self.reactions.values():
            if r.protein_contributions is not None:
                for p_id, contribution in r.protein_contributions.items():
                    f.write(r.id+";"+p_id+";"+str(contribution)+"\n")
        f.close()
    
    def write_subsystems_list( self, path: Optional[str] = ".", name: Optional[str] = "" ) -> None:
        """
        Write list of reaction's subsystems into CSV format.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to the folder.
        name : Optional[str], default=""
            Name of the folder.
        """
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        filename = path+"/"+(self.name if name == "" else name)+"_subsystems.csv"
        f        = open(filename, "w")
        f.write("reaction_id;subsystem\n")
        for r in self.reactions.values():
            f.write(r.id+";"+r.subsystem+"\n")
        f.close()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 7) Utility functions        #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def generate_kinetic_parameter_tables( self, path: Optional[str] = "." ):
        """
        Generate the kinetic parameter tables for the model.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to the folder.
        """
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Generate kcat prediction table #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open(path+"/kcat_table.csv", "w")
        f.write("reaction_id;reaction;direction;substrate_ids;product_ids;protein_id;substrate_KEGGs;product_KEGGs;sequence\n")
        for r_ID, r in self.reactions.items():
            r_expression = r.expression
            r_proteins   = list(r.proteins.keys())
            if len(r_proteins) == 0:
                continue
            r_sequences = {p_ID: self.proteins[p_ID].formula for p_ID in r_proteins}
            substrates  = {}
            products    = {}
            for m_ID, coef in r.metabolites.items():
                if "kegg.compound" in self.metabolites[m_ID].annotation:
                    m_KEGG = self.metabolites[m_ID].annotation["kegg.compound"]
                    if coef < 0:
                        substrates[m_ID] = m_KEGG
                    elif coef > 0:
                        products[m_ID] = m_KEGG
            if len(substrates) == 0 or len(products) == 0:
                continue
            for p_ID in r_proteins:
                f.write(r_ID+";"+r_expression+";forward;"+",".join(substrates.keys())+";"+",".join(products.keys())+";"+p_ID+";"+",".join(substrates.values())+";"+",".join(products.values())+";"+r_sequences[p_ID]+"\n")
                f.write(r_ID+";"+r_expression+";backward;"+";"+",".join(products.keys())+";"+",".join(substrates.keys())+p_ID+";"+",".join(products.values())+";"+",".join(substrates.values())+";"+r_sequences[p_ID]+"\n")
        f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Generate KM prediction table #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open(path+"/KM_table.csv", "w")
        f.write("reaction_id;reaction;direction;metabolite_id;protein_id;KEGG;sequence\n")
        for r_ID, r in self.reactions.items():
            r_expression = r.expression
            r_proteins   = list(r.proteins.keys())
            if len(r_proteins) == 0:
                continue
            r_sequences = {p_ID: self.proteins[p_ID].formula for p_ID in r_proteins}
            substrates  = {}
            products    = {}
            for m_ID, coef in r.metabolites.items():
                if "kegg.compound" in self.metabolites[m_ID].annotation:
                    m_KEGG = self.metabolites[m_ID].annotation["kegg.compound"]
                    if coef < 0:
                        substrates[m_ID] = m_KEGG
                    elif coef > 0:
                        products[m_ID] = m_KEGG
            for m_ID in substrates:
                for p_ID in r_proteins:
                    f.write(r_ID+";"+r_expression+";forward;"+m_ID+";"+p_ID+";"+substrates[m_ID]+";"+r_sequences[p_ID]+"\n")
            for m_ID in products:
                for p_ID in r_proteins:
                    f.write(r_ID+";"+r_expression+";backward;"+m_ID+";"+p_ID+";"+products[m_ID]+";"+r_sequences[p_ID]+"\n")
        f.close()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 8) Summary functions        #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def information( self ) -> None:
        """
        Print the model information.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Compile information #
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        dfs = {}
        for category in self.info.keys():
            data = self.info[category]
            df = {
                "Element": [],
                "Description": []
            }
            for key, content in data.items():
                df["Element"].append(key)
                df["Description"].append(content)
            df = pd.DataFrame(df)
            dfs[category] = df
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Display table       #
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        html_str  = "<h1>Model "+self.name+"</h1>"
        for category, df in dfs.items():
            html_str += "<table>"
            html_str += "<tr style='text-align:left'><td style='vertical-align:top'>"
            html_str += "<h2 style='text-align: left;'>"+category+"</h2>"
            html_str += df.to_html(escape=False, index=False, header=False)
            html_str += "</td></tr>"
            html_str += "</table>"
        display_html(html_str,raw=True)

    def summary( self ) -> None:
        """
        Print a summary of the model builder.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Compile information #
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        modeled_proteins = []
        for r in self.reactions.values():
            for p_id in r.proteins:
                if p_id not in modeled_proteins and not p_id in ["average_protein", "spontaneous_protein", "housekeeping_protein"]:
                    modeled_proteins.append(p_id)
        df1 = {
            "Category": ["Known proteins", "Modeled proteins", "Metabolites", "Reactions"],
            "Count": [len(self.proteins), len(modeled_proteins), len(self.metabolites), len(self.reactions)]
        }
        df1 = pd.DataFrame(df1)
        df2 = {
            "Category": ["Small molecules", "Macro-molecules", "DNA(s)", "RNA(s)", "Proteins", "Unknown"],
            "Count": [
                len([x for x in self.metabolites.values() if x.species_type == SpeciesType.SMALLMOLECULE]),
                len([x for x in self.metabolites.values() if x.species_type == SpeciesType.MACROMOLECULE]),
                len([x for x in self.metabolites.values() if x.species_type == SpeciesType.DNA]),
                len([x for x in self.metabolites.values() if x.species_type == SpeciesType.RNA]),
                len([x for x in self.metabolites.values() if x.species_type == SpeciesType.PROTEIN]),
                len([x for x in self.metabolites.values() if x.species_type == SpeciesType.UNKNOWN])
            ],
            "Percentage": [
                f"{len([x for x in self.metabolites.values() if x.species_type == SpeciesType.SMALLMOLECULE])/len(self.metabolites)*100:.2f}%",
                f"{len([x for x in self.metabolites.values() if x.species_type == SpeciesType.MACROMOLECULE])/len(self.metabolites)*100:.2f}%",
                f"{len([x for x in self.metabolites.values() if x.species_type == SpeciesType.DNA])/len(self.metabolites)*100:.2f}%",
                f"{len([x for x in self.metabolites.values() if x.species_type == SpeciesType.RNA])/len(self.metabolites)*100:.2f}%",
                f"{len([x for x in self.metabolites.values() if x.species_type == SpeciesType.PROTEIN])/len(self.metabolites)*100:.2f}%",
                f"{len([x for x in self.metabolites.values() if x.species_type == SpeciesType.UNKNOWN])/len(self.metabolites)*100:.2f}%"
            ]
        }
        df2 = pd.DataFrame(df2)
        df3 = {
            "Category": ["Metabolic", "Transport", "Spontaneous", "Exchange"],
            "Count": [
                len([x for x in self.reactions.values() if x.reaction_type == ReactionType.METABOLIC]),
                len([x for x in self.reactions.values() if x.reaction_type == ReactionType.TRANSPORT]),
                len([x for x in self.reactions.values() if x.reaction_type == ReactionType.SPONTANEOUS]),
                len([x for x in self.reactions.values() if x.reaction_type == ReactionType.EXCHANGE])
            ],
            "Percentage": [
                f"{len([x for x in self.reactions.values() if x.reaction_type == ReactionType.METABOLIC])/len(self.reactions)*100:.2f}%",
                f"{len([x for x in self.reactions.values() if x.reaction_type == ReactionType.TRANSPORT])/len(self.reactions)*100:.2f}%",
                f"{len([x for x in self.reactions.values() if x.reaction_type == ReactionType.SPONTANEOUS])/len(self.reactions)*100:.2f}%",
                f"{len([x for x in self.reactions.values() if x.reaction_type == ReactionType.EXCHANGE])/len(self.reactions)*100:.2f}%"
            ]
        }
        df3 = pd.DataFrame(df3)
        df4 = {
            "Category": ["Forward", "Backward", "Reversible"],
            "Count": [
                len([x for x in self.reactions.values() if x.direction == ReactionDirection.FORWARD]),
                len([x for x in self.reactions.values() if x.direction == ReactionDirection.BACKWARD]),
                len([x for x in self.reactions.values() if x.direction == ReactionDirection.REVERSIBLE])
            ],
            "Percentage": [
                f"{len([x for x in self.reactions.values() if x.direction == ReactionDirection.FORWARD])/len(self.reactions)*100:.2f}%",
                f"{len([x for x in self.reactions.values() if x.direction == ReactionDirection.BACKWARD])/len(self.reactions)*100:.2f}%",
                f"{len([x for x in self.reactions.values() if x.direction == ReactionDirection.REVERSIBLE])/len(self.reactions)*100:.2f}%"
            ]
        }
        df4 = pd.DataFrame(df4)
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Display tables      #
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        html_str  = "<h1>Model build "+self.name+" summary</h1>"
        html_str += "<table>"
        html_str += "<tr style='text-align:left'><td style='vertical-align:top'>"
        html_str += "<h2 style='text-align: left;'>General</h2>"
        html_str += df1.to_html(escape=False, index=False)
        html_str += "</td>"
        html_str += "<td style='vertical-align:top'>"
        html_str += "<h2 style='text-align: left;'>Metabolites</h2>"
        html_str += df2.to_html(escape=False, index=False)
        html_str += "</td>"
        html_str += "<td style='vertical-align:top'>"
        html_str += "<h2 style='text-align: left;'>Reaction types</h2>"
        html_str += df3.to_html(escape=False, index=False)
        html_str += "</td>"
        html_str += "<td style='vertical-align:top'>"
        html_str += "<h2 style='text-align: left;'>Reaction directions</h2>"
        html_str += df4.to_html(escape=False, index=False)
        html_str += "</td></tr>"
        html_str += "</table>"
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

def backup_builder( builder: Builder, name: Optional[str] = "", path: Optional[str] = "" ) -> None:
    """
    Backup a model builder in binary format (extension .gbabuild).

    Parameters
    ----------
    builder : Builder
        Model builder to backup.
    name : str
        Name of the backup file.
    path : str
        Path to the backup file.
    """
    filename = ""
    if name != "":
        filename = name+".gbabuild"
    else:
        filename = builder.name+".gbabuild"
    if path != "":
        filename = path+"/"+filename
    ofile = open(filename, "wb")
    pickle.dump(builder, ofile)
    ofile.close()
    assert os.path.isfile(filename), throw_message(MessageType.ERROR, ".gbabuild file creation failed.")

def load_builder( path: str ) -> Builder:
    """
    Load a model builder from a binary file.

    Parameters
    ----------
    path : str
        Path to the model builder file.
    """
    assert path.endswith(".gbabuild"), throw_message(MessageType.ERROR, "Model builder file extension is missing.")
    assert os.path.isfile(path), throw_message(MessageType.ERROR, "Model builder file not found.")
    ifile   = open(path, "rb")
    builder = pickle.load(ifile)
    ifile.close()
    return builder

