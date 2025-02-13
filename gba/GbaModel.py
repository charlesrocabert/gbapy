#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#***********************************************************************
# GBApy (Growth Balance Analysis for Python)
# Copyright © 2024-2025 Charles Rocabert, Furkan Mert
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
Filename: GbaModel.py
Author: Charles Rocabert, Furkan Mert
Date: 2024-10-22
Description:
    GbaModel class of the GBApy module.
License: GNU General Public License v3.0
Copyright: © 2024-2025 Charles Rocabert, Furkan Mert
"""

import os
import sys
import csv
import dill
import time
import pkgutil
import numpy as np
import pandas as pd
import gurobipy as gp
from pathlib import Path
from typing import Optional
from IPython.display import display_html


from .Enumerations import *
#from GbaBuilder import *

# Setting gurobi environment
env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()


class GbaModel:
    """
    Class to manipulate GBA models.

    Attributes
    ----------

    name : str
        Name of the GBA build.

    1) GBA model
    ============

    1.1) Identifier lists
    ---------------------
    metabolite_ids : list
        List of all metabolite ids.
    x_ids : list
        List of external metabolite ids.
    c_ids : list
        List of internal metabolite ids.
    reaction_ids : list
        List of reaction ids.
    condition_ids : list
        List of condition ids.
    condition_params : list
        List of condition parameter ids.
    
    1.2) Model structure
    --------------------
    Mx : np.array
        Total mass fraction matrix.
    M : np.array
        Internal mass fraction matrix.
    kcat_f : np.array
        Forward kcat vector.
    kcat_b : np.array
        Backward kcat vector.
    KM_f : np.array
        Forward KM matrix.
    KM_b : np.array
        Backward KM matrix.
    KA : np.array
        KA matrix.
    KI : np.array
        KI matrix.
    rKI : np.array
        1/KI matrix.
    reversible : list
        Indicates if the reaction is reversible.
    kinetic_model : list
        Indicates the kinetic model of the reaction.
    direction : list
        Indicates the direction of the reaction.
    conditions : np.array
        List of conditions.
    constant_reactions : dict
        Constant reactions.
    
    1.3) Proteomics
    ---------------
    protein_contribution : dict
        Protein contribution to each reaction.
    proteomics : dict
        Predicted proteomics.

    1.4) Loaded objects
    -------------------
    Mx_loaded : bool
        Is the mass fraction matrix loaded?
    kcat_loaded : bool
        Are the kcat constants loaded?
    KM_f_loaded : bool
        Are the KM forward constants loaded?
    KM_b_loaded : bool
        Are the KM backward constants loaded?
    KA_loaded : bool
        Are the KA constants loaded?
    KI_loaded : bool
        Are the KI constants loaded?
    conditions_loaded : bool
        Are the conditions loaded?
    constant_reactions_loaded : bool
        Are the constant reactions loaded?
    protein_contribution_loaded : bool
        Are the protein contributions loaded?
    LP_solution_loaded : bool
        Is the LP solution loaded?
    
    2) GBA model constants
    ======================

    2.1) Vector lengths
    -------------------
    nx : int
        Number of external metabolites.
    nc : int
        Number of internal metabolites.
    ni : int
        Total number of metabolites.
    nj : int
        Number of reactions.

    2.2) Indices for reactions
    --------------------------
    sM : list
        Columns sum of M.
    e : list
        Enzymatic reaction indices.
    s : list
        Transport reaction indices.
    r : int
        Ribosome reaction index.
    ne : int
        Number of enzymatic reactions.
    ns : int
        Number of transport reactions.

    2.3) Indices
    ------------
    m : list
        Metabolite indices.
    a : int
        Total proteins concentration index.

    2.4) Matrix column rank
    -----------------------
    column_rank : int
        Column rank of M.
    full_column_rank : bool
        Does the matrix have full column rank?

    3) Solutions
    ============
    LP_solution : np.array
        Linear programming solution.
    optimal_solutions : dict
        Optimal f vectors for all conditions.
    random_solutions : dict
        Random f vectors.

    4) GBA model variables
    ======================
    tau_j : np.array
        Tau values (turnover times).
    ditau_j : np.array
        Tau derivative values.
    x : np.array
        External metabolite concentrations.
    c : np.array
        Internal metabolite concentrations.
    xc : np.array
        Metabolite concentrations.
    v : np.array
        Fluxes vector.
    p : np.array
        Protein concentrations vector.
    b : np.array
        Biomass fractions vector.
    density : float
        Cell's relative density.
    mu : float
        Growth rate.
    consistent : bool
        Is the model consistent?
    adjust_concentrations : bool
        Adjust concentrations to avoid negative values.

    5) GBA model dynamical variables
    ================================
    condition : str
        External condition.
    rho : float
        Total density.
    f0 : np.array
        Initial LP solution.
    dmu_f : np.array
        Local mu derivatives with respect to f.
    GCC_f : np.array
        Local growth control coefficients with respect to f.
    f_trunc : np.array
        Truncated f vector (first element is removed).
    f : np.array
        Flux fractions vector.

    6) Trackers
    ===========
    random_data : pd.DataFrame
        Random solution data for all conditions.
    optima_data : pd.DataFrame
        Optima dataframe for all conditions.
    GA_tracker : pd.DataFrame
        Gradient ascent trajectory tracker.
    MC_tracker : pd.DataFrame
        Monte Carlo with genetic drift tracker.
    MCMC_tracker : pd.DataFrame
        MCMC trajectory tracker.

    Methods
    -------

    1) Model loading methods
    ========================
    read_Mx_from_csv( path: Optional[str] = "." ) -> None
        Read the mass fraction matrix M from a CSV file.
    read_kcat_from_csv( path: Optional[str] = "." ) -> None
        Read the kcat forward and backward constant vectors from a CSV file.
    read_KM_f_from_csv( path: Optional[str] = "." ) -> None
        Read the forward Michaelis constant matrix KM from a CSV file.
    read_KM_b_from_csv( path: Optional[str] = "." ) -> None
        Read the backward Michaelis constant matrix KM from a CSV file.
    read_KA_from_csv( path: Optional[str] = "." ) -> None
        Read the activation constants matrix KA from a CSV file.
    read_KI_from_csv( path: Optional[str] = "." ) -> None
        Read the inhibition constants matrix KI from a CSV file.
    read_conditions_from_csv( path: Optional[str] = "." ) -> None
        Read the list of conditions from a CSV file.
    read_constant_reactions_from_csv( path: Optional[str] = "." ) -> None
        Read the list of constant reactions from a CSV file.
    read_protein_contribution_from_csv( path: Optional[str] = "." ) -> None
        Read the list of protein contributions from a CSV file.
    read_LP_from_csv( path: Optional[str] = "." ) -> None
        Read the LP solution from a CSV file (on request).
    check_model_loading( verbose: Optional[bool] = False ) -> None
        Check if the model is loaded correctly.
    initialize_model_mathematical_variables() -> None
        Initialize the model mathematical variables.
    read_from_csv( path: Optional[str] = ".", verbose: Optional[bool] = False ) -> None
        Read the GBA model from CSV files.
    read_from_variables( builder ) -> None
        Read the GBA model diretly from variables.

    2) Getters
    ==========
    get_condition( condition_id: str, condition_param: str ) -> float
        Get the condition value.
    get_trajectory( self, algorithm: str, variable: str ) -> np.array
        Get the trajectory of a variable.
    
    3) Setters
    ==========
    reset_variables() -> None
        Reset the model variables.
    set_condition( condition_id: str ) -> None
        Set the external condition.
    set_f0( f0: np.array ) -> None
        Set the initial LP solution.
    set_f()
        Set the flux fractions vector.
    
    TO DO
    """

    def __init__( self, name: str ) -> None:

        """
        Constructor of the GbaModel class.
        
        Parameters
        ----------
        name : str
            Name of the GBA build.
        """
        assert name != "", "> Error: Empty name"
        self.name = name

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) GBA model                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        ### Identifier lists ###
        self.metabolite_ids   = [] # List of all metabolite ids 
        self.x_ids            = [] # List of external metabolite ids
        self.c_ids            = [] # List of internal metabolite ids
        self.reaction_ids     = [] # List of reaction ids
        self.condition_ids    = [] # List of condition ids
        self.condition_params = [] # List of condition parameter ids

        ### Model structure ###
        self.Mx                 = np.array([]) # Total mass fraction matrix
        self.M                  = np.array([]) # Internal mass fraction matrix
        self.kcat_f             = np.array([]) # Forward kcat vector
        self.kcat_b             = np.array([]) # Backward kcat vector
        self.KM_f               = np.array([]) # Forward KM matrix
        self.KM_b               = np.array([]) # Backward KM matrix
        self.KA                 = np.array([]) # KA matrix
        self.KI                 = np.array([]) # KI matrix
        self.rKI                = np.array([]) # 1/KI matrix
        self.reversible         = []           # Indicates if the reaction is reversible
        self.kinetic_model      = []           # Indicates the kinetic model of the reaction
        self.direction          = []           # Indicates the direction of the reaction
        self.conditions         = np.array([]) # List of conditions
        self.constant_reactions = {}           # Constant reactions

        ### Proteomics ###
        self.protein_contribution = {} # Protein contribution to each reaction
        self.proteomics           = {} # Predicted proteomics

        ### Loaded objects ###
        self.Mx_loaded                   = False # Is the mass fraction matrix loaded?
        self.kcat_loaded                 = False # Are the kcat constants loaded?
        self.KM_f_loaded                 = False # Are the KM forward constants loaded?
        self.KM_b_loaded                 = False # Are the KM backward constants loaded?
        self.KA_loaded                   = False # Are the KA constants loaded?
        self.KI_loaded                   = False # Are the KI constants loaded?
        self.conditions_loaded           = False # Are the conditions loaded?
        self.constant_reactions_loaded   = False # Are the constant reactions loaded?
        self.protein_contribution_loaded = False # Are the protein contributions loaded?
        self.LP_solution_loaded          = False # Is the LP solution loaded?

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) GBA model constants           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        ### Vector lengths ###
        self.nx = 0 # Number of external metabolites
        self.nc = 0 # Number of internal metabolites
        self.ni = 0 # Total number of metabolites
        self.nj = 0 # Number of reactions

        ### Indices for reactions: s (transport), e (enzymatic), and ribosome r ###
        self.sM = [] # Columns sum of M
        self.e  = [] # Enzymatic reaction indices
        self.s  = [] # Transport reaction indices
        self.r  = 0  # Ribosome reaction index
        self.ne = 0  # Number of enzymatic reactions
        self.ns = 0  # Number of transport reactions

        ### Indices: m (metabolite), a (all proteins) ###
        self.m = [] # Metabolite indices
        self.a = 0  # Total proteins concentration index

        ### Matrix column rank ###
        self.column_rank      = 0     # Column rank of M
        self.full_column_rank = False # Does the matrix have full column rank?

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Solutions                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.LP_solution       = np.array([]) # Linear programming solution
        self.optimal_solutions = {}           # Optimal f vectors for all conditions
        self.random_solutions  = {}           # Random f vectors
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) GBA model variables           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.tau_j                 = np.array([]) # Tau values (turnover times)
        self.ditau_j               = np.array([]) # Tau derivative values
        self.x                     = np.array([]) # External metabolite concentrations
        self.c                     = np.array([]) # Internal metabolite concentrations
        self.xc                    = np.array([]) # Metabolite concentrations
        self.v                     = np.array([]) # Fluxes vector
        self.p                     = np.array([]) # Protein concentrations vector
        self.b                     = np.array([]) # Biomass fractions vector
        self.density               = 0.0          # Cell's relative density
        self.mu                    = 0.0          # Growth rate
        self.consistent            = False        # Is the model consistent?
        self.adjust_concentrations = False        # Adjust concentrations to avoid negative values

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) GBA model dynamical variables #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.condition = ""           # External condition
        self.rho       = 0.0          # Total density
        self.f0        = np.array([]) # Initial LP solution
        self.dmu_f     = np.array([]) # Local mu derivatives with respect to f
        self.GCC_f     = np.array([]) # Local growth control coefficients with respect to f
        self.f_trunc   = np.array([]) # Truncated f vector (first element is removed)
        self.f         = np.array([]) # Flux fractions vector

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Trackers                      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.random_data  = pd.DataFrame() # Random solution data for all conditions
        self.optima_data  = pd.DataFrame() # Optima dataframe for all conditions
        self.GA_tracker   = pd.DataFrame() # Gradient ascent trajectory tracker
        self.MC_tracker   = pd.DataFrame() # Monte Carlo with genetic drift tracker
        self.MCMC_tracker = pd.DataFrame() # MCMC trajectory tracker
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Model loading methods           #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def read_Mx_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the mass fraction matrix M from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.Mx_loaded = False
        filename       = path+"/"+self.name+"/M.csv"
        if os.path.exists(filename):
            df                  = pd.read_csv(filename, sep=";")
            self.metabolite_ids = self.metabolite_ids+list(df["Unnamed: 0"])
            self.metabolite_ids = list(dict.fromkeys(self.metabolite_ids))
            self.c_ids          = self.c_ids+[x for x in self.metabolite_ids if not x.startswith("x_")]
            self.x_ids          = self.x_ids+[x for x in self.metabolite_ids if x.startswith("x_")]
            self.x_ids          = list(dict.fromkeys(self.x_ids))
            self.c_ids          = list(dict.fromkeys(self.c_ids))
            self.reaction_ids   = list(df.columns)[1:df.shape[1]]
            df                  = df.drop(["Unnamed: 0"], axis=1)
            df.index            = self.metabolite_ids
            self.Mx             = np.array(df)
            self.Mx             = self.Mx.astype(float)
            self.Mx_loaded      = True
            del(df)

    def read_kcat_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the kcat forward and backward constant vectors from a CSV
        file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.kcat_loaded = False
        filename         = path+"/"+self.name+"/kcat.csv"
        if os.path.exists(filename):
            df              = pd.read_csv(filename, sep=";")
            df              = df.drop(["Unnamed: 0"], axis=1)
            kcat            = np.array(df)
            kcat            = kcat.astype(float)
            self.kcat_f     = np.array(kcat[0,:])
            self.kcat_b     = np.array(kcat[1,:])
            self.reversible = []
            for j in range(len(self.kcat_b)):
                if self.kcat_b[j] > 0.0:
                    self.reversible.append(True)
                else:
                    self.reversible.append(False)
            self.kcat_loaded = True
            del(df)

    def read_KM_f_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the forward Michaelis constant matrix KM from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.KM_f_loaded = False
        filename         = path+"/"+self.name+"/KM_forward.csv"
        if os.path.exists(filename):
            df               = pd.read_csv(filename, sep=";")
            df               = df.drop(["Unnamed: 0"], axis=1)
            df.index         = self.metabolite_ids
            self.KM_f        = np.array(df)
            self.KM_f        = self.KM_f.astype(float)
            self.KM_f_loaded = True
            del(df)

    def read_KM_b_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the backward Michaelis constant matrix KM from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.KM_b_loaded = False
        self.KM_b        = np.zeros(self.KM_f.shape)
        filename         = path+"/"+self.name+"/KM_backward.csv"
        if os.path.exists(filename):
            df               = pd.read_csv(filename, sep=";")
            df               = df.drop(["Unnamed: 0"], axis=1)
            df.index         = self.metabolite_ids
            self.KM_b        = np.array(df)
            self.KM_b        = self.KM_b.astype(float)
            self.KM_b_loaded = True
            del(df)
    
    def read_KA_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the activation constants matrix KA from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.KA_loaded = False
        self.KA        = np.zeros(self.Mx.shape)
        filename       = path+"/"+self.name+"/KA.csv"
        if os.path.exists(filename):
            df             = pd.read_csv(filename, sep=";")
            metabolites    = list(df["Unnamed: 0"])
            df             = df.drop(["Unnamed: 0"], axis=1)
            df.index       = metabolites
            self.KA        = np.array(df)
            self.KA        = self.KA.astype(float)
            self.KA_loaded = True
            del(df)
    
    def read_KI_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the inhibition constants matrix KI from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.KI_loaded = False
        self.KI        = np.zeros(self.Mx.shape)
        filename       = path+"/"+self.name+"/KI.csv"
        if os.path.exists(filename):
            df             = pd.read_csv(filename, sep=";")
            metabolites    = list(df["Unnamed: 0"])
            df             = df.drop(["Unnamed: 0"], axis=1)
            df.index       = metabolites
            self.KI        = np.array(df)
            self.KI        = self.KI.astype(float)
            self.KI_loaded = True
            del(df)

    def read_conditions_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the list of conditions from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.conditions_loaded = False
        filename               = path+"/"+self.name+"/conditions.csv"
        if os.path.exists(filename):
            df                     = pd.read_csv(filename, sep=";")
            self.condition_params  = list(df["Unnamed: 0"])
            self.condition_ids     = list(df.columns)[1:df.shape[1]]
            self.condition_ids     = [str(int(name)) for name in self.condition_ids]
            df                     = df.drop(["Unnamed: 0"], axis=1)
            df.index               = self.condition_params
            self.conditions        = np.array(df)
            self.conditions_loaded = True
            del(df)

    def read_constant_reactions_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the list of constant reactions from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.constant_reactions_loaded = False
        filename                       = path+"/"+self.name+"/constants.csv"
        if os.path.exists(filename):
            f = open(filename, "r")
            l = f.readline()
            l = f.readline()
            self.constant_reactions.clear()
            while l:
                l = l.strip("\n").split(";")
                self.constant_reactions[l[0]] = float(l[1])
                l = f.readline()
            f.close()
            self.constant_reactions_loaded = True

    def read_protein_contribution_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the list of protein contributions from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.protein_contribution_loaded = False
        filename                         = path+"/"+self.name+"/protein_contribution.csv"
        if os.path.exists(filename):
            f = open(filename, "r")
            l = f.readline()
            l = f.readline()
            self.protein_contribution.clear()
            while l:
                l = l.strip("\n").split(";")
                r_id = l[0]
                p_id = l[1]
                val  = float(l[2])
                if r_id not in self.protein_contribution:
                    self.protein_contribution[r_id] = {p_id: val}
                else:
                    self.protein_contribution[r_id][p_id] = val
                l = f.readline()
            f.close()
            self.protein_contribution_loaded = True
    
    def read_LP_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the LP solution from a CSV file (on request).

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.LP_solution_loaded = False
        filename                = path+"/"+self.name+"/f0.csv"
        if os.path.exists(filename):
            df                      = pd.read_csv(filename, sep=";")
            self.LP_solution        = np.array(df["f0"])
            self.LP_solution_loaded = True
            del(df)

    def check_model_loading( self, verbose: Optional[bool] = False ) -> None:
        """
        Check if the model is loaded correctly.

        Parameters
        ----------
        verbose : Optional[bool], default=False
            Print the error messages.
        """
        if not self.Mx_loaded:
            raise ValueError("> Error: Mass fraction matrix Mx not loaded.")
        if not self.kcat_loaded:
            raise ValueError("> Error: kcat constants not loaded.")
        if not self.KM_f_loaded:
            raise ValueError("> Error: KM forward constants not loaded.")
        if not self.KM_b_loaded:
            if verbose:
                print("> Info: KM backward constants not found.")
        if not self.KA_loaded:
            if verbose:
                print("> Info: KA constants not found.")
        if not self.KI_loaded:
            if verbose:
                print("> Info: KI constants not found.")
        if not self.conditions_loaded:
            raise ValueError("> Error: conditions not loaded.")
        if not self.constant_reactions_loaded:
            if verbose:
                print("> Info: constant reactions not found.")
        if not self.protein_contribution_loaded:
            if verbose:
                print("> Info: protein contributions not found.")
        if not self.LP_solution_loaded:
            if verbose:
                print("> Info: LP solution not found.")
    
    def initialize_model_mathematical_variables( self ) -> None:
        """
        Initialize the model mathematical variables.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Inverse of KI                                       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        with np.errstate(divide='ignore'):
            self.rKI                     = 1/self.KI
            self.rKI[np.isinf(self.rKI)] = 0.0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Vector lengths                                      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.nx = len(self.x_ids)
        self.nc = len(self.c_ids)
        self.ni = self.nx+self.nc
        self.nj = len(self.reaction_ids)
        self.x  = np.zeros(self.nx)
        self.c  = np.zeros(self.nc)
        self.xc = np.zeros(self.ni)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Create M matrix                                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.M = np.zeros((self.nc, self.nj))
        for i in range(self.nc):
            met_id = self.c_ids[i]
            for j in range(self.nj):
                self.M[i,j] = self.Mx[self.metabolite_ids.index(met_id),j]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Indices: s (transport), e (enzymatic), r (ribosome) #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.sM = np.sum(self.M, axis=0)
        is_e    = [self.sM[0:(self.nj-1)] == 0]
        self.e  = []
        self.s  = []
        for i in range(self.nj-1):
            if is_e[0][i] == True:
                self.e.append(i)
            else:
                self.s.append(i)
        self.r  = self.nj-1
        self.ne = len(self.e)
        self.ns = len(self.s)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Indices: m (metabolite), a (all proteins)           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.m = list(range(self.nc-1))
        self.a = self.nc-1
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Matrix column rank                                  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.column_rank = np.linalg.matrix_rank(self.M)
        if self.column_rank == self.nj:
            self.full_column_rank = True
        else:
            self.full_column_rank = False
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 7) GBA model dynamical variables                       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.tau_j   = np.zeros(self.nj)
        self.ditau_j = np.zeros((self.nj, self.nc))
        self.x       = np.zeros(self.nx)
        self.c       = np.zeros(self.nc)
        self.xc      = np.zeros(self.ni)
        self.v       = np.zeros(self.nj)
        self.p       = np.zeros(self.nj)
        self.b       = np.zeros(self.nc)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 8) Evolutionary variables                              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.f0      = np.zeros(self.nj)
        self.dmu_f   = np.zeros(self.nj)
        self.GCC_f   = np.zeros(self.nj)
        self.f_trunc = np.zeros(self.nj-1)
        self.f       = np.zeros(self.nj)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 9) Define the kinetic model of each reaction           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.kinetic_model.clear()
        self.direction.clear()
        for j in range(self.nj):
            if (self.kcat_b[j] == 0 and self.KI[:,j].sum() == 0 and self.KA[:,j].sum() == 0):
                self.kinetic_model.append(GbaReactionType.iMM)
                self.direction.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] == 0 and self.KI[:,j].sum() == 0 and self.KA[:,j].sum() > 0):
                self.kinetic_model.append(GbaReactionType.iMMa)
                self.direction.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] == 0 and self.KI[:,j].sum() > 0 and self.KA[:,j].sum() == 0):
                self.kinetic_model.append(GbaReactionType.iMMi)
                self.direction.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] == 0 and self.KI[:,j].sum() > 0 and self.KA[:,j].sum() > 0):
                self.kinetic_model.append(GbaReactionType.iMMia)
                self.direction.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] > 0):
                assert self.KA[:,j].sum() == 0, f"> Error: Reversible Michaelis-Menten reaction cannot have activation (reaction {j})."
                assert self.KI[:,j].sum() == 0, f"> Error: Reversible Michaelis-Menten reaction cannot have inhibition (reaction {j})."
                self.kinetic_model.append(GbaReactionType.rMM)
                self.direction.append(self.direction.append(ReactionDirection.Reversible))
    
    def read_from_csv( self, path: Optional[str] = ".", verbose: Optional[bool] = False ) -> None:
        """
        Read the GBA model from CSV files.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV files.
        verbose : Optional[bool], default=False
            Verbose mode.
        """
        assert os.path.exists(path+"/"+self.name), "> Error: folder "+path+"/"+self.name+" does not exist."
        self.read_Mx_from_csv(path)
        self.read_kcat_from_csv(path)
        self.read_KM_f_from_csv(path)
        self.read_KM_b_from_csv(path)
        self.read_KA_from_csv(path)
        self.read_KI_from_csv(path)
        self.read_conditions_from_csv(path)
        self.read_constant_reactions_from_csv(path)
        self.read_protein_contribution_from_csv(path)
        self.read_LP_from_csv(path)
        self.check_model_loading(verbose)
        self.initialize_model_mathematical_variables()

    def read_from_variables( self, builder ) -> None:
        """
        Read the GBA model diretly from variables.

        Parameters
        ----------
        builder : GbaBuilder
            GBA builder object.
        """
        raise NotImplementedError("> Error: method not implemented yet.")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Getters                         #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    def get_condition( self, condition_id: str, condition_param: str ) -> float:
        """
        Get a condition parameter value.

        Parameters
        ----------
        condition_id : str
            Condition identifier.
        condition_param : str
            Condition parameter identifier.

        Returns
        -------
        float
            Condition parameter value.
        """
        assert condition_id in self.condition_ids, "> Error: unknown condition identifier "+condition_id+"."
        assert condition_param in self.condition_params, "> Error: unknown condition parameter "+condition_param+"."
        i = self.condition_params.index(condition_param)
        j = self.condition_ids.index(condition_id)
        return self.conditions[i,j]

    def get_trajectory( self, algorithm: str, variable: str ) -> np.array:
        """
        Get the trajectory of a variable.

        Parameters
        ----------
        algorithm : str
            Algorithm name.
        variable : str
            Variable name.

        Returns
        -------
        np.array
            Trajectory of the variable.
        """
        assert algorithm in ["GA", "MC", "MCMC"], "> Error: algorithm must be 'GA', 'MC' or 'MCMC'."
        if algorithm == "GA":
            assert not self.GA_tracker.empty, "> Error: no data available for gradient ascent."
            return self.GA_tracker[variable].values
        elif algorithm == "MC":
            assert not self.MC_tracker.empty, "> Error: no data available for Monte Carlo."
            return self.MC_tracker[variable].values
        elif algorithm == "MCMC":
            assert not self.MCMC_tracker.empty, "> Error: no data available for MCMC."
            return self.MCMC_tracker[variable].values
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 3) Setters                         #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def reset_variables( self ) -> None:
        """
        Reset the model variables (used before binary export).
        """
        self.tau_j   = np.zeros(self.nj)
        self.ditau_j = np.zeros((self.nj, self.nc))
        self.x       = np.zeros(self.nx)
        self.c       = np.zeros(self.nc)
        self.xc      = np.zeros(self.ni)
        self.v       = np.zeros(self.nj)
        self.p       = np.zeros(self.nj)
        self.b       = np.zeros(self.nc)
        self.f0      = np.zeros(self.nj)
        self.dmu_f   = np.zeros(self.nj)
        self.GCC_f   = np.zeros(self.nj)
        self.f_trunc = np.zeros(self.nj-1)
        self.f       = np.zeros(self.nj)
    
    def set_condition( self, condition_id: str ) -> None:
        """
        Set the external condition.
        (minimal values bounded to MIN_CONCENTRATION)

        Parameters
        ----------
        condition_id : str
            External condition identifier.
        """
        assert condition_id in self.condition_ids, "> Error: unknown condition identifier "+condition_id+"."
        self.condition = condition_id
        self.rho       = self.get_condition(self.condition, "rho")
        for i in range(self.nx):
            x_name    = self.x_ids[i]
            x_value   = self.get_condition(self.condition, x_name)
            self.x[i] = x_value
            if self.adjust_concentrations and self.x[i] < GbaConstants.MIN_CONCENTRATION.value:
                self.x[i] = GbaConstants.MIN_CONCENTRATION.value

    def set_f0( self, f0: np.array ) -> None:
        """
        Set the initial flux fraction vector f0.
        
        Parameters
        ----------
        f0 : np.array
            Initial flux fraction vector.
        """
        assert len(f0) == self.nj, "> Error: incorrect f0 length."
        self.f0      = np.copy(f0)
        self.f_trunc = np.copy(self.f0[1:self.nj])
        self.f       = np.copy(self.f0)
    
    def set_f( self ) -> None:
        """
        Set the flux fraction vector f from the truncated vector
        f_trunc.
        """
        term1  = (1-self.sM[1:].dot(self.f_trunc))/self.sM[0]
        self.f = np.copy(np.concatenate([np.array([term1]), self.f_trunc]))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 4) Analytical methods              #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def compute_c( self ) -> None:
        """
        Compute the internal metabolite concentrations.
        """
        self.c = self.rho*self.M.dot(self.f)
        if self.adjust_concentrations:
            self.c[self.c < GbaConstants.MIN_CONCENTRATION.value] = GbaConstants.MIN_CONCENTRATION.value
        self.xc = np.concatenate([self.x, self.c])
    
    def iMM( self, j: int ) -> None:
        """
        Compute the turnover time tau for an irreversible
        Michaelis-Menten reaction.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        term1         = np.prod(1.0+self.KM_f[:,j]/self.xc)
        term2         = self.kcat_f[j]
        self.tau_j[j] = term1/term2

    def iMMa( self, j: int ) -> None:
        """
        Compute the turnover time tau for an irreversible
        Michaelis-Menten reaction with activation.
        (only one actibitor per reaction)

        Parameters
        ----------
        j : int
            Reaction index.
        """
        term1         = np.prod(1.0+self.KA[:,j]/self.xc)
        term2         = np.prod(1.0+self.KM_f[:,j]/self.xc)
        term3         = self.kcat_f[j]
        self.tau_j[j] = term1*term2/term3

    def iMMi( self, j: int ) -> None:
        """
        Compute the turnover time tau for an irreversible
        Michaelis-Menten reaction with inhibition.
        (only one inhibitor per reaction)

        Parameters
        ----------
        j : int
            Reaction index.
        """
        term1         = np.prod(1.0+self.xc*self.rKI[:,j])
        term2         = np.prod(1.0+self.KM_f[:,j]/self.xc)
        term3         = self.kcat_f[j]
        self.tau_j[j] = term1*term2/term3
    
    def iMMia( self, j: int ) -> None:
        """
        Compute the turnover time tau for an irreversible
        Michaelis-Menten reaction with inhibition and activation.
        (only one inhibitor and one activator per reaction)

        Parameters
        ----------
        j : int
            Reaction index.
        """
        term1         = np.prod(1.0+self.xc*self.rKI[:,j])
        term2         = np.prod(1.0+self.KA[:,j]/self.xc)
        term3         = np.prod(1.0+self.KM_f[:,j]/self.xc)
        term4         = self.kcat_f[j]
        self.tau_j[j] = term1*term2*term3/term4

    def rMM( self, j: int ) -> None:
        """
        Compute the turnover time tau for a reversible Michaelis-Menten
        reaction.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        term1         = self.kcat_f[j]
        term2         = np.prod(1+self.KM_f[:,j]/self.xc)
        term3         = self.kcat_b[j]
        term4         = np.prod(1+self.KM_b[:,j]/self.xc)
        self.tau_j[j] = 1.0/(term1/term2-term3/term4)
    
    def compute_tau( self, j: int ) -> None:
        """
        Compute the turnover time tau for a reaction j.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        if self.kinetic_model[j] == GbaReactionType.iMM:
            self.iMM(j)
        elif self.kinetic_model[j] == GbaReactionType.iMMa:
            self.iMMa(j)
        elif self.kinetic_model[j] == GbaReactionType.iMMi:
            self.iMMi(j)
        elif self.kinetic_model[j] == GbaReactionType.iMMia:
            self.iMMia(j)
        elif self.kinetic_model[j] == GbaReactionType.rMM:
            self.rMM(j)
    
    def diMM( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for an
        irreversible Michaelis-Menten reaction with respect to
        metabolite concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        constant1 = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            term1             = self.KM_f[y,j]/np.power(self.c[i], 2.0)
            term2             = np.prod(1+self.KM_f[indices,j]/self.xc[indices])
            self.ditau_j[j,i] = -term1*term2/constant1

    def diMMa( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for an
        irreversible Michaelis-Menten reaction with activation
        with respect to metabolite concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        constant1 = np.prod(1+self.KM_f[:,j]/self.xc)
        constant2 = np.prod(1+self.KA[:,j]/self.xc)
        constant3 = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            term1             = self.KA[y,j]/np.power(self.c[i], 2.0)
            term2             = self.KM_f[y,j]/np.power(self.c[i], 2.0)
            term3             = np.prod(1+self.KM_f[indices,j]/self.xc[indices])
            self.ditau_j[j,i] = -(constant1*term1+constant2*term2*term3)/constant3
    
    def diMMi( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for an
        irreversible Michaelis-Menten reaction with inhibition
        with respect to metabolite concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        constant1 = np.prod(1+self.KM_f[:,j]/self.xc)
        constant2 = np.prod(1+self.xc*self.rKI[:,j])
        constant3 = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            term1             = self.rKI[y,j]*constant1
            term2             = self.KM_f[y,j]/np.power(self.c[i], 2)
            term3             = np.prod(1+self.KM_f[indices,j]/self.xc[indices])
            self.ditau_j[j,i] = (term1-constant2*term2*term3)/constant3

    def diMMia( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for an
        irreversible Michaelis-Menten reaction with activation and
        inhibition with respect to metabolite concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        constant1 = np.prod(1.0+self.c*self.rKI[:,j])
        constant2 = np.prod(1.0+self.KA[:,j]/self.c)
        constant3 = np.prod(1.0+self.KM_f[:,j]/self.c)
        constant4 = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            term1             = self.rKI[y,j]
            term2             = -self.KA[y,j]/np.power(self.c[i], 2.0)
            term3             = -self.KM_f[y,j]/np.power(self.c[i], 2.0)
            term4             = np.prod(1+self.KM_f[indices,j]/self.c[indices])
            term5             = (term1*constant2*constant3)+(term2*constant1*constant3)+(term3*term4*constant1*constant2)
            self.ditau_j[j,i] = term5/constant4

    def drMM( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for a
        reversible Michaelis-Menten reaction with respect to
        metabolite concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        constant1 = self.kcat_f[j]
        constant2 = self.kcat_b[j]
        constant3 = np.prod(1+self.KM_f[:,j]/self.xc)
        constant4 = np.prod(1+self.KM_b[:,j]/self.xc)
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            term1             = self.KM_f[y,j] / np.power(self.c[i] + self.KM_f[y,j], 2.0)
            term2             = self.KM_b[y,j] / np.power(self.c[i] + self.KM_b[y,j], 2.0)
            term3             = np.prod(1 + self.KM_f[indices,j]/self.xc[indices])
            term4             = np.prod(1 + self.KM_b[indices,j]/self.xc[indices])
            term5             = term1*constant1/term3-term2*constant2/term4
            term6             = constant1/constant3-constant2/constant4
            self.ditau_j[j,i] = -term5/np.power(term6, 2.0)
    
    def compute_dtau( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for a reaction
        j.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        if self.kinetic_model[j] == GbaReactionType.iMM:
            self.diMM(j)
        elif self.kinetic_model[j] == GbaReactionType.iMMa:
            self.diMMa(j)
        elif self.kinetic_model[j] == GbaReactionType.iMMi:
            self.diMMi(j)
        elif self.kinetic_model[j] == GbaReactionType.iMMia:
            self.diMMia(j)
        elif self.kinetic_model[j] == GbaReactionType.rMM:
            self.drMM(j)
    
    def compute_mu( self ) -> None:
        """
        Compute the growth rate mu.
        """
        self.mu = self.M[self.a,self.r]*self.f[self.r]/(self.tau_j.dot(self.f))

    def compute_v( self ) -> None:
        """
        Compute the fluxes v.
        """
        self.v = self.mu*self.rho*self.f

    def compute_p( self ) -> None:
        """
        Compute the protein concentrations p.
        """
        self.p = self.tau_j*self.v

    def compute_b( self ) -> None:
        """
        Compute the biomass fractions b.
        """
        self.b = self.M.dot(self.f)

    def compute_density( self ) -> None:
        """
        Compute the cell density.
        (should be equal to 1)
        """
        self.density = self.sM.dot(self.f)

    def compute_dmu_f( self ) -> None:
        """
        Compute the local growth rate gradient with respect to f.
        """
        term1      = np.power(self.mu, 2)/self.b[self.a]
        term2      = self.M[self.a,:]/self.mu
        term3      = self.f.T.dot(self.rho*self.ditau_j.dot(self.M))
        term4      = self.tau_j
        self.dmu_f = term1*(term2-term3-term4)

    def compute_GCC_f( self ) -> None:
        """
        Compute the local growth control coefficients with respect to f.
        """
        self.GCC_f = self.dmu_f-self.dmu_f[0]*(self.sM/self.sM[0])
    
    def calculate_state( self ) -> None:
        """
        Calculate the model state.
        """
        self.compute_c()
        for j in range(self.nj):
            self.compute_tau(j)
        self.compute_mu()
        self.compute_v()
        self.compute_p()
        self.compute_b()
        self.compute_density()
        for j in range(self.nj):
            self.compute_dtau(j)
        self.compute_dmu_f()
        self.compute_GCC_f()

    def check_model_consistency( self ) -> None:
        """
        Check the model state's consistency.
        """
        test1 = (np.abs(self.density-1.0) < GbaConstants.DENSITY_TOL.value)
        test2 = (sum(1 for x in self.c if x < -GbaConstants.NEGATIVE_C_TOL.value) == 0)
        test3 = (sum(1 for x in self.p if x < -GbaConstants.NEGATIVE_P_TOL.value) == 0)
        self.consistent = True
        if not (test1 and test2 and test3):
            self.consistent = False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 5) Generation of initial solutions #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def solve_local_linear_problem( self, rhs_factor: Optional[float] = 10.0 ) -> None:
        """
        Solve the local linear problem to find the initial solution.

        Description
        -----------
        The local linear problem consists in finding the maximal ribosome
        flux fraction f^r, with a minimal production of each metabolite.
        The constraints are mass conservation (M*f = b) and surface flux
        balance (sM*f = 1).

        Parameters
        ----------
        rhs_factor : Optional[float], default=100.0
            Factor dividing the rhs of the mass conservation constraint.
        """
        lb_vec = [GbaConstants.MIN_FLUX_FRACTION.value]*self.nj
        ub_vec = [GbaConstants.MAX_FLUX_FRACTION.value]*self.nj
        for item in self.constant_reactions.items():
           r_index         = self.reaction_ids.index(item[0])
           lb_vec[r_index] = item[1]
           ub_vec[r_index] = item[1]
        gpmodel = gp.Model(env=env)
        v       = gpmodel.addMVar(self.nj, lb=lb_vec, ub=ub_vec)
        min_b   = 1/self.nc/rhs_factor
        rhs     = np.repeat(min_b, self.nc)
        gpmodel.setObjective(v[-1], gp.GRB.MAXIMIZE)
        gpmodel.addConstr(self.M @ v >= rhs, name="c1")
        gpmodel.addConstr(self.sM @ v == 1, name="c2")
        gpmodel.optimize()
        try:
            self.LP_solution = np.copy(v.X)
            return True
        except:
            return False

    def generate_random_initial_solutions( self, condition_id: str, nb_solutions: int, max_trials: int, min_mu: Optional[float] = 1e-3, verbose: Optional[bool] = False ) -> None:
        """
        Generate random initial solutions.

        Parameters
        ----------
        condition_id : str
            Condition identifier.
        nb_solutions : int
            Number of solutions to generate.
        max_trials : int
            Maximum number of trials.
        min_mu : Optional[float], default=1e-3
            Minimal growth rate.
        verbose : Optional[bool], default=False
            Verbose mode.
        """
        assert condition_id in self.condition_ids, f"> Error: unknown condition identifier ({condition_id})."
        assert nb_solutions > 0, f"> Error: number of solutions must be greater than 0."
        assert max_trials >= nb_solutions, f"> Error: number of trials must be greater than the number of solutions."
        assert min_mu >= 0.0, f"> Error: minimal growth rate must be positive."
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the random data frame #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        columns          = ["condition", "mu", "density"]
        columns          = columns + self.reaction_ids
        self.random_data = pd.DataFrame(columns=columns)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Find the random solutions        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.set_condition(condition_id)
        self.random_solutions.clear()
        solutions = 0
        trials    = 0
        while solutions < nb_solutions and trials < max_trials:
            trials        += 1
            negative_term  = True
            while negative_term:
                self.f_trunc = np.random.rand(self.nj-1)
                self.f_trunc = self.f_trunc*(GbaConstants.MAX_FLUX_FRACTION-GbaConstants.MIN_FLUX_FRACTION)+GbaConstants.MIN_FLUX_FRACTION
                self.set_f()
                if self.f[0] >= 0.0:
                    negative_term = False
            self.calculate_state()
            self.check_model_consistency()
            if self.consistent and np.isfinite(self.mu) and self.mu > min_mu:
                solutions += 1
                data_dict  = {"condition": condition_id, "mu": self.mu, "density": self.density}
                for reaction_id, fluxfraction in zip(self.reaction_ids, self.f):
                    data_dict[reaction_id] = fluxfraction
                data_row                         = pd.Series(data=data_dict)
                self.random_data                 = pd.concat([self.random_data, data_row.to_frame().T], ignore_index=True)
                self.random_solutions[solutions] = np.copy(self.f)
                if verbose:
                    print("> ", solutions, " solutions were found after ", trials, " trials (last mu = "+str(round(self.mu,5))+")")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 6) Optimization Methods            #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def mutate_f( self, index: int, sigma: float ) -> np.array:
        """
        Mutate one element 'index' of f with a Gaussian standard deviation 'sigma'.

        Parameters
        ----------
        index : int
            Index of the element to mutate.
        sigma : float
            Standard deviation of the Gaussian distribution.
        
        Returns
        -------
        np.array
            Non-mutated flux fraction vector.
        """
        assert index < self.nj, f"> Error: index {index} is out of bounds."
        assert sigma > 0.0, f"> Error: sigma must be positive."
        non_mutated_f        = np.copy(self.f_trunc)
        epsilon              = np.random.normal(0.0, sigma, size=1)
        self.f_trunc[index] += epsilon 
        self.f_trunc[self.f_trunc < GbaConstants.MIN_FLUX_FRACTION] = GbaConstants.MIN_FLUX_FRACTION
        self.set_f()
        return non_mutated_f
    
    def calculate_pi( self, selection_coefficient: float, N_e: float ) -> float:
        """
        Calculate the fixation probability pi for a given selection coefficient and effective population size.

        Parameters
        ----------
        selection_coefficient : float
            Selection coefficient.
        N_e : float
            Effective population size.

        Returns
        -------
        float
            Fixation probability.
        """
        pi = 0.0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if selection_coefficient == 0.0:
                pi = 1.0/N_e
            else:
                pi = (1-np.exp(-2*selection_coefficient)) / (1-np.exp(-2*N_e*selection_coefficient))
        return pi
    
    def track_variables( self, variables: list[int], data_dict: dict[str, float] ) -> None:
        """
        Track additional variables.

        Parameters
        ----------
        variables : list[int]
            Variables to track (f, v, p, b, c).
        data_dict : dict[str, float]
            Data dictionary.
        """
        allowed_variables = ["f", "v", "p", "b", "c"]
        for variable in variables:
            assert variable in allowed_variables, f"> Error: variable {variable} is not allowed."
            if variable == "f":
                for r_id, value in zip(self.reaction_ids, getattr(self, variable)):
                    data_dict["f_"+r_id] = value
            elif variable == "v":
                for r_id, value in zip(self.reaction_ids, getattr(self, variable)):
                    data_dict["v_"+r_id] = value
            elif variable == "p":
                for r_id, value in zip(self.reaction_ids, getattr(self, variable)):
                    data_dict["p_"+r_id] = value
            elif variable == "b":
                for c_id, value in zip(self.c_ids, getattr(self, variable)):
                    data_dict["b_"+c_id] = value
            elif variable == "c":
                for c_id, value in zip(self.c_ids, getattr(self, variable)):
                    data_dict["c_"+c_id] = value

    def block_reactions( self ) -> None:
        """
        Block reactions tending to zero.

        Description
        -----------
        f values tending towards zero are bounded to min value.
        Corresponding derivative values are set to zero depending
        on the direction:
        - f -> 0+ and gcc < 0: f = min_f, gcc = 0
        - f -> 0- and gcc > 0: f = -min_f, gcc = 0
        """
        for j in range(self.nj-1):
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 1) Reaction is irreversible and positive #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            if self.direction[j+1] == ReactionDirection.Forward and self.f_trunc[j] <= GbaConstants.MIN_FLUX_FRACTION.value:
                self.f_trunc[j] = GbaConstants.MIN_FLUX_FRACTION.value
                if self.GCC_f[(j+1)] < 0.0:
                    self.GCC_f[(j+1)] = 0.0
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 2) Reaction is irreversible and negative #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            elif self.direction[j+1] == ReactionDirection.Backward and self.f_trunc[j] >= -GbaConstants.MIN_FLUX_FRACTION.value:
                self.f_trunc[j] = -GbaConstants.MIN_FLUX_FRACTION.value
                if self.GCC_f[(j+1)] > 0.0:
                    self.GCC_f[(j+1)] = 0.0
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 3) Reaction is reversible                #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # elif self.direction[j+1] == ReactionDirection.Reversible and np.abs(self.f_trunc[j]) <= GbaConstants.MIN_FLUX_FRACTION.value:
            #     self.GCC_f[(j+1)] = 0.0
            #     if self.f_trunc[j] >= 0.0:
            #         self.f_trunc[j] = GbaConstants.MIN_FLUX_FRACTION.value
            #     elif self.f_trunc[j] < 0.0:
            #         self.f_trunc[j] = -GbaConstants.MIN_FLUX_FRACTION.value
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 4) Reaction is constant                  #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            if self.reaction_ids[(j+1)] in self.constant_reactions:
                self.f_trunc[j] = self.constant_reactions[self.reaction_ids[(j+1)]]
                self.GCC_f[(j+1)] = 0.0
    
    def gradient_ascent( self, condition_id: Optional[str] = "1", max_time: Optional[float] = 10.0,
                         initial_dt: Optional[float] = 0.01, track: Optional[bool] = False,
                        variables: Optional[list[str]] = ["f"], label: Optional[int] = 1,
                        verbose: Optional[bool] = False, print_period: Optional[int] = 0 ) -> tuple[bool, float]:
        """
        Run a gradient ascent algorithm to find the optimal flux state.

        Parameters
        ----------
        condition_id : str, default="1"
            Condition identifier.
        max_time : float, default=10.0
            Maximum time for the algorithm.
        initial_dt : float, default=0.01
            Initial time step.
        track : bool, default=False
            Track the trajectory of variables.
        variables : list[str], default=["f"]
            Additional variables to track.
        label : int, default=1
            Label for the trajectory.
        verbose : bool, default=False
            Verbose mode.
        print_period : int, default=0
            Period for printing the state.

        Returns
        -------
        tuple[bool, float]
            Convergence status and run time.
        """
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.adjust_concentrations = False
        self.set_condition(condition_id)
        self.calculate_state()
        self.check_model_consistency()
        assert self.consistent, "> Error: Initial model is not consistent."
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize the tracker   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if track:
            if self.GA_tracker.empty:
                columns = ["label", "condition", "dt", "t", "mu", "fixed"]
                self.GA_tracker = pd.DataFrame(columns=columns)
            data_dict = {"label": label, "condition": condition_id, "dt": initial_dt, "t": 0.0, "mu": self.mu, "fixed": 0}
            self.track_variables(variables, data_dict)
            data_row        = pd.Series(data=data_dict)
            self.GA_tracker = pd.concat([self.GA_tracker, data_row.to_frame().T], ignore_index=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Initialize the algorithm #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        t                     = 0.0
        dt                    = initial_dt
        mu_alteration_counter = 0
        previous_f            = np.copy(self.f_trunc)
        previous_mu           = self.mu
        self.converged        = False
        nb_iterations         = 0
        nb_fixed              = 0
        dt_counter            = 0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Start the main loop      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        while (t < max_time):
            nb_iterations += 1
            if verbose and print_period > 0 and nb_iterations%print_period == 0:
               print("> Iteration: ",nb_iterations, " ( time =", t, ", mu =", self.mu, ", dt =", dt, ")")
            ### 4.1) Test trajectory convergence ###
            if mu_alteration_counter >= GbaConstants.TRAJECTORY_CONVERGENCE_COUNT.value:
                self.converged = True
                break
            ### 4.2) Calculate the next step ###
            previous_mu = self.mu
            self.block_reactions()
            self.f_trunc = self.f_trunc+self.GCC_f[1:]*dt
            self.set_f()
            self.calculate_state()
            self.check_model_consistency()
            ### 4.3) If the model is consistent: ###
            if self.consistent and self.mu >= previous_mu:
                previous_f  = np.copy(self.f_trunc)
                t           = t + dt
                dt_counter += 1
                nb_fixed   += 1
                if track and nb_iterations%GbaConstants.EXPORT_DATA_COUNT == 0:
                    data_dict = {"label": label, "condition": condition_id, "dt": dt, "t": t, "mu": self.mu, "fixed": nb_fixed}
                    self.track_variables(variables, data_dict)
                    data_row        = pd.Series(data=data_dict)
                    self.GA_tracker = pd.concat([self.GA_tracker, data_row.to_frame().T], ignore_index=True)
                ### Check if mu changes significantly ###
                if np.abs(self.mu - previous_mu) < GbaConstants.TRAJECTORY_CONVERGENCE_TOL.value:
                    mu_alteration_counter += 1
                else:
                    mu_alteration_counter = 0
                ### Check if dt is never changing, and possibly increase it ###
                if dt_counter == GbaConstants.INCREASING_DT_COUNT.value:
                    dt         = dt*GbaConstants.INCREASING_DT_FACTOR.value
                    dt_counter = 0
            ### 4.4) If the model is inconsistent: ###
            else:
                self.f_trunc = np.copy(previous_f)
                self.set_f()
                self.calculate_state()
                self.check_model_consistency()
                assert self.consistent, "> Error: Previous model is not consistent"
                if (dt > GbaConstants.MIN_DT):
                    dt         = dt/GbaConstants.DECREASING_DT_FACTOR.value
                    dt_counter = 0
                else:
                    raise AssertionError(f"> Error: Adaptative timestep < {GbaConstants.MIN_DT}.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if t >= max_time:
            if verbose:
                print("> Gradient ascent: maximum time reached (condition="+str(condition_id)+",\tmu="+str(round(self.mu, 5))+",\tnb iterations="+str(nb_iterations)+",\tnb fixed="+str(nb_fixed)+")")
            return False, run_time
        else:
            if verbose:
                print("> Gradient ascent: convergence reached (condition="+str(condition_id)+",\tmu="+str(round(self.mu, 5))+",\tnb iterations="+str(nb_iterations)+",\tnb fixed="+str(nb_fixed)+")")
            return True, run_time

    def compute_optima( self, max_time: Optional[int] = 10, initial_dt: Optional[float] = 0.01,
                        verbose: Optional[bool] = False ) -> float:
        """
        Compute the optima by gradient ascent for all conditions.

        Parameters
        ----------
        max_time : Optional[int], default=10
            Maximum time for the algorithm.
        initial_dt : Optional[float], default=0.01
            Initial time step.
        verbose : Optional[bool], default=False
            Verbose mode.

        Returns
        -------
        float
            Run time.
        """
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the optima data frame #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        columns = ["condition", "mu", "density", "converged", "run_time"]
        for r_id in self.reaction_ids:
            columns.append("f_"+r_id)
        self.optima_data = pd.DataFrame(columns=columns)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Calculate the optima             #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.optimal_solutions.clear()
        for condition_id in self.condition_ids:
            self.set_f0(self.LP_solution)
            converged, run_time = self.gradient_ascent(condition_id=condition_id, max_time=max_time, initial_dt=initial_dt)
            data_dict           = {"condition": condition_id, "mu": self.mu, "density": self.density, "converged": int(converged), "run_time": run_time}
            self.track_variables(["f"], data_dict)
            data_row                             = pd.Series(data=data_dict)
            self.optima_data                     = pd.concat([self.optima_data, data_row.to_frame().T], ignore_index=True)
            self.optimal_solutions[condition_id] = np.copy(self.f)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Return the result                  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if verbose:
            print(f"> All optima were computed in {run_time} seconds")
        return run_time
    
    def MC_simulation( self, condition_id: Optional[str] = "1", max_time: Optional[float] = 10.0,
                       max_iterations: Optional[int] = 100000, sigma: Optional[float] = 0.1,
                       N_e: Optional[float] = 2.5e7, track: Optional[bool] = False,
                       variables: Optional[list[str]] = ["f"], label: Optional[int] = 1,
                       verbose: Optional[bool] = False, print_period: Optional[int] = 0 ) -> tuple[bool, float]:
        """
        Run a Monte Carlo simulation with genetic drift.

        Notes
        -----
        The algorithm is based on the Pál & Miklós (1999) model.
        f(t+1) = f(t) + sigma * dmu/df + epsilon.

        Parameters
        ----------
        condition_id : Optional[str], default="1"
            Condition identifier.
        max_time : Optional[float], default=10.0
            Maximum time for the algorithm.
        max_iterations : Optional[int], default=100000
            Maximum number of iterations.
        sigma : Optional[float], default=0.1
            Standard deviation of the noise.
        N_e : Optional[float], default=2.5e7
            Effective population size.
        track : Optional[bool], default=False
            Track the trajectory of variables.
        variables : Optional[list[str]], default=["f"]
            Additional variables to track.
        label : Optional[int], default=1
            Label for the trajectory.
        verbose : Optional[bool], default=False
            Verbose mode.
        print_period : Optional[int], default=0
            Period for printing the state.

        Returns
        -------
        tuple[bool, float]
            Convergence status and run time.
        """
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.set_condition(condition_id)
        self.calculate_state()
        self.check_model_consistency()
        assert self.consistent, "> Error: initial model is not consistent."
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize tracker       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if track:
            if self.MC_tracker.empty:
                columns = ["label", "condition", "t", "mu", "fixed"]
                self.MC_tracker = pd.DataFrame(columns=columns)
            data_dict = {"label": label, "condition": condition_id, "t": 0.0, "mu": self.mu, "fixed": 0}
            self.track_variables(variables, data_dict)
            data_row        = pd.Series(data=data_dict)
            self.MC_tracker = pd.concat([self.MC_tracker, data_row.to_frame().T], ignore_index=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Initialize the algorithm #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        t             = 0.0
        previous_f    = np.copy(self.f_trunc)
        previous_mu   = self.mu
        nb_iterations = 0
        nb_fixed      = 0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Start the loop           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        while (t < max_time):
            nb_iterations += 1
            if verbose and print_period > 0 and nb_iterations%print_period == 0:
               print("> Iteration: "+str(nb_iterations)+" (time = "+str(t)+", mu = "+str(self.mu)+", fixed = "+str(nb_fixed)+")")
            if nb_iterations >= max_iterations:
                if verbose:
                    print("> Maximum number of iterations reached (condition "+condition_id+").")
                break
            ### 4.1) Calculate the next step ###
            epsilon      = np.random.normal(0.0, np.sqrt(sigma/(2.0*N_e)), size=self.nj-1)
            self.f_trunc = self.f_trunc+sigma*self.GCC_f[1:]+epsilon
            #self.block_reactions()
            self.set_f()
            self.calculate_state()
            self.check_model_consistency()
            ### 4.2) If the model is consistent: ###
            if self.consistent and self.mu > 1e-10:# and self.mu >= previous_mu:
                previous_f   = np.copy(self.f_trunc)
                previous_mu  = self.mu
                t           += 1
                nb_fixed    += 1
                if track and nb_iterations % GbaConstants.EXPORT_DATA_COUNT == 0:
                    data_dict = {"label": label, "condition": condition_id, "t": t, "mu": self.mu, "fixed": nb_fixed}
                    self.track_variables(variables, data_dict)
                    data_row        = pd.Series(data=data_dict)
                    self.MC_tracker = pd.concat([self.MC_tracker, data_row.to_frame().T], ignore_index=True)
            ### 4.3) If the model is inconsistent: ###
            else:
                self.f_trunc = np.copy(previous_f)
                self.set_f()
                self.calculate_state()
                self.check_model_consistency()
                assert self.consistent, "> Error: previous model is not consistent."
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if nb_fixed == 0 and nb_iterations < max_iterations:
            print("> MC simulation completed with no fixed mutation (condition="+condition_id+", mu="+str(round(self.mu, 5))+", nb iterations="+str(nb_iterations)+", nb fixed="+str(nb_fixed)+").")
            return False, run_time
        elif nb_fixed > 0 and nb_iterations < max_iterations:
            print("> MC simulation completed (condition="+condition_id+", mu="+str(round(self.mu, 5))+", nb iterations="+str(nb_iterations)+", nb fixed="+str(nb_fixed)+").")
            return True, run_time
        elif nb_iterations >= max_iterations:
            print("> MC simulation: maximum iterations reached (condition="+condition_id+", mu="+str(round(self.mu, 5))+", nb iterations="+str(nb_iterations)+", nb fixed="+str(nb_fixed)+").")
            return False, run_time

    def MCMC_simulation( self, condition_id: Optional[str] = "1", max_iterations: Optional[int] = 100000,
                         sigma: Optional[float] = 0.1, N_e: Optional[float] = 2.5e7,
                         track: Optional[bool] = False , variables: Optional[list[str]] = ["f"],
                         label: Optional[int] = 1, verbose: Optional[bool] = False,
                         print_period: Optional[int] = 0 ) -> tuple[bool, float]:
        """
        Run a Markov Monte Carlo simulation with genetic drift.

        Notes
        -----
        The algorithm is based on the standard MCMC formulation (Gillespie, 1983).

        Parameters
        ----------
        condition_id : Optional[str], default="1"
            Condition identifier.
        max_iterations : Optional[int], default=100000
            Maximum number of iterations.
        sigma : Optional[float], default=0.1
            Standard deviation of the noise.
        N_e : Optional[float], default=2.5e7
            Effective population size.
        track : Optional[bool], default=False
            Track the trajectory of variables.
        variables : Optional[list[str]], default=["f"]
            Additional variables to track.
        label : Optional[int], default=1
            Label for the trajectory.
        verbose : Optional[bool], default=False
            Verbose mode.
        print_period : Optional[int], default=0
            Period for printing the state.

        Returns
        -------
        tuple[bool, float]
            Convergence status and run time.
        """
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.set_condition(condition_id)
        self.calculate_state()
        self.check_model_consistency()
        assert self.consistent, "> Error: initial model is not consistent"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize trackers       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if track:
            if self.MCMC_tracker.empty:
                columns           = ["label", "condition", "t", "mu", "fixed"]
                self.MCMC_tracker = pd.DataFrame(columns=columns)
            data_dict = {"label": label, "condition": condition_id, "t": 0.0, "mu": self.mu, "fixed": 0}
            self.track_variables(variables, data_dict)
            data_row          = pd.Series(data=data_dict)
            self.MCMC_tracker = pd.concat([self.MCMC_tracker, data_row.to_frame().T], ignore_index=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Initialize the algorithm  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        current_mu = self.mu
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Start the MCMC            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        nb_iterations = 0
        nb_fixed      = 0
        while nb_iterations < max_iterations:
            nb_iterations += 1
            if verbose and print_period > 0 and nb_iterations%print_period == 0:
               print("> Iteration: "+str(nb_iterations)+" (mu = "+str(self.mu)+", fixed = "+str(nb_fixed)+")")
            ### 4.1) Draw reaction to mutate at random ###
            reaction_index = np.random.randint(len(self.f_trunc))
            current_mu     = self.mu
            non_mutated_f  = self.mutate_f(reaction_index, sigma)
            self.calculate_state()
            self.check_model_consistency()
            ### 4.2) Check model consistency and simulate fixation ###
            if self.consistent:
                mutated_mu = self.mu
                s          = 1.0-current_mu/mutated_mu
                pi         = self.calculate_pi(s, N_e)
                ### 4.3) Undo Mutation if no fixation occurs ###
                if np.random.rand() >= pi:
                    self.f_trunc = np.copy(non_mutated_f)
                    self.set_f()
                ### 4.4) Save Mutation for trajectory if fixation occurs ###
                else:
                    nb_fixed += 1
                    if track and nb_iterations % GbaConstants.EXPORT_DATA_COUNT == 0:
                        data_dict = {"label": label, "condition": condition_id, "t": nb_iterations, "mu": self.mu, "fixed": nb_fixed}
                        self.track_variables(variables, data_dict)
                        data_row          = pd.Series(data=data_dict)
                        self.MCMC_tracker = pd.concat([self.MCMC_tracker, data_row.to_frame().T], ignore_index=True)
            ### 4.5) Undo Mutation if model is inconsistent ###
            else:
                self.f_trunc = np.copy(non_mutated_f)
                self.set_f()
            self.calculate_state()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if nb_fixed == 0 and nb_iterations < max_iterations:
            print("> MCMC: simulation completed with no fixed mutation (condition="+condition_id+", mu="+str(round(self.mu, 5))+", nb iterations="+str(nb_iterations)+", nb fixed="+str(nb_fixed)+").")
            return False, run_time, nb_fixed
        elif nb_fixed > 0 and nb_iterations < max_iterations:
            print("> MCMC: simulation completed (condition="+condition_id+", mu="+str(round(self.mu, 5))+", nb iterations="+str(nb_iterations)+", nb fixed="+str(nb_fixed)+").")
            return True, run_time, nb_fixed
        elif nb_iterations >= max_iterations:
            print("> MCMC: maximum iterations reached (condition="+condition_id+", mu="+str(round(self.mu, 5))+", nb iterations="+str(nb_iterations)+", nb fixed="+str(nb_fixed)+").")
            return False, run_time, nb_fixed

    def save_f0( self, path: Optional[str] = "." ) -> None:
        """
        Save the initial flux state to CSV.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to save the initial flux state.
        """
        f = open(path+"/"+self.name+"/f0.csv", "w")
        f.write("reaction;f0\n")
        for i in range(self.nj):
            f.write(self.reaction_ids[i]+";"+str(self.f0[i])+"\n")
        f.close()

    def save_random_solutions( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None:
        """
        Save the random data to CSV.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to save the trajectory.
        label : Optional[str], default=""
            Label for the trajectory.
        """
        assert os.path.exists(path), f"> Error: path {path} does not exist."
        header = path+"/"+self.name
        if label != "":
            header += "_"+str(label)
        if not self.random_data.empty:
            self.random_data.to_csv(header+"_random_solutions.csv", sep=';', index=False)
    
    def save_optima( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None:
        """
        Save the optima data to CSV.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to save the trajectory.
        label : Optional[str], default=""
            Label for the trajectory.
        """
        assert os.path.exists(path), f"> Error: path {path} does not exist."
        header = path+"/"+self.name
        if label != "":
            header += "_"+str(label)
        if not self.optima_data.empty:
            self.optima_data.to_csv(header+"_optima.csv", sep=';', index=False)
    
    def save_gradient_ascent_trajectory( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None:
        """
        Save the gradient ascent trajectory to CSV.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to save the trajectory.
        label : Optional[str], default=""
            Label for the trajectory.
        """
        assert os.path.exists(path), f"> Error: path {path} does not exist."
        header = path+"/"+self.name
        if label != "":
            header += "_"+str(label)
        if not self.GA_tracker.empty:
            self.GA_tracker.to_csv(header+"_GA_trajectory.csv", sep=';', index=False)

    def save_MC_trajectory( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None:
        """
        Save the Monte Carlo trajectory to CSV.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to save the trajectory.
        label : Optional[str], default=""
            Label for the trajectory.
        """
        assert os.path.exists(path), f"> Error: path {path} does not exist."
        header = path+"/"+self.name
        if label != "":
            header += "_"+str(label)
        if not self.MC_tracker.empty:
            self.MC_tracker.to_csv(header+"_MC_trajectory.csv", sep=';', index=False)
    
    def save_MCMC_trajectory( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None:
        """
        Save the Markov Chain Monte Carlo trajectory to CSV.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to save the trajectory.
        label : Optional[str], default=""
            Label for the trajectory.
        """
        assert os.path.exists(path), f"> Error: path {path} does not exist."
        header = path+"/"+self.name
        if label != "":
            header += "_"+str(label)
        if not self.MCMC_tracker.empty:
            self.MCMC_tracker.to_csv(header+"_MCMC_trajectory.csv", sep=';', index=False)
    
    def save_all_trajectories( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None:
        """
        Save all trajectories to CSV.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to save the trajectories.
        label : Optional[str], default=""
            Label for the trajectories.
        """
        self.save_gradient_ascent_trajectory(path, label)
        self.save_MC_tracker_trajectory(path, label) 
        self.save_MCMC_trajectory(path, label)

    def clear_gradient_ascent_trajectory( self ) -> None:
        """
        Clear the gradient ascent trajectory.
        """
        self.GA_tracker = pd.DataFrame()
    
    def clear_MC_trajectory( self ) -> None:
        """
        Clear the Monte Carlo trajectory.
        """
        self.MC_tracker = pd.DataFrame()
    
    def clear_MCMC_trajectory( self ) -> None:
        """
        Clear the Markov Chain Monte Carlo trajectory.
        """
        self.MCMC_tracker = pd.DataFrame()
    
    def clear_all_trajectories( self ) -> None:
        """
        Clear all trajectories.
        """
        self.clear_gradient_ascent_trajectory()
        self.clear_MC_trajectory()
        self.clear_MCMC_trajectory()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 7) Summary function         #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def summary( self ) -> None:
        """
        Print a summary of the GBA model.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Compile information #
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        df1 = {
            "Category": ["Nb metabolites", "Nb external metabolites", "Nb internal metabolites"],
            "Count": [self.ni, self.nx, self.nc]
        }
        df1 = pd.DataFrame(df1)
        df2 = {
            "Category": ["Nb reactions", "Nb exchange reactions", "Nb internal reactions"],
            "Count": [self.nj, self.ns, self.ne]
        }
        df2 = pd.DataFrame(df2)
        df3 = {
            "Category": ["Column rank", "Is full column rank?"],
            "Count": [self.column_rank, self.full_column_rank]
        }
        df3 = pd.DataFrame(df3)
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Display tables      #
        #~~~~~~~~~~~~~~~~~~~~~~~~#
        html_str  = "<h1>GBA model "+self.name+" summary</h1>"
        html_str += "<table>"
        html_str += "<tr style='text-align:left'><td style='vertical-align:top'>"
        html_str += "<h2 style='text-align: left;'>Metabolites</h2>"
        html_str += df1.to_html(escape=False, index=False)
        html_str += "</td>"
        html_str += "<td style='vertical-align:top'>"
        html_str += "<h2 style='text-align: left;'>Reactions</h2>"
        html_str += df2.to_html(escape=False, index=False)
        html_str += "</td>"
        html_str += "<td style='vertical-align:top'>"
        html_str += "<h2 style='text-align: left;'>Matrix rank</h2>"
        html_str += df3.to_html(escape=False, index=False)
        html_str += "</td></tr>"
        html_str += "</table>"
        display_html(html_str,raw=True)


def read_csv_model( name: str, path: Optional[str] = "." ) -> GbaModel:
    """
    Read a GBA model from CSV files.

    Parameters
    ----------
    name : str
        Name of the GBA model.
    path : Optional[str], default="."
        Path to the model folder.

    Returns
    -------
    GbaModel
        The loaded GBA model.
    """
    assert os.path.exists(path+"/"+name), "> Error: the folder "+path+"/"+name+" does not exist."
    model = GbaModel(name)
    model.read_from_csv(path=path)
    return model

def get_toy_model_path( model_name: str ) -> str:
    """
    Get the path of a GBA toy model included in the Python package as CSV files.

    Parameters
    ----------
    model_name : str
        Name of the toy model.

    Returns
    -------
    str
        Path to the toy model.
    """
    model_dir  = Path(pkgutil.resolve_name("gba.data").__file__).parent
    model_path = Path(model_dir , "toy_models/"+model_name)
    return str(model_path)

def read_toy_model( name: str ) -> GbaModel:
    """
    Read a GBA toy model included in the Python package as CSV files.

    Parameters
    ----------
    name : str
        Name of the toy model.

    Returns
    -------
    GbaModel
        The loaded GBA model.
    """
    model_dir  = Path(pkgutil.resolve_name("gba.data").__file__).parent
    model_path = str(Path(model_dir , "toy_models/"))
    model      = read_csv_model(name=name, path=model_path)
    return model

def backup_gba_model( model: GbaModel, name: Optional[str] = "", path: Optional[str] = "." ) -> None:
    """
    Backup a GBA model in binary format (extension .gba).

    Parameters
    ----------
    model : GbaModel
        GBA model to backup.
    name : str
        Name of the backup file.
    path : str
        Path to the backup file.
    """
    filename = ""
    if name != "":
        filename = path+"/"+name+".gba"
    else:
        filename = path+"/"+model.name+".gba"
    ofile = open(filename, "wb")
    dill.dump(model, ofile)
    ofile.close()
    assert os.path.isfile(filename), "> Error: .gba file creation failed."

def load_gba_model( path: str ) -> GbaModel:
    """
    Load a GBA model from a binary file.

    Parameters
    ----------
    path : str
        Path to the GBA model file.
    """
    assert path.endswith(".gba"), "> Error: GBA model file extension is missing."
    assert os.path.isfile(path), "> Error: GBA model file not found."
    ifile = open(path, "rb")
    model = dill.load(ifile)
    ifile.close()
    return model

def create_gba_model( name: str, path: Optional[str] = ".", gba_path: Optional[str] = ".", save_LP: Optional[bool] = False, save_optima: Optional[bool] = False ) -> None:
    """
    Create a GBA model from CSV files, and save it as a binary file.

    Parameters
    ----------
    name : str
        Name of the GBA model.
    path : Optional[str], default="."
        Path to the binary file.
    gba_path : Optional[str], default=""
        Path to save the GBA model.
    save_LP : Optional[bool], default=False
        Save the LP solution.
    save_optima : Optional[bool], default=False
        Save the optima.
    """
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Create and load the model from CSV files #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    model = read_csv_model(name=name, path=path)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Compute and save f0 if requested         #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if save_LP:
        print("> Computing LP solution for model "+model.name+"...")
        model.solve_local_linear_problem()
        model.set_f0(model.LP_solution)
        model.set_condition("1")
        model.calculate_state()
        model.check_model_consistency()
        if model.consistent:
            model.save_f0(path=path)
        else:
            raise Exception("> Error: model is inconsistent with condition 1. f0 vector cannot be saved.")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 3) Compute and save optima if requested     #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if save_optima:
        print("> Computing optima for model "+model.name+"...")
        if not save_LP:
            model.read_LP_from_csv(path=path)
        model.compute_optima(max_time=10000, initial_dt=0.01)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 4) Clean model and dump binary backup       #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    model.reset_variables()
    backup_gba_model(model=model, name=name, path=gba_path)
    del model

