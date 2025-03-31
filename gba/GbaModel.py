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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display_html

try:
    from .Enumerations import *
except:
    from Enumerations import *

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
        Name of the GBA model.
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
    directions : list
        Indicates the direction of the reaction.
    conditions : np.array
        List of conditions.
    constant_rhs : dict
        Constant right-hand side terms.
    constant_reactions : dict
        Constant reactions.
    protein_contributions : dict
        Protein contributions for each reaction.
    proteomics : dict
        Predicted proteomics.
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
    constant_rhs_loaded : bool
        Are the constant right-hand side terms loaded?
    constant_reactions_loaded : bool
        Are the constant reactions loaded?
    protein_contributions_loaded : bool
        Are the protein contributions loaded?
    LP_solution_loaded : bool
        Is the LP solution loaded?
    nx : int
        Number of external metabolites.
    nc : int
        Number of internal metabolites.
    ni : int
        Total number of metabolites.
    nj : int
        Number of reactions.
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
    m : list
        Metabolite indices.
    a : int
        Total proteins concentration index.
    column_rank : int
        Column rank of M.
    full_column_rank : bool
        Does the matrix have full column rank?
    LP_solution : np.array
        Linear programming solution.
    optimal_solutions : dict
        Optimal f vectors for all conditions.
    random_solutions : dict
        Random f vectors.
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
    read_Mx_from_csv( path: Optional[str] = "." ) -> None
        Read the mass fraction matrix M from a CSV file.
    read_kcat_from_csv( path: Optional[str] = "." ) -> None
        Read the kcat forward and backward constant vectors from a CSV
        file.
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
    read_constant_rhs_from_csv( path: Optional[str] = "." ) -> None
        Read the list of constant RHS terms from a CSV file.
    read_constant_reactions_from_csv( path: Optional[str] = "." ) -> None
        Read the list of constant reactions from a CSV file.
    read_protein_contributions_from_csv( path: Optional[str] = "." ) -> None
        Read the list of protein contributions from a CSV file.
    read_LP_from_csv( path: Optional[str] = "." ) -> None
        Read the LP solution from a CSV file (on request).
    check_model_loading( verbose: Optional[bool] = False ) -> None
        Check if the model is loaded correctly.
    initialize_model_mathematical_variables( ) -> None
        Initialize the model mathematical variables.
    read_from_csv( path: Optional[str] = ".", verbose: Optional[bool] = False ) -> None
        Read the GBA model from CSV files.
    write_to_csv( path: Optional[str] = ".", verbose: Optional[bool] = False ) -> None
        Write the GBA model to CSV files.
    get_condition( self, condition_id: str, condition_param: str ) -> float
        Get the value of a condition parameter.
    get_vector( self, source: str, variable: str ) -> np.array
        Get the value of a variable from a source.
    clear_conditions( self ) -> None
        Clear all external conditions from the GBA model.
    add_condition( self, condition_id: str, rho: float, default_concentration: Optional[float] = 1.0, metabolites: Optional[dict[str, float]] = None ) -> None
        Add a new condition to the GBA model.
    clear_constant_rhs( self ) -> None
        Clear all constant right-hand side terms from the GBA model.
    add_constant_rhs( self, metabolite_id: str, value: float ) -> None
        Add a new constant right-hand side term to the GBA model.
    clear_constant_reactions( self ) -> None
        Clear all constant reactions from the GBA model.
    add_constant_reaction( self, reaction_id: str, value: float ) -> None
        Add a new constant reaction to the GBA model.
    reset_variables( self ) -> None
        Reset all variables of the GBA model.
    set_condition( self, condition_id: str ) -> None
        Set the current condition of the GBA model.
    set_f0( self, f0: np.array ) -> None
        Set the initial LP solution of the GBA model.
    set_f( self ) -> None
        Set the flux fractions vector of the GBA model.
    compute_c( self ) -> None
        Compute the internal metabolite concentrations.
    iMM( self, j: int ) -> None
        Compute the turnover time tau for an irreversible
        Michaelis-Menten reaction.
    iMMa( self, j: int ) -> None
        Compute the turnover time tau for an irreversible
        Michaelis-Menten reaction with activation.
        (only one actibitor per reaction)
    iMMi( self, j: int ) -> None
        Compute the turnover time tau for an irreversible
        Michaelis-Menten reaction with inhibition.
        (only one inhibitor per reaction)
    iMMia( self, j: int ) -> None
        Compute the turnover time tau for an irreversible
        Michaelis-Menten reaction with inhibition and activation.
        (only one inhibitor and one activator per reaction)
    rMM( self, j: int ) -> None
        Compute the turnover time tau for a reversible Michaelis-Menten
        reaction.
    compute_tau( self, j: int ) -> None
        Compute the turnover time tau for a reaction j.
    diMM( self, j: int ) -> None
        Compute the derivative of the turnover time tau for an
        irreversible Michaelis-Menten reaction with respect to
        metabolite concentrations.
    diMMa( self, j: int ) -> None
        Compute the derivative of the turnover time tau for an
        irreversible Michaelis-Menten reaction with activation
        with respect to metabolite concentrations.
    diMMi( self, j: int ) -> None
        Compute the derivative of the turnover time tau for an
        irreversible Michaelis-Menten reaction with inhibition
        with respect to metabolite concentrations.
    diMMia( self, j: int ) -> None
        Compute the derivative of the turnover time tau for an
        irreversible Michaelis-Menten reaction with activation and
        inhibition with respect to metabolite concentrations.
    drMM( self, j: int ) -> None
        Compute the derivative of the turnover time tau for a
        reversible Michaelis-Menten reaction with respect to
        metabolite concentrations.
    compute_dtau( self, j: int ) -> None
        Compute the derivative of the turnover time tau for a
        reaction j.
    compute_mu( self ) -> None
        Compute the growth rate mu.
    compute_v( self ) -> None
        Compute the fluxes v.
    compute_p( self ) -> None
        Compute the protein concentrations p.
    compute_b( self ) -> None
        Compute the biomass fractions b.
    compute_density( self ) -> None
        Compute the cell density (should be equal to 1).
    compute_dmu_f( self ) -> None
        Compute the local growth rate gradient with respect to f.
    compute_GCC_f( self ) -> None
        Compute the local growth control coefficients with respect to f.
    calculate_state( self ) -> None
        Calculate the model state.
    check_model_consistency( self ) -> None
        Check the model state's consistency.
    solve_local_linear_problem( self,max_flux_fraction: Optional[float] = 10.0, rhs_factor: Optional[float] = 10.0 ) -> None
        Solve the local linear problem to find the initial solution.
    generate_LP_initial_solution( self, max_flux_fraction: Optional[float] = 10.0, rhs_factor: Optional[float] = 10.0, condition_id: Optional[str] = "1", save_f0: Optional[str] = None ) -> None
        Generate an initial solution using a linear program.
    generate_random_initial_solutions( self, condition_id: str, nb_solutions: int, max_trials: int, max_flux_fraction: Optional[float] = 10.0, min_mu: Optional[float] = 1e-3, verbose: Optional[bool] = False ) -> None
        Generate random initial solutions.
    mutate_f( self, index: int, sigma: float ) -> np.array
        Mutate one element 'index' of f with a Gaussian standard deviation 'sigma'.
    calculate_pi( self, selection_coefficient: float, N_e: float ) -> float
        Calculate the fixation probability pi for a given selection coefficient and effective population size.
    track_variables( self, variables: list[int], data_dict: dict[str, float] ) -> None
        Track additional variables.
    block_reactions( self, block_GCC: Optional[bool] = True ) -> None
        Block reactions tending to zero.
    gradient_ascent( self, condition_id: Optional[str] = "1", max_time: Optional[float] = 10.0, initial_dt: Optional[float] = 0.01, track: Optional[bool] = False, variables: Optional[list[str]] = ["f"], label: Optional[int] = 1, verbose: Optional[bool] = False, print_period: Optional[int] = 0 ) -> tuple[bool, float]
        Run a gradient ascent algorithm to find the optimal flux state.
    compute_optima( self, max_time: Optional[int] = 10, initial_dt: Optional[float] = 0.01, verbose: Optional[bool] = False ) -> float
        Compute the optima by gradient ascent for all conditions.
    MC_simulation( self, condition_id: Optional[str] = "1", max_time: Optional[float] = 10.0, max_iterations: Optional[int] = 100000, sigma: Optional[float] = 0.1, N_e: Optional[float] = 2.5e7, track: Optional[bool] = False, variables: Optional[list[str]] = ["f"], label: Optional[int] = 1, verbose: Optional[bool] = False, print_period: Optional[int] = 0 ) -> tuple[bool, float]
        Run a Monte Carlo simulation with genetic drift.
    MCMC_simulation( self, condition_id: Optional[str] = "1", max_iterations: Optional[int] = 100000, sigma: Optional[float] = 0.1, N_e: Optional[float] = 2.5e7, track: Optional[bool] = False , variables: Optional[list[str]] = ["f"], label: Optional[int] = 1, verbose: Optional[bool] = False, print_period: Optional[int] = 0 ) -> tuple[bool, float]
        Run a Markov Monte Carlo simulation with genetic drift.
    save_f0( self, path: Optional[str] = "." ) -> None
        Save the initial flux state to CSV.
    save_random_solutions( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None
        Save the random data to CSV.
    save_optima( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None
        Save the optima data to CSV.
    save_gradient_ascent_trajectory( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None
        Save the gradient ascent trajectory to CSV.
    save_MC_trajectory( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None
        Save the Monte Carlo trajectory to CSV.
    save_MCMC_trajectory( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None
        Save the Markov Chain Monte Carlo trajectory to CSV.
    save_all_trajectories( self, path: Optional[str] = ".", label: Optional[str] = "" ) -> None
        Save all trajectories to CSV.
    clear_gradient_ascent_trajectory( self ) -> None
        Clear the gradient ascent trajectory.
    clear_MC_trajectory( self ) -> None
        Clear the Monte Carlo trajectory.
    clear_MCMC_trajectory( self ) -> None
        Clear the Markov Chain Monte Carlo trajectory.
    clear_all_trajectories( self ) -> None
        Clear all trajectories.
    summary( self ) -> None
        Print a summary of the GBA model.
    create_figure( self, title: str ) -> go.Figure
        Create a figure for plotting.
    add_trajectory( self, fig: go.Figure, source: str, x_var: str, y_var: str, x_factor: Optional[float] = 1.0, y_factor: Optional[float] = 1.0, name: Optional[str] = "", data: Optional[pd.DataFrame] = None ) -> None
        Add a trajectory to a figure.
    """

    def __init__( self, name: str ) -> None:

        """
        Constructor of the GbaModel class.
        
        Parameters
        ----------
        name : str
            Name of the GBA model.
        """
        assert name != "", throw_message(MessageType.Error, "You must provide a name to the GbaModel constructor.")
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
        self.directions         = []           # Indicates the direction of the reaction
        self.conditions         = np.array([]) # List of conditions
        self.constant_rhs       = {}           # Constant right-hand side terms
        self.constant_reactions = {}           # Constant reactions

        ### Proteomics ###
        self.protein_contributions = {} # Protein contributions for each reaction
        self.proteomics            = {} # Predicted proteomics

        ### Loaded objects ###
        self.Mx_loaded                    = False # Is the mass fraction matrix loaded?
        self.kcat_loaded                  = False # Are the kcat constants loaded?
        self.KM_f_loaded                  = False # Are the KM forward constants loaded?
        self.KM_b_loaded                  = False # Are the KM backward constants loaded?
        self.KA_loaded                    = False # Are the KA constants loaded?
        self.KI_loaded                    = False # Are the KI constants loaded?
        self.conditions_loaded            = False # Are the conditions loaded?
        self.constant_rhs_loaded          = False # Are the constant right-hand side terms loaded?
        self.constant_reactions_loaded    = False # Are the constant reactions loaded?
        self.protein_contributions_loaded = False # Are the protein contributions loaded?
        self.LP_solution_loaded           = False # Is the LP solution loaded?

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

    def read_constant_rhs_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the list of constant RHS terms from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.constant_rhs_loaded = False
        filename                 = path+"/"+self.name+"/constant_rhs.csv"
        if os.path.exists(filename):
            f = open(filename, "r")
            l = f.readline()
            l = f.readline()
            self.constant_rhs.clear()
            while l:
                l = l.strip("\n").split(";")
                self.constant_rhs[l[0]] = float(l[1])
                l = f.readline()
            f.close()
            self.constant_rhs_loaded = True
    
    def read_constant_reactions_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the list of constant reactions from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.constant_reactions_loaded = False
        filename                       = path+"/"+self.name+"/constant_reactions.csv"
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

    def read_protein_contributions_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the list of protein contributions from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.protein_contributions_loaded = False
        filename                          = path+"/"+self.name+"/protein_contributions.csv"
        if os.path.exists(filename):
            f = open(filename, "r")
            l = f.readline()
            l = f.readline()
            self.protein_contributions.clear()
            while l:
                l = l.strip("\n").split(";")
                r_id = l[0]
                p_id = l[1]
                val  = float(l[2])
                if r_id not in self.protein_contributions:
                    self.protein_contributions[r_id] = {p_id: val}
                else:
                    self.protein_contributions[r_id][p_id] = val
                l = f.readline()
            f.close()
            self.protein_contributions_loaded = True
    
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
        assert self.Mx_loaded, throw_message(MessageType.Error, "Mass fraction matrix Mx not loaded.")
        assert self.kcat_loaded, throw_message(MessageType.Error, "kcat constants not loaded.")
        assert self.KM_f_loaded, throw_message(MessageType.Error, "KM forward constants not loaded.")
        assert self.conditions_loaded, throw_message(MessageType.Error, "Conditions not loaded.")
        if not self.KM_b_loaded:
            throw_message(MessageType.Info, "KM backward constants not loaded.")
        if not self.KA_loaded:
            throw_message(MessageType.Info, "KA constants not loaded.")
        if not self.KI_loaded:
            throw_message(MessageType.Info, "KI constants not loaded.")
        if not self.constant_rhs_loaded:
            throw_message(MessageType.Info, "Constant RHS terms not loaded.")
        if not self.constant_reactions_loaded:
            throw_message(MessageType.Info, "Constant reactions not loaded.")
        if not self.protein_contributions_loaded:
            throw_message(MessageType.Info, "Protein contributions not loaded.")
        if not self.LP_solution_loaded:
            throw_message(MessageType.Info, "LP solution not loaded.")
    
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
        self.directions.clear()
        for j in range(self.nj):
            if (self.kcat_b[j] == 0 and self.KI[:,j].sum() == 0 and self.KA[:,j].sum() == 0):
                self.kinetic_model.append(GbaReactionType.iMM)
                self.directions.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] == 0 and self.KI[:,j].sum() == 0 and self.KA[:,j].sum() > 0):
                self.kinetic_model.append(GbaReactionType.iMMa)
                self.directions.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] == 0 and self.KI[:,j].sum() > 0 and self.KA[:,j].sum() == 0):
                self.kinetic_model.append(GbaReactionType.iMMi)
                self.directions.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] == 0 and self.KI[:,j].sum() > 0 and self.KA[:,j].sum() > 0):
                self.kinetic_model.append(GbaReactionType.iMMia)
                self.directions.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] > 0):
                assert self.KA[:,j].sum() == 0, throw_message(MessageType.Error, f"Reversible Michaelis-Menten reaction cannot have activation (reaction <code>{j}</code>).")
                assert self.KI[:,j].sum() == 0, throw_message(MessageType.Error, f"Reversible Michaelis-Menten reaction cannot have inhibition (reaction <code>{j}</code>).")
                self.kinetic_model.append(GbaReactionType.rMM)
                self.directions.append(self.directions.append(ReactionDirection.Reversible))
    
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
        assert os.path.exists(path+"/"+self.name), throw_message(MessageType.Error, "Folder "+path+"/"+self.name+" does not exist.")
        self.read_Mx_from_csv(path)
        self.read_kcat_from_csv(path)
        self.read_KM_f_from_csv(path)
        self.read_KM_b_from_csv(path)
        self.read_KA_from_csv(path)
        self.read_KI_from_csv(path)
        self.read_conditions_from_csv(path)
        self.read_constant_rhs_from_csv(path)
        self.read_constant_reactions_from_csv(path)
        self.read_protein_contributions_from_csv(path)
        self.read_LP_from_csv(path)
        self.check_model_loading(verbose)
        self.initialize_model_mathematical_variables()

    def write_to_csv( self, path: Optional[str] = ".", verbose: Optional[bool] = False ) -> None:
        """
        Write the GBA model to CSV files.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV files.
        verbose : Optional[bool], default=False
            Verbose mode.        
        """
        assert os.path.exists(path), throw_message(MessageType.Error, f"The path <code>{path}</code> does not exist")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Check the existence of the folder #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        model_path = path+"/"+self.name
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            files = ["M.csv", "intM.csv", "kcat.csv", "KM_forward.csv", "KM_backward.csv", "KA.csv", "KI.csv",
                     "conditions.csv", "directions.csv", "constant_rhs.csv", "constant_reactions.csv",
                     "protein_contributions.csv"]
            for f in files:
                if os.path.exists(model_path+"/"+f):
                    os.system(f"rm {model_path}/{f}")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Write the mass fraction matrix    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        M_df = pd.DataFrame(self.Mx, index=self.metabolite_ids, columns=self.reaction_ids)
        M_df.to_csv(model_path+"/M.csv", sep=";")
        del(M_df)
        intM_df = pd.DataFrame(self.M, index=self.c_ids, columns=self.reaction_ids)
        intM_df.to_csv(model_path+"/intM.csv", sep=";")
        del(intM_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Write the kcat vectors            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        kcat_df = pd.DataFrame(self.kcat_f, index=self.reaction_ids, columns=["forward"])
        kcat_df["backward"] = self.kcat_b
        kcat_df = kcat_df.transpose()
        kcat_df.to_csv(model_path+"/kcat.csv", sep=";")
        del(kcat_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Write the forward KM matrices     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        KM_df = pd.DataFrame(self.KM_f, index=self.metabolite_ids, columns=self.reaction_ids)
        KM_df.to_csv(model_path+"/KM_forward.csv", sep=";")
        del(KM_df)
        KM_df = pd.DataFrame(self.KM_b, index=self.metabolite_ids, columns=self.reaction_ids)
        KM_df.to_csv(model_path+"/KM_backward.csv", sep=";")
        del(KM_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Write the KA and KI matrices      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        KA_df = pd.DataFrame(self.KA, index=self.metabolite_ids, columns=self.reaction_ids)
        KA_df.to_csv(model_path+"/KA.csv", sep=";")
        del(KA_df)
        KI_df = pd.DataFrame(self.KI, index=self.metabolite_ids, columns=self.reaction_ids)
        KI_df.to_csv(model_path+"/KI.csv", sep=";")
        del(KI_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Write the conditions              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        conditions_df = pd.DataFrame(self.conditions, index=self.condition_params, columns=self.condition_ids)
        conditions_df.to_csv(model_path+"/conditions.csv", sep=";")
        del(conditions_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 7) Write the directions              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open(model_path+"/directions.csv", "w")
        f.write("reaction;direction\n")
        for j in range(len(self.reaction_ids)):
            if self.kcat_b[j] == 0.0 and self.kcat_f[j] > 0.0:
                f.write(self.reaction_ids[j]+";forward\n")
            elif self.kcat_b[j] > 0.0 and self.kcat_f[j] == 0.0:
                f.write(self.reaction_ids[j]+";backward\n")
            elif self.kcat_b[j] > 0.0 and self.kcat_f[j] > 0.0:
                f.write(self.reaction_ids[j]+";reversible\n")
            else:
                throw_message(MessageType.Error, f"Unknown direction for reaction <code>{self.reaction_ids[j]}</code>.")
                sys.exit(1)
        f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 8) Write the constant RHS terms      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open(model_path+"/constant_rhs.csv", "w")
        f.write("metabolite;value\n")
        for item in self.constant_rhs.items():
            f.write(item[0]+";"+str(item[1])+"\n")
        f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 9) Write the constant reactions      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open(model_path+"/constant_reactions.csv", "w")
        f.write("reaction;value\n")
        for item in self.constant_reactions.items():
            f.write(item[0]+";"+str(item[1])+"\n")
        f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 10) Save protein contributions       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open(model_path+"/protein_contributions.csv", "w")
        f.write("reaction;protein;contribution\n")
        for item in self.protein_contributions.items():
            r_id = item[0]
            for p_id, val in item[1].items():
                f.write(r_id+";"+p_id+";"+str(val)+"\n")
        f.close()

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
        assert condition_id in self.condition_ids, throw_message(MessageType.Error, f"Unknown condition identifier <code>{condition_id}</code>.")
        assert condition_param in self.condition_params, throw_message(MessageType.Error, f"Unknown condition parameter <code>{condition_param}</code>.")
        i = self.condition_params.index(condition_param)
        j = self.condition_ids.index(condition_id)
        return self.conditions[i,j]

    def get_vector( self, source: str, variable: str ) -> np.array:
        """
        Get the vector of a variable in a data-frame.

        Parameters
        ----------
        source : str
            Source of the vector.
        variable : str
            Variable name.

        Returns
        -------
        np.array
            Vector of the variable.
        """
        assert source in ["random", "optima", "GA", "MC", "MCMC"], throw_message(MessageType.Error, "Source must be <code>random</code>, <code>optima</code>, <code>GA</code>, <code>MC</code> or <code>MCMC</code>.")
        if source == "random":
            assert not self.random_data.empty, throw_message(MessageType.Error, "No data available for random solutions.")
            return self.random_data[variable].values
        elif source == "optima":
            assert not self.optima_data.empty, throw_message(MessageType.Error, "No data available for optima.")
            return self.optima_data[variable].values
        elif source == "GA":
            assert not self.GA_tracker.empty, throw_message(MessageType.Error, "No data available for gradient ascent.")
            return self.GA_tracker[variable].values
        elif source == "MC":
            assert not self.MC_tracker.empty, throw_message(MessageType.Error, "No data available for Monte Carlo.")
            return self.MC_tracker[variable].values
        elif source == "MCMC":
            assert not self.MCMC_tracker.empty, throw_message(MessageType.Error, "No data available for MCMC.")
            return self.MCMC_tracker[variable].values
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 3) Setters                         #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def clear_conditions( self ) -> None:
        """
        Clear all external conditions from the GBA model.
        """
        self.condition_ids    = []
        self.condition_params = ["rho"] + self.x_ids
        self.conditions       = np.array([])
    
    def add_condition( self, condition_id: str, rho: float, default_concentration: Optional[float] = 1.0, metabolites: Optional[dict[str, float]] = None ) -> None:
        """
        Add an external condition to the GBA model.

        Parameters
        ----------
        condition_id : str
            Identifier of the condition.
        rho : float
            Total density of the cell (g/L).
        default_concentration : float
            Default concentration of metabolites (g/L).
        metabolites : dict[str, float]
            Dictionary of metabolite concentrations (g/L).
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Assertions                             #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        assert condition_id not in self.condition_ids, throw_message(MessageType.Error, f"Condition <code>{condition_id}</code> already exists.")
        assert rho > 0.0, throw_message(MessageType.Error, "The total density must be positive.")
        assert default_concentration >= 0.0, throw_message(MessageType.Error, "The default concentration must be positive.")
        if metabolites is not None:
            for m_id, concentration in metabolites.items():
                assert m_id in self.metabolite_ids, throw_message(MessageType.Error, f"Metabolite <code>{m_id}</code> does not exist.")
                assert m_id in self.condition_params, throw_message(MessageType.Error, f"Metabolite <code>{m_id}</code> is not a condition parameter.")
                assert concentration >= 0.0, throw_message(MessageType.Error, f"The concentration of metabolite <code>{m_id}</code> must be positive.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Set the condition                      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        vec = [rho]
        if metabolites is None:
            vec = vec + [default_concentration]*self.nx
        else:
            for x_id in self.x_ids:
                if x_id in metabolites:
                    vec.append(metabolites[x_id])
                else:
                    vec.append(default_concentration)
        self.condition_ids.append(condition_id)
        # add vec as a new column
        self.conditions = np.column_stack([self.conditions, np.array(vec)]) if self.conditions.size else np.array(vec)
    
    def clear_constant_rhs( self ) -> None:
        """
        Clear all constant RHS terms from the GBA model.
        """
        self.constant_rhs = {}
    
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
        assert metabolite_id in self.metabolite_ids, throw_message(MessageType.Error, f"Unknown metabolite identifier <code>{metabolite_id}</code>.")
        assert value >= 0.0, throw_message(MessageType.Error, "The constant value must be positive.")
        self.constant_rhs[metabolite_id] = value
    
    def clear_constant_reactions( self ) -> None:
        """
        Clear all constant reactions from the GBA model.
        """
        self.constant_reactions = {}
    
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
        assert reaction_id in self.reaction_ids, throw_message(MessageType.Error, f"Unknown reaction identifier <code>{reaction_id}</code>.")
        self.constant_reactions[reaction_id] = value
    
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
        assert condition_id in self.condition_ids, throw_message(MessageType.Error, "Unknown condition identifier <code>{condition_id}</code>.")
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
        assert len(f0) == self.nj, throw_message(MessageType.Error, "Incorrect f0 length.")
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

    def solve_local_linear_problem( self, max_flux_fraction: Optional[float] = 10.0, rhs_factor: Optional[float] = 10.0 ) -> None:
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
        max_flux_fraction : Optional[float], default=10.0
            Maximal flux fraction.
        rhs_factor : Optional[float], default=100.0
            Factor dividing the rhs of the mass conservation constraint.
        """
        assert max_flux_fraction > GbaConstants.MIN_FLUX_FRACTION.value, throw_message(MessageType.Error, f"Maximal flux fraction must be greater than {GbaConstants.MIN_FLUX_FRACTION.value}.")
        assert rhs_factor > 0.0, throw_message(MessageType.Error, "RHS factor must be positive.")
        lb_vec = [GbaConstants.MIN_FLUX_FRACTION.value]*self.nj
        ub_vec = [max_flux_fraction]*self.nj
        for item in self.constant_reactions.items():
           r_index         = self.reaction_ids.index(item[0])
           lb_vec[r_index] = item[1]
           ub_vec[r_index] = item[1]
        gpmodel = gp.Model(env=env)
        v       = gpmodel.addMVar(self.nj, lb=lb_vec, ub=ub_vec)
        min_b   = 1/self.nc/rhs_factor
        rhs     = np.repeat(min_b, self.nc)
        for m_id, value in self.constant_rhs.items():
            rhs[self.c_ids.index(m_id)] = value
        gpmodel.setObjective(v[-1], gp.GRB.MAXIMIZE)
        gpmodel.addConstr(self.M @ v >= rhs, name="c1")
        gpmodel.addConstr(self.sM @ v == 1, name="c2")
        gpmodel.optimize()
        try:
            self.LP_solution = np.copy(v.X)
            return True
        except:
            throw_message(MessageType.Error, "Local linear problem could not be solved.")
            return False

    def generate_LP_initial_solution( self, max_flux_fraction: Optional[float] = 10.0, rhs_factor: Optional[float] = 10.0,
                                      condition_id: Optional[str] = "1", save_f0: Optional[str] = None ) -> None:
        """
        Generate an initial solution using a linear program.

        Parameters
        ----------
        max_flux_fraction : Optional[float], default=10.0
            Maximal flux fraction.
        rhs_factor : Optional[float], default=10.0
            Factor dividing the rhs of the mass conservation constraint.
        condition_id : Optional[str], default="1"
            Condition identifier.
        save_f0 : Optional[str], default=None
            Path to save the initial solution.
        """
        solved = self.solve_local_linear_problem(max_flux_fraction=max_flux_fraction, rhs_factor=rhs_factor)
        if solved:
            self.set_condition(condition_id)
            self.set_f0(self.LP_solution)
            self.calculate_state()
            self.check_model_consistency()
            if self.consistent:
                throw_message(MessageType.Info, f"Model is consistent with mu = {self.mu}.")
                if save_f0 is not None:
                    self.save_f0(path=save_f0)
            else:
                throw_message(MessageType.Info, "Model is inconsistent.")
        else:
            throw_message(MessageType.Warning, "Impossible to find an initial solution.")

    def generate_random_initial_solutions( self, condition_id: str, nb_solutions: int, max_trials: int, max_flux_fraction: Optional[float] = 10.0, min_mu: Optional[float] = 1e-3, verbose: Optional[bool] = False ) -> None:
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
        max_flux_fraction : Optional[float], default=10.0
            Maximal flux fraction.
        min_mu : Optional[float], default=1e-3
            Minimal growth rate.
        verbose : Optional[bool], default=False
            Verbose mode.
        """
        assert condition_id in self.condition_ids, throw_message(MessageType.Error, f"Unknown condition identifier (<code>{condition_id}</code>).")
        assert nb_solutions > 0, throw_message(MessageType.Error, f"Number of solutions must be greater than 0.")
        assert max_trials >= nb_solutions, throw_message(MessageType.Error, f"Number of trials must be greater than the number of solutions.")
        assert max_flux_fraction > GbaConstants.MIN_FLUX_FRACTION.value, throw_message(MessageType.Error, f"Maximal flux fraction must be greater than {GbaConstants.MIN_FLUX_FRACTION.value}.")
        assert min_mu >= 0.0, throw_message(MessageType.Error, f"Minimal growth rate must be positive.")
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
                self.f_trunc = self.f_trunc*(max_flux_fraction-GbaConstants.MIN_FLUX_FRACTION)+GbaConstants.MIN_FLUX_FRACTION
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
                    throw_message(MessageType.Plain, f"{solutions} solutions were found after {trials} trials (last mu = {round(self.mu,5)}).")

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
        assert index < self.nj, throw_message(MessageType.Error, f"Index <code>{index}</code> is out of bounds.")
        assert sigma > 0.0, throw_message(MessageType.Error, f"Sigma must be positive.")
        non_mutated_f        = np.copy(self.f_trunc)
        epsilon              = np.random.normal(0.0, sigma, size=1)
        self.f_trunc[index] += epsilon
        self.block_reactions(block_GCC=False)
        #self.f_trunc[self.f_trunc < GbaConstants.MIN_FLUX_FRACTION] = GbaConstants.MIN_FLUX_FRACTION
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
            assert variable in allowed_variables, throw_message(MessageType.Error, f"Variable <code>{variable}</code> is not allowed.")
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

    def block_reactions( self, block_GCC: Optional[bool] = True ) -> None:
        """
        Block reactions tending to zero.

        Description
        -----------
        f values tending towards zero are bounded to min value.
        Corresponding derivative values are set to zero depending
        on the direction:
        - f -> 0+ and gcc < 0: f = min_f, gcc = 0
        - f -> 0- and gcc > 0: f = -min_f, gcc = 0

        Parameters
        ----------
        block_GCC : Optional[bool], default=True
            Block the GCC values.
        """
        for j in range(self.nj-1):
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 1) Reaction is irreversible and positive #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            if self.directions[j+1] == ReactionDirection.Forward and self.f_trunc[j] <= GbaConstants.MIN_FLUX_FRACTION.value:
                self.f_trunc[j] = GbaConstants.MIN_FLUX_FRACTION.value
                if block_GCC and self.GCC_f[(j+1)] < 0.0:
                    self.GCC_f[(j+1)] = 0.0
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 2) Reaction is irreversible and negative #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            elif self.directions[j+1] == ReactionDirection.Backward and self.f_trunc[j] >= -GbaConstants.MIN_FLUX_FRACTION.value:
                self.f_trunc[j] = -GbaConstants.MIN_FLUX_FRACTION.value
                if block_GCC and self.GCC_f[(j+1)] > 0.0:
                    self.GCC_f[(j+1)] = 0.0
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 3) Reaction is reversible                #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # elif self.directions[j+1] == ReactionDirection.Reversible and np.abs(self.f_trunc[j]) <= GbaConstants.MIN_FLUX_FRACTION.value:
            #     if block_GCC:
            #         self.GCC_f[(j+1)] = 0.0
            #     if self.f_trunc[j] >= 0.0:
            #         self.f_trunc[j] = GbaConstants.MIN_FLUX_FRACTION.value
            #     elif self.f_trunc[j] < 0.0:
            #         self.f_trunc[j] = -GbaConstants.MIN_FLUX_FRACTION.value
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 4) Reaction is constant                  #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            if self.reaction_ids[(j+1)] in self.constant_reactions:
                self.f_trunc[j] = self.constant_reactions[self.reaction_ids[(j+1)]]
                if block_GCC:
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
        assert self.consistent, throw_message(MessageType.Error, "Initial model is not consistent.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize the tracker   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if track:
            if self.GA_tracker.empty:
                columns = ["label", "condition", "iter", "dt", "t", "mu", "fixed"]
                self.GA_tracker = pd.DataFrame(columns=columns)
            data_dict = {"label": label, "condition": condition_id, "iter": 0, "dt": initial_dt, "t": 0.0, "mu": self.mu, "fixed": 0}
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
               throw_message(MessageType.Plain, f"Iteration: {nb_iterations} (time = {t}, mu = {self.mu}, dt = {dt}).")
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
                    data_dict = {"label": label, "condition": condition_id, "iter": nb_iterations, "dt": dt, "t": t, "mu": self.mu, "fixed": nb_fixed}
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
                assert self.consistent, throw_message(MessageType.Error, "Previous model is not consistent.")
                if (dt > GbaConstants.MIN_DT):
                    dt         = dt/GbaConstants.DECREASING_DT_FACTOR.value
                    dt_counter = 0
                else:
                    throw_message(MessageType.Error, f"Adaptative timestep < {GbaConstants.MIN_DT}.")
                    sys.exit(1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if t >= max_time:
            if verbose:
                throw_message(MessageType.Plain, f"Gradient ascent: maximum time reached (condition={condition_id}, mu={round(self.mu, 5)}, nb iterations={nb_iterations}, nb fixed={nb_fixed}).")
            return False, run_time
        else:
            if verbose:
                throw_message(MessageType.Plain, f"Gradient ascent: convergence reached (condition={condition_id}, mu={round(self.mu, 5)}, nb iterations={nb_iterations}, nb fixed={nb_fixed}).")
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
            throw_message(MessageType.Plain, f"All optima were computed in {run_time} seconds.")
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
        assert self.consistent, throw_message(MessageType.Error, "Initial model is not consistent.")
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
               throw_message(MessageType.Plain, f"Iteration: {nb_iterations} (time = {t}, mu = {self.mu}, fixed = {nb_fixed}).")
            if nb_iterations >= max_iterations:
                if verbose:
                    throw_message(MessageType.Plain, f"Maximum number of iterations reached (condition <code>{condition_id}</code>).")
                break
            ### 4.1) Calculate the next step ###
            self.block_reactions()
            epsilon      = np.random.normal(0.0, np.sqrt(sigma/(2.0*N_e)), size=self.nj-1)
            self.f_trunc = self.f_trunc+sigma*self.GCC_f[1:]+epsilon
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
                assert self.consistent, throw_message(MessageType.Error, "Previous model is not consistent.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if nb_fixed == 0 and nb_iterations < max_iterations:
            if verbose:
                throw_message(MessageType.Plain, f"MC simulation completed with no fixed mutation (condition <code>{condition_id}</code>, mu = {round(self.mu, 5)}, nb iterations = {nb_iterations}, nb fixed = {nb_fixed}).")
            return False, run_time
        elif nb_fixed > 0 and nb_iterations < max_iterations:
            if verbose:
                throw_message(MessageType.Plain, f"MC simulation completed (condition <code>{condition_id}</code>, mu = {round(self.mu, 5)}, nb iterations = {nb_iterations}, nb fixed = {nb_fixed}).")
            return True, run_time
        elif nb_iterations >= max_iterations:
            if verbose:
                throw_message(MessageType.Plain, f"MC simulation: maximum iterations reached (condition <code>{condition_id}</code>, mu = {round(self.mu, 5)}, nb iterations = {nb_iterations}, nb fixed = {nb_fixed}).")
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
        assert self.consistent, throw_message(MessageType.Error, "Initial model is not consistent.")
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
                throw_message(MessageType.Plain, f"Iteration: {nb_iterations} (mu = {self.mu}, fixed = {nb_fixed}).")
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
            if verbose:
                throw_message(MessageType.Plain, f"MCMC simulation completed with no fixed mutation (condition={condition_id}, mu={round(self.mu, 5)}, nb iterations={nb_iterations}, nb fixed={nb_fixed}).")
            return False, run_time, nb_fixed
        elif nb_fixed > 0 and nb_iterations < max_iterations:
            if verbose:
                throw_message(MessageType.Plain, f"MCMC simulation completed (condition={condition_id}, mu={round(self.mu, 5)}, nb iterations={nb_iterations}, nb fixed={nb_fixed}).")
            return True, run_time, nb_fixed
        elif nb_iterations >= max_iterations:
            if verbose:
                throw_message(MessageType.Plain, f"MCMC simulation: maximum iterations reached (condition={condition_id}, mu={round(self.mu, 5)}, nb iterations={nb_iterations}, nb fixed={nb_fixed}).")
            return False, run_time, nb_fixed

    def save_f0( self, path: Optional[str] = "." ) -> None:
        """
        Save the initial flux state to CSV.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to save the initial flux state.
        """
        assert os.path.exists(path+"/"+self.name), throw_message(MessageType.Error, f"Path <code>{path}/{self.name}</code> does not exist.")
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
        assert os.path.exists(path), throw_message(MessageType.Error, f"Path <code>{path}</code> does not exist.")
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
        assert os.path.exists(path), throw_message(MessageType.Error, f"Path <code>{path}</code> does not exist.")
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
        assert os.path.exists(path), throw_message(MessageType.Error, f"Path <code>{path}</code> does not exist.")
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
        assert os.path.exists(path), throw_message(MessageType.Error, f"Path <code>{path}</code> does not exist.")
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
        assert os.path.exists(path), throw_message(MessageType.Error, f"Path <code>{path}</code> does not exist.")
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
        assert os.path.exists(path), throw_message(MessageType.Error, f"Path <code>{path}</code> does not exist.")
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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 8) Graphic generation      #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def create_figure( self, title: str ) -> go.Figure:
        """
        Create a figure for plotting.

        Parameters
        ----------
        title : str
            Title of the figure.

        Returns
        -------
        go.Figure
            Plotly figure.
        """
        fig = go.Figure()
        fig.update_layout(plot_bgcolor='white', template="plotly_white", title=dict(text=title, x=0.5))
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        return fig
    
    def add_trajectory( self, fig: go.Figure, source: str, x_var: str, y_var: str,
                        x_factor: Optional[float] = 1.0, y_factor: Optional[float] = 1.0,
                        name: Optional[str] = "", data: Optional[pd.DataFrame] = None ) -> None:
        """
        Add a trajectory to a figure.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure.
        source : str
            Source of the trajectory (random, optima, GA, MC, MCMC).
        x_var : str
            X-axis variable.
        y_var : str
            Y-axis variable.
        x_factor : Optional[float], default=1.0
            X-axis factor.
        y_factor : Optional[float], default=1.0
            Y-axis factor.
        name : Optional[str], default=""
            Name of the trajectory.
        data : Optional[pd.DataFrame], default=None
            Data for the trajectory.
        """
        assert source in ["random", "optima", "GA", "MC", "MCMC", "data"], throw_message(MessageType.Error, "Source must be <code>random</code>, <code>optima</code>, <code>GA</code>, <code>MC</code> or <code>MCMC</code> or <code>data</code>.")
        if source == "data":
            assert data is not None, throw_message(MessageType.Error, "Data must be provided for source <code>data</code>.")
        X = None
        Y = None
        if data is not None:
            X = data[x_var].values*x_factor
            Y = data[y_var].values*y_factor
        else:
            X = self.get_vector(source, x_var)*x_factor
            Y = self.get_vector(source, y_var)*y_factor
        fig.add_trace(go.Scatter(x=X, y=Y, mode="lines", name=name))

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
    assert os.path.exists(path+"/"+name), throw_message(MessageType.Error, "The folder "+path+"/"+name+" does not exist.")
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
    assert os.path.isfile(filename), throw_message(MessageType.Error, ".gba file creation failed.")

def load_gba_model( path: str ) -> GbaModel:
    """
    Load a GBA model from a binary file.

    Parameters
    ----------
    path : str
        Path to the GBA model file.
    """
    assert path.endswith(".gba"), throw_message(MessageType.Error, "GBA model file extension is missing.")
    assert os.path.isfile(path), throw_message(MessageType.Error, "GBA model file not found.")
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
        throw_message(MessageType.Plain, f"Computing LP solution for model {model.name}...")
        model.solve_local_linear_problem()
        model.set_f0(model.LP_solution)
        model.set_condition("1")
        model.calculate_state()
        model.check_model_consistency()
        if model.consistent:
            model.save_f0(path=path)
        else:
            throw_message(MessageType.Error, "Model is inconsistent with condition 1. f0 vector cannot be saved.")
            sys.exit(1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 3) Compute and save optima if requested     #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if save_optima:
        throw_message(MessageType.Plain, f"Computing optima for model {model.name}...")
        if not save_LP:
            model.read_LP_from_csv(path=path)
        model.compute_optima(max_time=10000, initial_dt=0.01)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 4) Clean model and dump binary backup       #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    model.reset_variables()
    backup_gba_model(model=model, name=name, path=gba_path)
    del model

