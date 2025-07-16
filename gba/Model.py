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
Filename: Model.py
Author: Charles Rocabert, Furkan Mert
Date: 2024-10-22
Description:
    Model class of the gbapy module.
License: MIT License
Copyright: © 2024-2025 Charles Rocabert, Furkan Mert. All rights reserved.
"""

import os
import sys
import csv
import time
import pickle
import pkgutil
import numpy as np
import pandas as pd
import gurobipy as gp
from pathlib import Path
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
from pyexcel_xlsx import get_data
from pyexcel_ods3 import save_data
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


class Model:
    """
    Class to manipulate cell growth models (CGMs).

    Attributes
    ----------
    name : str
        Name of the model.
    info : str
        Info about the model.
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
    K: np.array
        Complete K matrix.
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
    KR : np.array
        KR matrix.
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
    K_loaded : bool
        Are the KM constants loaded?
    KA_loaded : bool
        Are the KA constants loaded?
    KI_loaded : bool
        Are the KI constants loaded?
    KR_loaded : bool
        Are the KR constants loaded?
    conditions_loaded : bool
        Are the conditions loaded?
    constant_rhs_loaded : bool
        Are the constant right-hand side terms loaded?
    constant_reactions_loaded : bool
        Are the constant reactions loaded?
    protein_contributions_loaded : bool
        Are the protein contributions loaded?
    initial_solution_loaded : bool
        Is the initial solution loaded?
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
    initial_solution : np.array
        Initial solution.
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
    doubling_time : float
        Doubling time.
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
    read_Info_from_csv( self, path: Optional[str] = "." ) -> None
        Read the model information from a CSV file.
    read_Mx_from_csv( path: Optional[str] = "." ) -> None
        Read the mass fraction matrix M from a CSV file.
    read_kcat_from_csv( path: Optional[str] = "." ) -> None
        Read the kcat constant vectors from a CSV file.
    read_K_from_csv( path: Optional[str] = "." ) -> None
        Read the Michaelis constant matrix K from a CSV file.
    read_KA_from_csv( path: Optional[str] = "." ) -> None
        Read the activation constants matrix KA from a CSV file.
    read_KI_from_csv( path: Optional[str] = "." ) -> None
        Read the inhibition constants matrix KI from a CSV file.
    read_KR_from_csv( path: Optional[str] = "." ) -> None
        Read the regulation constants matrix KR from a CSV file.
    read_conditions_from_csv( path: Optional[str] = "." ) -> None
        Read the list of conditions from a CSV file.
    read_constant_rhs_from_csv( path: Optional[str] = "." ) -> None
        Read the list of constant RHS terms from a CSV file.
    read_constant_reactions_from_csv( path: Optional[str] = "." ) -> None
        Read the list of constant reactions from a CSV file.
    read_protein_contributions_from_csv( path: Optional[str] = "." ) -> None
        Read the list of protein contributions from a CSV file.
    read_initial_solution_from_csv( path: Optional[str] = "." ) -> None
        Read the initial solution from a CSV file (on request).
    check_model_loading( verbose: Optional[bool] = False ) -> None
        Check if the model is loaded correctly.
    initialize_model_mathematical_variables( ) -> None
        Initialize the model mathematical variables.
    read_from_csv( path: Optional[str] = ".", verbose: Optional[bool] = False ) -> None
        Read the CGM from CSV files.
    write_to_csv( path: Optional[str] = ".", verbose: Optional[bool] = False ) -> None
        Write the CGM to CSV files.
    get_condition( self, condition_id: str, condition_param: str ) -> float
        Get the value of a condition parameter.
    get_vector( self, source: str, variable: str ) -> np.array
        Get the value of a variable from a source.
    clear_conditions( self ) -> None
        Clear all external conditions from the CGM.
    add_condition( self, condition_id: str, rho: float, default_concentration: Optional[float] = 1.0, metabolites: Optional[dict[str, float]] = None ) -> None
        Add a new condition to the CGM.
    clear_constant_rhs( self ) -> None
        Clear all constant right-hand side terms from the CGM.
    add_constant_rhs( self, metabolite_id: str, value: float ) -> None
        Add a new constant right-hand side term to the CGM.
    clear_constant_reactions( self ) -> None
        Clear all constant reactions from the CGM.
    add_constant_reaction( self, reaction_id: str, value: float ) -> None
        Add a new constant reaction to the CGM.
    reset_variables( self ) -> None
        Reset all variables of the CGM.
    set_condition( self, condition_id: str ) -> None
        Set the current condition of the CGM.
    set_f0( self, f0: np.array ) -> None
        Set the initial solution f0 of the CGM.
    set_f( self ) -> None
        Set the flux fractions vector of the CGM.
    gaussian_kernel( self, x: np.array, mu: float ) -> np.array
        Compute the Gaussian kernel for a vector x with mean mu.
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
    calculate_first_order_terms( self ) -> None
        Calculate first order terms of the model state.
    calculate_second_order_terms( self ) -> None
        Calculate second order terms of the model state.
    calculate( self ) -> None
        Calculate the model state.
    check_model_consistency( self ) -> None
        Check the model state's consistency.
    solve_local_linear_problem( self,max_flux_fraction: Optional[float] = 10.0, rhs_factor: Optional[float] = 10.0 ) -> None
        Solve the local linear problem to find the initial solution.
    find_initial_solution( self, max_flux_fraction: Optional[float] = 10.0, rhs_factor: Optional[float] = 10.0, condition_id: Optional[str] = "1", save_f0: Optional[str] = None ) -> None
        Generate an initial solution using a linear program.
    generate_random_initial_solutions( self, condition_id: str, nb_solutions: int, max_trials: int, max_flux_fraction: Optional[float] = 10.0, min_mu: Optional[float] = 1e-3, verbose: Optional[bool] = False ) -> None
        Generate random initial solutions.
    information( self ) -> None
        Print some informations about the CGM.
    summary( self ) -> None
        Print a summary of the CGM.
    """

    def __init__( self, name: str ) -> None:
        """
        Constructor of the Model class.
        
        Parameters
        ----------
        name : str
            Name of the CGM.
        """
        assert name != "", throw_message(MessageType.Error, "You must provide a name to the CGM constructor.")
        self.name = name
        self.info = {}

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) CGM                           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        ### Identifier lists ###
        self.metabolite_ids   = []
        self.x_ids            = []
        self.c_ids            = []
        self.reaction_ids     = []
        self.condition_ids    = []
        self.condition_params = []

        ### Model structure ###
        self.Mx                 = np.array([])
        self.M                  = np.array([])
        self.kcat_f             = np.array([])
        self.kcat_b             = np.array([])
        self.K                  = np.array([])
        self.KM_f               = np.array([])
        self.KM_b               = np.array([])
        self.KA                 = np.array([])
        self.KI                 = np.array([])
        self.rKI                = np.array([])
        self.KR                 = np.array([])
        self.reversible         = []
        self.kinetic_model      = []
        self.directions         = []
        self.conditions         = np.array([])
        self.constant_rhs       = {}
        self.constant_reactions = {}

        ### Proteomics ###
        self.protein_contributions = {}
        self.proteomics            = {}

        ### Loaded objects ###
        self.Info_loaded                  = False
        self.Mx_loaded                    = False
        self.kcat_loaded                  = False
        self.K_loaded                     = False
        self.KA_loaded                    = False
        self.KI_loaded                    = False
        self.KR_loaded                    = False
        self.conditions_loaded            = False
        self.constant_rhs_loaded          = False
        self.constant_reactions_loaded    = False
        self.protein_contributions_loaded = False
        self.initial_solution_loaded      = False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) CGM constants                 #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        ### Vector lengths ###
        self.nx = 0
        self.nc = 0
        self.ni = 0
        self.nj = 0

        ### Indices for reactions: s (transport), e (enzymatic), and ribosome r ###
        self.sM = []
        self.e  = []
        self.s  = []
        self.r  = 0
        self.ne = 0
        self.ns = 0

        ### Indices: m (metabolite), a (all proteins) ###
        self.m = []
        self.a = 0

        ### Matrix column rank ###
        self.column_rank      = 0
        self.full_column_rank = False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Solutions                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.initial_solution  = np.array([])
        self.optimal_solutions = {}
        self.random_solutions  = {}
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) CGM variables                 #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.tau_j                 = np.array([])
        self.ditau_j               = np.array([])
        self.x                     = np.array([])
        self.c                     = np.array([])
        self.xc                    = np.array([])
        self.v                     = np.array([])
        self.p                     = np.array([])
        self.b                     = np.array([])
        self.density               = 0.0
        self.mu                    = 0.0
        self.doubling_time         = 0.0
        self.consistent            = False
        self.adjust_concentrations = False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) CGM dynamical variables       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.condition = ""
        self.rho       = 0.0
        self.f0        = np.array([])
        self.dmu_f     = np.array([])
        self.GCC_f     = np.array([])
        self.f_trunc   = np.array([])
        self.f         = np.array([])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Trackers                      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.random_data  = pd.DataFrame()
        self.optima_data  = pd.DataFrame()
        self.GA_tracker   = pd.DataFrame()
        self.MC_tracker   = pd.DataFrame()
        self.MCMC_tracker = pd.DataFrame()
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Model loading methods           #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def read_Info_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the model information from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.Info_loaded = False
        filename         = path+"/"+self.name+"/Info.csv"
        if os.path.exists(filename):
            f = open(path+"/"+self.name+"/info.csv", "r")
            for line in f:
                parts = line.strip().split(";")
                parts += [""] * (3 - len(parts))  # pad to ensure 3 elements
                category, key, content = parts[:3]
                if category:
                    current_category = category.strip()
                    self.info[current_category] = {}
                elif key and current_category:
                    self.info[current_category][key.strip()] = content.strip()
            f.close()
            self.Info_loaded = True
    
    def read_Mx_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the mass fraction matrix M from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        filename       = path+"/"+self.name+"/M.csv"
        assert os.path.exists(filename), throw_message(MessageType.Error, "The file M.csv does not exist in the specified path: "+filename)
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
        assert os.path.exists(filename), throw_message(MessageType.Error, "The file kcat.csv does not exist in the specified path: "+filename)
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

    def read_K_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the Michaelis constant matrix K from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.K_loaded = False
        filename       = path+"/"+self.name+"/K.csv"
        assert os.path.exists(filename), throw_message(MessageType.Error, "The file K.csv does not exist in the specified path: "+filename)
        df            = pd.read_csv(filename, sep=";")
        df            = df.drop(["Unnamed: 0"], axis=1)
        df.index      = self.metabolite_ids
        self.K        = np.array(df)
        self.K        = self.K.astype(float)
        self.K_loaded = True
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

    def read_KR_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the regulation constants matrix KR from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.KR_loaded = False
        self.KR        = np.zeros(self.Mx.shape)
        filename       = path+"/"+self.name+"/KR.csv"
        if os.path.exists(filename):
            df             = pd.read_csv(filename, sep=";")
            metabolites    = list(df["Unnamed: 0"])
            df             = df.drop(["Unnamed: 0"], axis=1)
            df.index       = metabolites
            self.KR        = np.array(df)
            self.KR        = self.KR.astype(float)
            self.KR_loaded = True
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
        assert os.path.exists(filename), throw_message(MessageType.Error, "The file conditions.csv does not exist in the specified path: "+filename)
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
    
    def read_initial_solution_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the initial solution from a CSV file (on request).

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.initial_solution_loaded = False
        filename                     = path+"/"+self.name+"/f0.csv"
        if os.path.exists(filename):
            df                           = pd.read_csv(filename, sep=";")
            self.initial_solution        = np.array(df["f0"])
            self.initial_solution_loaded = True
            del(df)

    def check_model_loading( self, verbose: Optional[bool] = False ) -> None:
        """
        Check if the model is loaded correctly.

        Parameters
        ----------
        verbose : Optional[bool], default=False
            Print the error messages.
        """
        if not self.Info_loaded and verbose:
            throw_message(MessageType.Info, "Model information is missing.")
        if not self.KA_loaded and verbose:
            throw_message(MessageType.Info, "No KA constants.")
        if not self.KI_loaded and verbose:
            throw_message(MessageType.Info, "No KI constants.")
        if not self.KR_loaded and verbose:
            throw_message(MessageType.Info, "No KR constants.")
        if not self.constant_rhs_loaded and verbose:
            throw_message(MessageType.Info, "No constant RHS terms.")
        if not self.constant_reactions_loaded and verbose:
            throw_message(MessageType.Info, "No constant reactions.")
        if not self.protein_contributions_loaded and verbose:
            throw_message(MessageType.Info, "Protein contributions are missing.")
        if not self.initial_solution_loaded and verbose:
            throw_message(MessageType.Info, "The initial solution is missing.")
    
    def initialize_model_mathematical_variables( self ) -> None:
        """
        Initialize the model mathematical variables.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Forward and backward KM matrices                    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.KM_f = np.zeros(self.Mx.shape)
        self.KM_b = np.zeros(self.Mx.shape)
        for i in range(self.Mx.shape[0]):
            for j in range(self.Mx.shape[1]):
                if self.Mx[i,j] < 0:
                    self.KM_f[i,j] = self.K[i,j]
                elif self.Mx[i,j] > 0:
                    self.KM_b[i,j] = self.K[i,j]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Inverse of KI                                       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        with np.errstate(divide='ignore'):
            self.rKI                     = 1.0/self.KI
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
        # 7) CGM dynamical variables                             #
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
            if (self.kcat_b[j] == 0 and self.KA[:,j].sum() == 0 and self.KI[:,j].sum() == 0 and self.KR[:,j].sum() == 0):
                self.kinetic_model.append(CgmReactionType.iMM)
                self.directions.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] == 0 and self.KA[:,j].sum() > 0 and self.KI[:,j].sum() == 0 and self.KR[:,j].sum() == 0):
                self.kinetic_model.append(CgmReactionType.iMMa)
                self.directions.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] == 0 and self.KA[:,j].sum() == 0 and self.KI[:,j].sum() > 0 and self.KR[:,j].sum() == 0):
                self.kinetic_model.append(CgmReactionType.iMMi)
                self.directions.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] == 0 and self.KA[:,j].sum() > 0 and self.KI[:,j].sum() > 0 and self.KR[:,j].sum() == 0):
                self.kinetic_model.append(CgmReactionType.iMMia)
            elif (self.kcat_b[j] == 0 and self.KA[:,j].sum() == 0 and self.KI[:,j].sum() == 0 and self.KR[:,j].sum() > 0):
                self.kinetic_model.append(CgmReactionType.iMMr)
                self.directions.append(ReactionDirection.Forward)
            elif (self.kcat_b[j] > 0):
                assert self.KA[:,j].sum() == 0, throw_message(MessageType.Error, f"Reversible Michaelis-Menten reaction cannot have activation (reaction <code>{j}</code>).")
                assert self.KI[:,j].sum() == 0, throw_message(MessageType.Error, f"Reversible Michaelis-Menten reaction cannot have inhibition (reaction <code>{j}</code>).")
                assert self.KR[:,j].sum() == 0, throw_message(MessageType.Error, f"Reversible Michaelis-Menten reaction cannot have regulation (reaction <code>{j}</code>).")
                self.kinetic_model.append(CgmReactionType.rMM)
                self.directions.append(self.directions.append(ReactionDirection.Reversible))
    
    def read_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the CGM from CSV files.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV files.
        """
        model_path = path+"/"+self.name
        assert os.path.exists(model_path), throw_message(MessageType.Error, "Folder "+model_path+" does not exist.")
        self.read_Info_from_csv(path)
        self.read_Mx_from_csv(path)
        self.read_kcat_from_csv(path)
        self.read_K_from_csv(path)
        self.read_KA_from_csv(path)
        self.read_KI_from_csv(path)
        self.read_KR_from_csv(path)
        self.read_conditions_from_csv(path)
        self.read_constant_rhs_from_csv(path)
        self.read_constant_reactions_from_csv(path)
        self.read_protein_contributions_from_csv(path)
        self.read_initial_solution_from_csv(path)
        self.check_model_loading()
        self.initialize_model_mathematical_variables()

    def read_from_ods( self, path: Optional[str] = "." ) -> None:
        """
        Read the CGM from ODS files.

        Parameters
        ----------
        path : str, default="."
            Path to the ODS files.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Temporarily convert ODS to CSV files #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        filename = path+"/"+self.name+".ods"
        assert os.path.exists(filename), throw_message(MessageType.Error, "Folder "+filename+" does not exist.")
        xls = pd.ExcelFile(filename, engine="odf")
        if not os.path.exists("./temp/"):
            os.mkdir("./temp/")
        if not os.path.exists("./temp/"+self.name):
            os.mkdir("./temp/"+self.name)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name).fillna("")
            df.to_csv("./temp/"+self.name+"/"+sheet_name+".csv", sep=";", index=False)
        # 2) Load the model from the temporary CSV files #
        self.read_from_csv(path="./temp")
        self.check_model_loading()
        self.initialize_model_mathematical_variables()
        # 3) Delete the temporary files #
        for sheet_name in xls.sheet_names:
            if os.path.exists("./temp/"+self.name+"/"+sheet_name+".csv"):
               os.remove("./temp/"+self.name+"/"+sheet_name+".csv")
            if os.path.exists("./temp/"+self.name+"/f0.csv"):
                os.remove("./temp/"+self.name+"/f0.csv")
        os.rmdir("./temp/"+self.name)
        os.rmdir("./temp/")
    
    def write_to_csv( self, path: Optional[str] = ".", name: Optional[str] = "" ) -> None:
        """
        Write the CGM to CSV files.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV files.
        name : str, default=""
            Name of the CGM. If not provided, the name of the CGM instance will be used.
        """
        assert os.path.exists(path), throw_message(MessageType.Error, f"The path <code>{path}</code> does not exist")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Check the existence of the folder #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        model_path = path+"/"+(name if name != "" else self.name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            files = ["Info.csv",
                     "M.csv", "kcat.csv", "K.csv",
                     "KA.csv", "KI.csv", "KR.csv",
                     "conditions.csv", "f0.csv",
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
        M_df = pd.DataFrame(self.Mx, index=self.metabolite_ids, columns=self.reaction_ids)
        M_df.to_csv(model_path+"/M.csv", sep=";")
        del(M_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Write the kcat vectors            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        kcat_df = pd.DataFrame(self.kcat_f, index=self.reaction_ids, columns=["kcat_f"])
        kcat_df["kcat_b"] = self.kcat_b
        kcat_df = kcat_df.transpose()
        kcat_df.to_csv(model_path+"/kcat.csv", sep=";")
        del(kcat_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Write the KM matrix               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        K_df = pd.DataFrame(self.KM_f+self.KM_b, index=self.metabolite_ids, columns=self.reaction_ids)
        K_df.to_csv(model_path+"/K.csv", sep=";")
        del(K_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Write the KA, KI and KR matrices  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if np.any(self.KA):
            KA_df = pd.DataFrame(self.KA, index=self.metabolite_ids, columns=self.reaction_ids)
            KA_df.to_csv(model_path+"/KA.csv", sep=";")
            del(KA_df)
        if np.any(self.KI):
            KI_df = pd.DataFrame(self.KI, index=self.metabolite_ids, columns=self.reaction_ids)
            KI_df.to_csv(model_path+"/KI.csv", sep=";")
            del(KI_df)
        if np.any(self.KR):
            KR_df = pd.DataFrame(self.KR, index=self.metabolite_ids, columns=self.reaction_ids)
            KR_df.to_csv(model_path+"/KR.csv", sep=";")
            del(KR_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 7) Write the conditions              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        conditions_df = pd.DataFrame(self.conditions, index=self.condition_params, columns=self.condition_ids)
        conditions_df.to_csv(model_path+"/conditions.csv", sep=";")
        del(conditions_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 8) Write the constant RHS terms      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(self.constant_rhs) > 0:
            f = open(model_path+"/constant_rhs.csv", "w")
            f.write("metabolite;value\n")
            for item in self.constant_rhs.items():
                f.write(item[0]+";"+str(item[1])+"\n")
            f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 9) Write the constant reactions      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(self.constant_reactions) > 0:
            f = open(model_path+"/constant_reactions.csv", "w")
            f.write("reaction;value\n")
            for item in self.constant_reactions.items():
                f.write(item[0]+";"+str(item[1])+"\n")
            f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 10) Save protein contributions       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(self.protein_contributions) > 0:
            f = open(model_path+"/protein_contributions.csv", "w")
            f.write("reaction;protein;contribution\n")
            for item in self.protein_contributions.items():
                r_id = item[0]
                for p_id, val in item[1].items():
                    f.write(r_id+";"+p_id+";"+str(val)+"\n")
            f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 11) Save the initial solution        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(self.initial_solution) > 0:
            f = open(model_path+"/f0.csv", "w")
            f.write("reaction;f0\n")
            for j in range(self.nj):
                f.write(self.reaction_ids[j]+";"+str(self.initial_solution[j])+"\n")
            f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 12) Save the optimums per condition  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if not self.optima_data.empty:
            self.optima_data.to_csv(model_path+"/optimal_solutions.csv", sep=';', index=False)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 13) Save random initial solutions    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if not self.random_data.empty:
            self.random_data.to_csv(model_path+"/random_solutions.csv", sep=';', index=False)
    
    def write_to_ods( self, path: Optional[str] = ".", name: Optional[str] = "" ) -> None:
        """
        Export the CGM to a folder in ODS format.

        Parameters
        ----------
        path : Optional[str], default="."
            Path to the folder.
        name : Optional[str], default=""
            Name of the CGM. If not provided, the name of the CGM instance will be used.
        """
        assert os.path.exists(path), throw_message(MessageType.Error, f"The path <code>{path}</code> does not exist")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Check the existence of the folder #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        xls_path = path+"/"+(name if name != "" else self.name)+".xlsx"
        ods_path = path+"/"+(name if name != "" else self.name)+".ods"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Write the information             #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Write the mass fraction matrix    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        M_df = pd.DataFrame(self.Mx, index=self.metabolite_ids, columns=self.reaction_ids)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Write the kcat vectors            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        kcat_df           = pd.DataFrame(self.kcat_f, index=self.reaction_ids, columns=["kcat_f"])
        kcat_df["kcat_b"] = self.kcat_b
        kcat_df           = kcat_df.transpose()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Write the forward KM matrices     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        K_df = pd.DataFrame(self.KM_f+self.KM_b, index=self.metabolite_ids, columns=self.reaction_ids)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Write the KA, KI and KR matrices  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        KA_df = None
        KI_df = None
        KR_df = None
        if np.any(self.KA):
            KA_df = pd.DataFrame(self.KA, index=self.metabolite_ids, columns=self.reaction_ids)
        if np.any(self.KI):
            KI_df = pd.DataFrame(self.KI, index=self.metabolite_ids, columns=self.reaction_ids)
        if np.any(self.KR):
            KR_df = pd.DataFrame(self.KR, index=self.metabolite_ids, columns=self.reaction_ids)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 7) Write the conditions              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        conditions_df = pd.DataFrame(self.conditions, index=self.condition_params, columns=self.condition_ids)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 8) Write the constant terms          #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        constant_rhs_df       = None
        constant_reactions_df = None
        if len(self.constant_rhs) > 0:
            constant_rhs_df = pd.DataFrame(list(self.constant_rhs.items()), columns=["metabolite", "value"])
        if len(self.constant_reactions) > 0:
            constant_reactions_df = pd.DataFrame(list(self.constant_reactions.items()), columns=["reaction", "value"])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 9) Write the protein contributions   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        protein_contributions_df = None
        if len(self.protein_contributions) > 0:
            rows = []
            for r_id, contributions in self.protein_contributions.items():
                for p_id, contribution in contributions.items():
                    rows.append([r_id, p_id, contribution])
            protein_contributions_df = pd.DataFrame(rows, columns=["reaction", "protein", "contribution"])
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 10) Save the initial solution        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f0_df = None
        if len(self.initial_solution) > 0:
            f0_df            = pd.DataFrame(self.initial_solution, index=self.reaction_ids, columns=["f0"])
            f0_df.index.name = "reaction"
            f0_df.reset_index(inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 11) Write the variables in xlsx      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        with pd.ExcelWriter(xls_path) as writer:
            if Info_df is not None:
                Info_df.to_excel(writer, sheet_name="Info", index=False, header=False)
            M_df.to_excel(writer, sheet_name="M")
            kcat_df.to_excel(writer, sheet_name="kcat")
            K_df.to_excel(writer, sheet_name="K")
            conditions_df.to_excel(writer, sheet_name="conditions")
            if KA_df is not None:
              KA_df.to_excel(writer, sheet_name="KA")
            if KR_df is not None:
                KI_df.to_excel(writer, sheet_name="KI")
            if constant_rhs_df is not None:
                constant_rhs_df.to_excel(writer, sheet_name="constant_rhs", index=False)
            if constant_reactions_df is not None:
                constant_reactions_df.to_excel(writer, sheet_name="constant_reactions", index=False)
            if protein_contributions_df is not None:
                protein_contributions_df.to_excel(writer, sheet_name="protein_contributions", index=False)
            if f0_df is not None:
                f0_df.to_excel(writer, sheet_name="f0", index=False)
            if not self.optima_data.empty:
                self.optima_data.to_excel(writer, sheet_name="optimal_solutions", index=False)
            if not self.random_data.empty:
                self.random_data.to_excel(writer, sheet_name="random_solutions", index=False)
        data_xlsx = get_data(xls_path)
        save_data(ods_path, data_xlsx)
        os.system("rm "+xls_path)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 11) Free memory                      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        del(Info_df)
        del(M_df)
        del(kcat_df)
        del(K_df)
        del(KA_df)
        del(KI_df)
        del(KR_df)
        del(conditions_df)
        del(constant_rhs_df)
        del(constant_reactions_df)
        del(protein_contributions_df)
        del(f0_df)
    
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
        Clear all external conditions from the CGM.
        """
        self.condition_ids    = []
        self.condition_params = ["rho"] + self.x_ids
        self.conditions       = np.array([])
    
    def add_condition( self, condition_id: str, rho: float, default_concentration: Optional[float] = 1.0, metabolites: Optional[dict[str, float]] = None ) -> None:
        """
        Add an external condition to the CGM.

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
        self.conditions = np.column_stack([self.conditions, np.array(vec)]) if self.conditions.size else np.array(vec)
    
    def clear_constant_rhs( self ) -> None:
        """
        Clear all constant RHS terms from the CGM.
        """
        self.constant_rhs = {}
    
    def add_constant_rhs( self, metabolite_id: str, value: float ) -> None:
        """
        Make a CGM metabolite constant in the RHS term for the initial solution.

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
        Clear all constant reactions from the CGM.
        """
        self.constant_reactions = {}
    
    def add_constant_reaction( self, reaction_id: str, value: float ) -> None:
        """
        Make a CGM reaction constant to a given flux value.

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
            if self.adjust_concentrations and self.x[i] < CgmConstants.TOL.value:
                self.x[i] = CgmConstants.TOL.value

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
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 4) Analytical methods              #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    def gaussian_term( self, x: np.array, mu: float ) -> np.array:
        """
        Compute the Gaussian term.

        Parameters
        ----------
        x : np.array
            Input array.
        mu : float
            Mean of the Gaussian kernel function.

        Returns
        -------
        np.array
            Gaussian term values.
        """
        return (x - mu)/(CgmConstants.REGULATION_SIGMA*x)**2
    
    def gaussian_kernel( self, x: np.array, mu: float ) -> np.array:
        """
        Compute the Gaussian kernel function.

        Parameters
        ----------
        x : np.array
            Input array.
        mu : float
            Mean of the Gaussian kernel function.

        Returns
        -------
        np.array
            Gaussian kernel values.
        """
        return np.exp(-0.5 * ((x - mu)/(CgmConstants.REGULATION_SIGMA*x))**2)

    def compute_c( self ) -> None:
        """
        Compute the internal metabolite concentrations.
        """
        self.c = self.rho*self.M.dot(self.f)
        if self.adjust_concentrations:
            self.c[self.c < CgmConstants.TOL.value] = CgmConstants.TOL.value
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

    def iMMr( self, j: int ) -> None:
        """
        Compute the turnover time tau for an irreversible
        Michaelis-Menten reaction with regulation
        (only one regulator per reaction)

        Parameters
        ----------
        j : int
            Reaction index.
        """
        kr_vec = self.KR[:,j]
        kr_vec[kr_vec < CgmConstants.TOL.value] = self.xc[kr_vec < CgmConstants.TOL.value]
        gaussian_kernel = self.log_gaussian_kernel(self.xc, kr_vec)
        term1           = np.prod(1.0+self.KM_f[:,j]/(self.xc*gaussian_kernel))
        term2           = self.kcat_f[j]
        self.tau_j[j]   = term1/term2
    
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
        if self.kinetic_model[j] == CgmReactionType.iMM:
            self.iMM(j)
        elif self.kinetic_model[j] == CgmReactionType.iMMa:
            self.iMMa(j)
        elif self.kinetic_model[j] == CgmReactionType.iMMi:
            self.iMMi(j)
        elif self.kinetic_model[j] == CgmReactionType.iMMia:
            self.iMMia(j)
        elif self.kinetic_model[j] == CgmReactionType.iMMr:
            self.iMMr(j)
        elif self.kinetic_model[j] == CgmReactionType.rMM:
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

    def diMMr( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for an
        irreversible Michaelis-Menten reaction with regulation with
        respect to metabolite concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        kr_vec = self.KR[:,j]
        kr_vec[kr_vec < CgmConstants.TOL.value] = self.xc[kr_vec < CgmConstants.TOL.value]
        gaussian_kernel = self.gaussian_kernel(self.xc, kr_vec)
        constant1       = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            term1             = -self.KM_f[y,j]/self.c[i]**2
            term2             = (self.KM_f[y,j]+self.c[i])/self.c[i]
            term3             = self.gaussian_term(self.c[i], kr_vec[y])
            term4             = gaussian_kernel[y]*(term1+term2*term3)
            term5             = np.prod((self.xc[indices]+self.KM_f[indices,j])/(self.xc[indices]*gaussian_kernel[indices]))
            self.ditau_j[j,i] = term4*term5/constant1

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
        if self.kinetic_model[j] == CgmReactionType.iMM:
            self.diMM(j)
        elif self.kinetic_model[j] == CgmReactionType.iMMa:
            self.diMMa(j)
        elif self.kinetic_model[j] == CgmReactionType.iMMi:
            self.diMMi(j)
        elif self.kinetic_model[j] == CgmReactionType.iMMia:
            self.diMMia(j)
        elif self.kinetic_model[j] == CgmReactionType.iMMr:
            self.diMMr(j)
        elif self.kinetic_model[j] == CgmReactionType.rMM:
            self.drMM(j)
    
    def compute_mu( self ) -> None:
        """
        Compute the growth rate mu.
        """
        self.mu            = self.M[self.a,self.r]*self.f[self.r]/(self.tau_j.dot(self.f))
        self.doubling_time = np.log(2)/np.log(1+self.mu)

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
    
    def calculate_first_order_terms( self ) -> None:
        """
        Calculate the first order terms of the model state.
        """
        self.compute_c()
        for j in range(self.nj):
            self.compute_tau(j)
        self.compute_mu()
        self.compute_v()
        self.compute_p()
        self.compute_b()
        self.compute_density()
    
    def calculate_second_order_terms( self ) -> None:
        """
        Calculate the second order terms of the model state.
        """
        for j in range(self.nj):
            self.compute_dtau(j)
        self.compute_dmu_f()
        self.compute_GCC_f()
    
    def calculate( self ) -> None:
        """
        Calculate the model state.
        """
        self.calculate_first_order_terms()
        self.calculate_second_order_terms()

    def check_model_consistency( self ) -> None:
        """
        Check the model state's consistency.
        """
        test1 = (np.abs(self.density-1.0) < CgmConstants.TOL.value)
        test2 = (sum(1 for x in self.c if x < -CgmConstants.TOL.value) == 0)
        test3 = (sum(1 for x in self.p if x < -CgmConstants.TOL.value) == 0)
        self.consistent = True
        if not (test1 and test2 and test3):
            self.consistent = False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 5) Generation of initial solutions #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def solve_local_linear_problem( self, max_flux_fraction: Optional[float] = 10.0, rhs_factor: Optional[float] = 1000.0 ) -> None:
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
        assert max_flux_fraction > CgmConstants.TOL.value, throw_message(MessageType.Error, f"Maximal flux fraction must be greater than {CgmConstants.MIN_FLUX_FRACTION.value}.")
        assert rhs_factor > 0.0, throw_message(MessageType.Error, "RHS factor must be positive.")
        lb_vec = []
        for j in range(self.nj):
            if self.reversible[j]:
                lb_vec.append(-max_flux_fraction)
            else:
                lb_vec.append(CgmConstants.TOL.value)
        #lb_vec = [CgmConstants.TOL.value]*self.nj
        ub_vec = [max_flux_fraction]*self.nj
        print(lb_vec)
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
            self.initial_solution = np.copy(v.X)
            return True
        except:
            throw_message(MessageType.Error, "Local linear problem could not be solved.")
            return False

    def find_initial_solution( self, max_flux_fraction: Optional[float] = 10.0, rhs_factor: Optional[float] = 10.0,
                               condition_id: Optional[str] = "1" ) -> None:
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
        """
        solved = self.solve_local_linear_problem(max_flux_fraction=max_flux_fraction, rhs_factor=rhs_factor)
        if solved:
            self.set_condition(condition_id)
            self.set_f0(self.initial_solution)
            self.calculate()
            self.check_model_consistency()
            if self.consistent:
                throw_message(MessageType.Info, f"Model is consistent with mu = {self.mu}.")
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
        assert max_flux_fraction > CgmConstants.TOL.value, throw_message(MessageType.Error, f"Maximal flux fraction must be greater than {CgmConstants.TOL.value}.")
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
                self.f_trunc = self.f_trunc*(max_flux_fraction-CgmConstants.TOL)+CgmConstants.TOL
                self.set_f_from_f_trunc()
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
    # 7) Summary functions               #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def information( self ) -> None:
        """
        Print the CGM information.
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
        html_str  = "<h1>CGM "+self.name+"</h1>"
        for category, df in dfs.items():
            html_str += "<table>"
            html_str += "<tr style='text-align:left'><td style='vertical-align:top'>"
            html_str += "<h2 style='text-align: left;'>"+category+"</h2>"
            html_str += df.to_html(escape=False, index=False)
            html_str += "</td></tr>"
            html_str += "</table>"
        display_html(html_str,raw=True)

    def summary( self ) -> None:
        """
        Print a summary of the CGM.
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
        html_str  = "<h1>CGM "+self.name+" summary</h1>"
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

def read_csv_model( name: str, path: Optional[str] = "." ) -> Model:
    """
    Read a CGM from CSV files.

    Parameters
    ----------
    name : str
        Name of the CGM.
    path : Optional[str], default="."
        Path to the model folder.

    Returns
    -------
    Model
        The loaded CGM.
    """
    assert os.path.exists(path+"/"+name), throw_message(MessageType.Error, "The folder "+path+"/"+name+" does not exist.")
    model = Model(name)
    model.read_from_csv(path=path)
    return model

def read_ods_model( name: str, path: Optional[str] = "." ) -> Model:
    """
    Read a CGM from ODS files.

    Parameters
    ----------
    name : str
        Name of the CGM.
    path : Optional[str], default="."
        Path to the model folder.

    Returns
    -------
    Model
        The loaded CGM.
    """
    assert os.path.exists(path+"/"+name+".ods"), throw_message(MessageType.Error, "The folder "+path+"/"+name+".ods does not exist.")
    model = Model(name)
    model.read_from_ods(path=path)
    return model

def get_toy_model_path( model_name: str ) -> str:
    """
    Get the path of a toy CGM included in the Python package as CSV files.

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

def read_toy_model( name: str ) -> Model:
    """
    Read a toy CGM included in the Python package as CSV files.

    Parameters
    ----------
    name : str
        Name of the toy model.

    Returns
    -------
    Model
        The loaded CGM.
    """
    model_dir  = Path(pkgutil.resolve_name("gba.data").__file__).parent
    model_path = str(Path(model_dir , "toy_models/"))
    model      = read_csv_model(name=name, path=model_path)
    return model

def backup_model( model: Model, name: Optional[str] = "", path: Optional[str] = "." ) -> None:
    """
    Backup a CGM in binary format (extension .cgm).

    Parameters
    ----------
    model : Model
        CGM to backup.
    name : str
        Name of the backup file.
    path : str
        Path to the backup file.
    """
    filename = ""
    if name != "":
        filename = path+"/"+name+".cgm"
    else:
        filename = path+"/"+model.name+".cgm"
    ofile = open(filename, "wb")
    pickle.dump(model, ofile)
    ofile.close()
    assert os.path.isfile(filename), throw_message(MessageType.Error, ".cgm file creation failed.")

def load_model( path: str ) -> Model:
    """
    Load a CGM from a binary file.

    Parameters
    ----------
    path : str
        Path to the CGM file.
    """
    assert path.endswith(".cgm"), throw_message(MessageType.Error, "CGM file extension is missing.")
    assert os.path.isfile(path), throw_message(MessageType.Error, "CGM file not found.")
    ifile = open(path, "rb")
    model = pickle.load(ifile)
    ifile.close()
    return model

def create_model( name: str, path: Optional[str] = ".", cgm_path: Optional[str] = ".", save_LP: Optional[bool] = False, save_optima: Optional[bool] = False ) -> None:
    """
    Create a CGM from CSV files, and save it as a binary file.

    Parameters
    ----------
    name : str
        Name of the CGM.
    path : Optional[str], default="."
        Path to the binary file.
    cgm_path : Optional[str], default=""
        Path to save the CGM.
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
        model.set_f0(model.initial_solution)
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
    backup_model(model=model, name=name, path=cgm_path)
    del model

