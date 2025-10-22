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
Filename: Model.py
Author: Charles Rocabert, Furkan Mert
Date: 2024-10-22
Description:
    Model class of the gbapy module.
License: GNU General Public License v3 (GPLv3)
Copyright: © 2024-2025 Charles Rocabert, Furkan Mert.
"""

import os
import time
import copy
import random
import pickle
import pkgutil
import subprocess
import numpy as np
import pandas as pd
import gurobipy as gp
import plotly.express as px
from typing import Optional
from pyexcel_xlsx import get_data
from pyexcel_ods3 import save_data

from IPython.display import display_html, clear_output

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
    Class to manipulate models.

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
    s : list
        Transport reaction indices.
    e : list
        Enzymatic reaction indices.
    r : int
        Ribosome reaction index.
    ns : int
        Number of transport reactions.
    ne : int
        Number of enzymatic reactions.
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
    consistent : bool
        Is the model consistent?
    adjust_concentrations : bool
        Adjust concentrations to avoid negative values.
    condition : str
        External condition.
    rho : float
        Total density.
    q0 : np.array
        Initial LP solution.
    dmu_dq : np.array
        Local mu derivatives with respect to q.
    Gamma : np.array
        Local growth control coefficients with respect to q.
    q_trunc : np.array
        Truncated q vector (first element is removed).
    q : np.array
        Flux fractions vector.
    data : pd.DataFrame
        Data from optimizations.
    """

    def __init__( self, name: str ) -> None:
        """
        Constructor of the Model class.
        
        Parameters
        ----------
        name : str
            Name of the model.
        """
        assert name != "", throw_message(MessageType.ERROR, "You must provide a name to the model constructor.")
        self.name = name
        self.info = {}

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Model                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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
        self.rho                = 0.0
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
        self.conditions_loaded            = False
        self.constant_rhs_loaded          = False
        self.constant_reactions_loaded    = False
        self.protein_contributions_loaded = False
        self.initial_solution_loaded      = False
        self.optimal_solutions_loaded     = False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Model constants           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        ### Vector lengths ###
        self.nx = 0
        self.nc = 0
        self.ni = 0
        self.nj = 0

        ### Indices for reactions: s (transport), e (enzymatic), and ribosome r ###
        self.sM = []
        self.s  = []
        self.e  = []
        self.r  = 0
        self.ns = 0
        self.ne = 0

        ### Indices: m (metabolite), a (all proteins) ###
        self.m = []
        self.a = 0

        ### Matrix column rank ###
        self.column_rank      = 0
        self.full_column_rank = False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Solutions                 #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.initial_solution  = np.array([])
        self.optimal_solutions = {}
        self.random_solutions  = {}
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Model variables           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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
        self.consistent            = False
        self.adjust_concentrations = False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Model dynamical variables #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.condition = ""
        self.q0        = np.array([])
        self.dmu_dq    = np.array([])
        self.Gamma     = np.array([])
        self.q_trunc   = np.array([])
        self.q         = np.array([])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Optimization data         #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.data = None
        
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
        assert os.path.exists(filename), throw_message(MessageType.ERROR, "The file M.csv does not exist in the specified path: "+filename)
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
        assert os.path.exists(filename), throw_message(MessageType.ERROR, "The file kcat.csv does not exist in the specified path: "+filename)
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
        assert os.path.exists(filename), throw_message(MessageType.ERROR, "The file K.csv does not exist in the specified path: "+filename)
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

    def read_rho_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the total density from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        filename = path+"/"+self.name+"/rho.csv"
        if os.path.exists(filename):
            f = open(filename, "r")
            l = f.readline()
            l = f.readline()
            l = l.strip("\n").split(";")
            self.rho = float(l[1])
            f.close()
    
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
        assert os.path.exists(filename), throw_message(MessageType.ERROR, "The file conditions.csv does not exist in the specified path: "+filename)
        df                     = pd.read_csv(filename, sep=";")
        self.condition_params  = list(df["Unnamed: 0"])
        self.condition_ids     = list(df.columns)[1:df.shape[1]]
        self.condition_ids     = [str(int(name)) for name in self.condition_ids]
        df                     = df.drop(["Unnamed: 0"], axis=1)
        df.index               = self.condition_params
        self.conditions        = np.array(df)
        self.conditions_loaded = True
        del(df)

    def read_q_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the initial and optimal q solution from a CSV file.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        self.initial_solution_loaded  = False
        self.optimal_solutions_loaded = False
        filename                      = path+"/"+self.name+"/q.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename, sep=";", index_col=0)
            df = df.T
            if "q0" in df.columns:
                self.initial_solution        = np.array(df["q0"])
                self.initial_solution_loaded = True
            for c_name in df.columns:
                if c_name in self.condition_ids:
                    self.optimal_solutions[str(int(c_name))] = np.array(df[c_name])
                    self.optimal_solutions_loaded            = True
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
    
    def read_random_solutions_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the random solutions from a CSV file (on request).

        Parameters
        ----------
        path : str, default="."
            Path to the CSV file.
        """
        filename = path+"/"+self.name+"/qrandom.csv"
        if os.path.exists(filename):
            self.random_data      = pd.read_csv(filename, sep=";")
            self.random_solutions = {
                str(int(row['condition'])): row.drop(['condition', 'mu']).to_numpy()
                for _, row in self.random_data.iterrows()
            }

    def check_model_loading( self, verbose: Optional[bool] = False ) -> None:
        """
        Check if the model is loaded correctly.

        Parameters
        ----------
        verbose : Optional[bool], default=False
            Print the error messages.
        """
        if not self.Info_loaded and verbose:
            throw_message(MessageType.INFO, "Model information is missing.")
        if not self.KA_loaded and verbose:
            throw_message(MessageType.INFO, "No KA constants.")
        if not self.KI_loaded and verbose:
            throw_message(MessageType.INFO, "No KI constants.")
        if not self.constant_rhs_loaded and verbose:
            throw_message(MessageType.INFO, "No constant RHS terms.")
        if not self.constant_reactions_loaded and verbose:
            throw_message(MessageType.INFO, "No constant reactions.")
        if not self.protein_contributions_loaded and verbose:
            throw_message(MessageType.INFO, "Protein contributions are missing.")
        if not self.initial_solution_loaded and verbose:
            throw_message(MessageType.INFO, "The initial solution is missing.")
        if not self.optimal_solutions_loaded and verbose:
            throw_message(MessageType.INFO, "No optimal solutions.")
    
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
        self.r  = self.nj-1
        for j in range(self.nj-1):
            for i in range(self.ni):
                if self.Mx[i,j] != 0 and self.metabolite_ids[i] in self.x_ids:
                    self.s.append(j)
                    break
        self.e  = [j for j in range(self.nj-1) if j not in self.s]
        self.ns = len(self.s)
        self.ne = len(self.e)
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
        # 7) Model dynamical variables                           #
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
        self.q0      = np.zeros(self.nj)
        self.dmu_dq  = np.zeros(self.nj)
        self.Gamma   = np.zeros(self.nj)
        self.q_trunc = np.zeros(self.nj-1)
        self.q       = np.zeros(self.nj)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 9) Define the kinetic model of each reaction           #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.kinetic_model.clear()
        self.directions.clear()
        for j in range(self.nj):
            if (self.kcat_b[j] == 0 and self.KA[:,j].sum() == 0 and self.KI[:,j].sum() == 0):
                self.kinetic_model.append(GbaReactionType.IMM)
                self.directions.append(ReactionDirection.FORWARD)
            elif (self.kcat_b[j] == 0 and self.KA[:,j].sum() > 0 and self.KI[:,j].sum() == 0):
                self.kinetic_model.append(GbaReactionType.IMMA)
                self.directions.append(ReactionDirection.FORWARD)
            elif (self.kcat_b[j] == 0 and self.KA[:,j].sum() == 0 and self.KI[:,j].sum() > 0):
                self.kinetic_model.append(GbaReactionType.IMMI)
                self.directions.append(ReactionDirection.FORWARD)
            elif (self.kcat_b[j] == 0 and self.KA[:,j].sum() > 0 and self.KI[:,j].sum() > 0):
                self.kinetic_model.append(GbaReactionType.IMMIA)
                self.directions.append(ReactionDirection.FORWARD)
            elif (self.kcat_b[j] > 0):
                assert self.KA[:,j].sum() == 0, throw_message(MessageType.ERROR, f"Reversible Michaelis-Menten reaction cannot have activation (reaction <code>{j}</code>).")
                assert self.KI[:,j].sum() == 0, throw_message(MessageType.ERROR, f"Reversible Michaelis-Menten reaction cannot have inhibition (reaction <code>{j}</code>).")
                self.kinetic_model.append(GbaReactionType.RMM)
                self.directions.append(ReactionDirection.REVERSIBLE)
    
    def read_from_csv( self, path: Optional[str] = "." ) -> None:
        """
        Read the model from CSV files.

        Parameters
        ----------
        path : str, default="."
            Path to the CSV files.
        """
        model_path = path+"/"+self.name
        assert os.path.exists(model_path), throw_message(MessageType.ERROR, "Folder "+model_path+" does not exist.")
        self.read_Info_from_csv(path)
        self.read_Mx_from_csv(path)
        self.read_kcat_from_csv(path)
        self.read_K_from_csv(path)
        self.read_KA_from_csv(path)
        self.read_KI_from_csv(path)
        self.read_rho_from_csv(path)
        self.read_conditions_from_csv(path)
        self.read_q_from_csv(path)
        self.read_constant_rhs_from_csv(path)
        self.read_constant_reactions_from_csv(path)
        self.read_protein_contributions_from_csv(path)
        self.read_random_solutions_from_csv(path)
        self.check_model_loading()
        self.initialize_model_mathematical_variables()

    def read_from_ods( self, path: Optional[str] = "." ) -> None:
        """
        Read the model from ODS files.

        Parameters
        ----------
        path : str, default="."
            Path to the ODS files.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Temporarily convert ODS to CSV files        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        random.seed(int(time.time()))
        filename       = path+"/"+self.name+".ods"
        random_number  = random.randint(0, 100000000)
        temporary_name = "temp_"+str(random_number)
        assert os.path.exists(filename), throw_message(MessageType.ERROR, "Folder "+filename+" does not exist.")
        xls = pd.ExcelFile(filename, engine="odf")
        if not os.path.exists(temporary_name):
            os.mkdir(temporary_name)
        if not os.path.exists(temporary_name+"/"+self.name):
            os.mkdir(temporary_name+"/"+self.name)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name).fillna("")
            df.to_csv(temporary_name+"/"+self.name+"/"+sheet_name+".csv", sep=";", index=False)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Load the model from the temporary CSV files #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.read_from_csv(path=temporary_name)
        self.check_model_loading()
        self.initialize_model_mathematical_variables()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Delete the temporary files                  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for sheet_name in xls.sheet_names:
            if os.path.exists(temporary_name+"/"+self.name+"/"+sheet_name+".csv"):
               os.remove(temporary_name+"/"+self.name+"/"+sheet_name+".csv")
        #if os.path.exists(temporary_name+"/"+self.name+"/q.csv"):
        #    os.remove(temporary_name+"/"+self.name+"/q.csv")
        os.rmdir(temporary_name+"/"+self.name)
        os.rmdir(temporary_name+"/")
    
    def export_to_csv( self, name: Optional[str] = "", path: Optional[str] = "." ) -> None:
        """
        Write the model to CSV files.

        Parameters
        ----------
        name : str, default=""
            Name of the model. If not provided, the name of the model instance
            will be used.
        path : str, default="."
            Path to the CSV files.
        """
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
                     "protein_contributions.csv",
                     "q.csv", "random_solutions.csv"]
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
        M_df.replace(-0.0, 0.0, inplace=True)
        M_df.to_csv(model_path+"/M.csv", sep=";")
        del(M_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Write the kcat vectors            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        kcat_df = pd.DataFrame(self.kcat_f, index=self.reaction_ids, columns=["kcat_f"])
        kcat_df["kcat_b"] = self.kcat_b
        kcat_df = kcat_df.transpose()
        kcat_df.replace(-0.0, 0.0, inplace=True)
        kcat_df.to_csv(model_path+"/kcat.csv", sep=";")
        del(kcat_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Write the KM matrix               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        K_df = pd.DataFrame(self.KM_f+self.KM_b, index=self.metabolite_ids, columns=self.reaction_ids)
        K_df.replace(-0.0, 0.0, inplace=True)
        K_df.to_csv(model_path+"/K.csv", sep=";")
        del(K_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Write the KA and KI matrices      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if np.any(self.KA):
            KA_df = pd.DataFrame(self.KA, index=self.metabolite_ids, columns=self.reaction_ids)
            KA_df.replace(-0.0, 0.0, inplace=True)
            KA_df.to_csv(model_path+"/KA.csv", sep=";")
            del(KA_df)
        if np.any(self.KI):
            KI_df = pd.DataFrame(self.KI, index=self.metabolite_ids, columns=self.reaction_ids)
            KI_df.replace(-0.0, 0.0, inplace=True)
            KI_df.to_csv(model_path+"/KI.csv", sep=";")
            del(KI_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 7) Write rho                         #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open(model_path+"/rho.csv", "w")
        f.write(";(g/L)\n")
        f.write("rho;"+str(self.rho)+"\n")
        f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 8) Write the conditions              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        conditions_df = pd.DataFrame(self.conditions, index=self.condition_params, columns=self.condition_ids)
        conditions_df.replace(-0.0, 0.0, inplace=True)
        conditions_df.to_csv(model_path+"/conditions.csv", sep=";")
        del(conditions_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 9) Save q data                       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(self.initial_solution) > 0 or len(self.optimal_solutions) > 0:
            data = {}
            if len(self.initial_solution) > 0:
                data["q0"] = self.initial_solution
            for key, val in self.optimal_solutions.items():
                data[str(int(key))] = val
            q_df = pd.DataFrame(data, index=self.reaction_ids).T
            q_df.replace(-0.0, 0.0, inplace=True)
            q_df.to_csv(model_path+"/q.csv", sep=";")
            del(q_df)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 10) Write the constant RHS terms     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(self.constant_rhs) > 0:
            f = open(model_path+"/constant_rhs.csv", "w")
            f.write("metabolite;value\n")
            for item in self.constant_rhs.items():
                f.write(item[0]+";"+str(item[1])+"\n")
            f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 11) Write the constant reactions     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if len(self.constant_reactions) > 0:
            f = open(model_path+"/constant_reactions.csv", "w")
            f.write("reaction;value\n")
            for item in self.constant_reactions.items():
                f.write(item[0]+";"+str(item[1])+"\n")
            f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 12) Save protein contributions       #
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
        # 13) Write optimization data          #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if self.data is not None:
            self.data.to_csv(model_path+"/optimization_data.csv", sep=";")
    
    def export_to_ods( self, name: Optional[str] = "", path: Optional[str] = "." ) -> None:
        """
        Export the model to a folder in ODS format.

        Parameters
        ----------
        name : Optional[str], default=""
            Name of the model. If not provided, the name of the model instance
            will be used.
        path : Optional[str], default="."
            Path to the folder.
        """
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
        M_df = pd.DataFrame(self.Mx, index=self.metabolite_ids, columns=self.reaction_ids)
        M_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Write the kcat vectors          #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        kcat_df           = pd.DataFrame(self.kcat_f, index=self.reaction_ids, columns=["kcat_f"])
        kcat_df["kcat_b"] = self.kcat_b
        kcat_df           = kcat_df.transpose()
        kcat_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Write the K matrix              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        K_df = pd.DataFrame(self.KM_f+self.KM_b, index=self.metabolite_ids, columns=self.reaction_ids)
        K_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Write the KA and KI matrices    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        KA_df = None
        KI_df = None
        if np.any(self.KA):
            KA_df = pd.DataFrame(self.KA, index=self.metabolite_ids, columns=self.reaction_ids)
            KA_df.replace(-0.0, 0.0, inplace=True)
        if np.any(self.KI):
            KI_df = pd.DataFrame(self.KI, index=self.metabolite_ids, columns=self.reaction_ids)
            KI_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Write rho                       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        rho_df = pd.DataFrame([["rho", self.rho]], columns=["", "(g/L)"])
        rho_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 7) Write the conditions            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        conditions_df = pd.DataFrame(self.conditions, index=self.condition_params, columns=self.condition_ids)
        conditions_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 8) Save q data                       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        q_df = None
        if len(self.initial_solution) > 0 or len(self.optimal_solutions) > 0:
            data = {}
            if len(self.initial_solution) > 0:
                data["q0"] = self.initial_solution
            for key, val in self.optimal_solutions.items():
                data[str(int(key))] = val
            q_df = pd.DataFrame(data, index=self.reaction_ids).T
            q_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 9) Write the constant terms        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        constant_rhs_df       = None
        constant_reactions_df = None
        if len(self.constant_rhs) > 0:
            constant_rhs_df = pd.DataFrame(list(self.constant_rhs.items()), columns=["metabolite", "value"])
            constant_rhs_df.replace(-0.0, 0.0, inplace=True)
        if len(self.constant_reactions) > 0:
            constant_reactions_df = pd.DataFrame(list(self.constant_reactions.items()), columns=["reaction", "value"])
            constant_reactions_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 10) Write protein contributions    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        protein_contributions_df = None
        if len(self.protein_contributions) > 0:
            rows = []
            for r_id, contributions in self.protein_contributions.items():
                for p_id, contribution in contributions.items():
                    rows.append([r_id, p_id, contribution])
            protein_contributions_df = pd.DataFrame(rows, columns=["reaction", "protein", "contribution"])
            protein_contributions_df.replace(-0.0, 0.0, inplace=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 11) Write the variables in xlsx    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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
            if q_df is not None:
                q_df.to_excel(writer, sheet_name="q", index=True)
            if self.data is not None:
                self.data.to_excel(writer, sheet_name="optimization_data", index=False)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 12) Free memory                    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        del(Info_df)
        del(M_df)
        del(kcat_df)
        del(K_df)
        del(KA_df)
        del(KI_df)
        del(rho_df)
        del(conditions_df)
        del(q_df)
        del(constant_rhs_df)
        del(constant_reactions_df)
        del(protein_contributions_df)
    
    def export_optimization_data( self, name: Optional[str] = "", path: Optional[str] = "." ) -> None:
        """
        Export the optimization data to CSV.

        Parameters
        ----------
        path : str, default="."
            Path to the output file.
        name : str, default=""
            Name of the model. If not provided, the name of the model instance
            will be used.
        """
        assert os.path.exists(path), throw_message(MessageType.ERROR, f"The path <code>{path}</code> does not exist")
        if self.data is not None:
            filename = path+"/"+(name if name != "" else self.name)+"_optimization_data.csv"
            self.data.to_csv(filename, sep=";")

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
        assert condition_id in self.condition_ids, throw_message(MessageType.ERROR, f"Unknown condition identifier <code>{condition_id}</code>.")
        assert condition_param in self.condition_params, throw_message(MessageType.ERROR, f"Unknown condition parameter <code>{condition_param}</code>.")
        i = self.condition_params.index(condition_param)
        j = self.condition_ids.index(condition_id)
        return self.conditions[i,j]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 3) Setters                         #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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
    
    def clear_conditions( self ) -> None:
        """
        Clear all external conditions from the model.
        """
        self.condition_ids    = []
        self.condition_params = ["rho"] + self.x_ids
        self.conditions       = np.array([])
    
    def add_condition( self, condition_id: str, rho: float, default_concentration: Optional[float] = 1.0, metabolites: Optional[dict[str, float]] = None ) -> None:
        """
        Add an external condition to the model.

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
        assert condition_id not in self.condition_ids, throw_message(MessageType.ERROR, f"Condition <code>{condition_id}</code> already exists.")
        assert rho > 0.0, throw_message(MessageType.ERROR, "The total density must be positive.")
        assert default_concentration >= 0.0, throw_message(MessageType.ERROR, "The default concentration must be positive.")
        if metabolites is not None:
            for m_id, concentration in metabolites.items():
                assert m_id in self.metabolite_ids, throw_message(MessageType.ERROR, f"Metabolite <code>{m_id}</code> does not exist.")
                assert m_id in self.condition_params, throw_message(MessageType.ERROR, f"Metabolite <code>{m_id}</code> is not a condition parameter.")
                assert concentration >= 0.0, throw_message(MessageType.ERROR, f"The concentration of metabolite <code>{m_id}</code> must be positive.")
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
        Clear all constant RHS terms from the model.
        """
        self.constant_rhs = {}
    
    def add_constant_rhs( self, metabolite_id: str, value: float ) -> None:
        """
        Make a metabolite constant in the RHS term for the initial solution.

        Parameters
        ----------
        metabolite_id : str
            Identifier of the metabolite.
        value : float
            Flux value.
        """
        assert metabolite_id in self.metabolite_ids, throw_message(MessageType.ERROR, f"Unknown metabolite identifier <code>{metabolite_id}</code>.")
        assert value >= 0.0, throw_message(MessageType.ERROR, "The constant value must be positive.")
        self.constant_rhs[metabolite_id] = value
    
    def clear_constant_reactions( self ) -> None:
        """
        Clear all constant reactions from the model.
        """
        self.constant_reactions = {}
    
    def add_constant_reaction( self, reaction_id: str, value: float ) -> None:
        """
        Make a reaction constant to a given flux value.

        Parameters
        ----------
        reaction_id : str
            Identifier of the reaction.
        value : float
            Flux value.
        """
        assert reaction_id in self.reaction_ids, throw_message(MessageType.ERROR, f"Unknown reaction identifier <code>{reaction_id}</code>.")
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
        self.q0      = np.zeros(self.nj)
        self.dmu_dq  = np.zeros(self.nj)
        self.Gamma   = np.zeros(self.nj)
        self.q_trunc = np.zeros(self.nj-1)
        self.q       = np.zeros(self.nj)
    
    def set_condition( self, condition_id: str ) -> None:
        """
        Set the external condition.
        (minimal values bounded to MIN_CONCENTRATION)

        Parameters
        ----------
        condition_id : str
            External condition identifier.
        """
        assert condition_id in self.condition_ids, throw_message(MessageType.ERROR, "Unknown condition identifier <code>{condition_id}</code>.")
        self.condition = condition_id
        self.rho       = self.get_condition(self.condition, "rho")
        for i in range(self.nx):
            x_name    = self.x_ids[i]
            x_value   = self.get_condition(self.condition, x_name)
            self.x[i] = x_value
            if self.adjust_concentrations and self.x[i] < GbaConstants.TOL.value:
                self.x[i] = GbaConstants.TOL.value

    def set_q0( self, q0: np.array ) -> None:
        """
        Set the initial flux fraction vector q0.
        
        Parameters
        ----------
        q0 : np.array
            Initial flux fraction vector.
        """
        assert len(q0) == self.nj, throw_message(MessageType.ERROR, "Incorrect q0 length.")
        self.q0      = np.copy(q0)
        self.q_trunc = np.copy(self.q0[1:self.nj])
        self.q       = np.copy(self.q0)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 4) Analytical methods              #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    def compute_c( self ) -> None:
        """
        Compute the internal metabolite concentrations.
        """
        self.c = self.rho*self.M.dot(self.q)
        if self.adjust_concentrations:
            self.c[self.c < GbaConstants.TOL.value] = GbaConstants.TOL.value
        self.xc = np.concatenate([self.x, self.c])
    
    def iMM( self, j: int ) -> None:
        """
        Compute the turnover time tau for an irreversible Michaelis-Menten
        reaction.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        KM_prod       = np.prod(1.0+self.KM_f[:,j]/self.xc)
        kcatf         = self.kcat_f[j]
        self.tau_j[j] = KM_prod/kcatf

    def iMMi( self, j: int ) -> None:
        """
        Compute the turnover time tau for an irreversible Michaelis-Menten
        reaction with inhibition (only one inhibitor per reaction).

        Parameters
        ----------
        j : int
            Reaction index.
        """
        KI_prod       = np.prod(1.0+self.xc*self.rKI[:,j])
        KM_prod       = np.prod(1.0+self.KM_f[:,j]/self.xc)
        kcatf         = self.kcat_f[j]
        self.tau_j[j] = KI_prod*KM_prod/kcatf
    
    def iMMa( self, j: int ) -> None:
        """
        Compute the turnover time tau for an irreversible Michaelis-Menten
        reaction with activation (only one activator per reaction).

        Parameters
        ----------
        j : int
            Reaction index.
        """
        KA_prod       = np.prod(1.0+self.KA[:,j]/self.xc)
        KM_prod       = np.prod(1.0+self.KM_f[:,j]/self.xc)
        kcatf         = self.kcat_f[j]
        self.tau_j[j] = KA_prod*KM_prod/kcatf
    
    def iMMia( self, j: int ) -> None:
        """
        Compute the turnover time tau for an irreversible Michaelis-Menten
        reaction with inhibition and activation (only one inhibitor and one
        activator per reaction).

        Parameters
        ----------
        j : int
            Reaction index.
        """
        KI_prod       = np.prod(1.0+self.xc*self.rKI[:,j])
        KA_prod       = np.prod(1.0+self.KA[:,j]/self.xc)
        KM_prod       = np.prod(1.0+self.KM_f[:,j]/self.xc)
        kcatf         = self.kcat_f[j]
        self.tau_j[j] = KI_prod*KA_prod*KM_prod/kcatf
    
    def rMM( self, j: int ) -> None:
        """
        Compute the turnover time tau for a reversible Michaelis-Menten
        reaction.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        KMf_prod      = np.prod(1+self.KM_f[:,j]/self.xc)
        KMb_prod      = np.prod(1+self.KM_b[:,j]/self.xc)
        kcatb         = self.kcat_b[j]
        kcatf         = self.kcat_f[j]
        self.tau_j[j] = 1.0/(kcatf/KMf_prod-kcatb/KMb_prod)

    def compute_tau( self, j: int ) -> None:
        """
        Compute the turnover time tau for a reaction j.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        if self.kinetic_model[j] == GbaReactionType.IMM:
            self.iMM(j)
        elif self.kinetic_model[j] == GbaReactionType.IMMI:
            self.iMMi(j)
        elif self.kinetic_model[j] == GbaReactionType.IMMA:
            self.iMMa(j)
        elif self.kinetic_model[j] == GbaReactionType.IMMIA:
            self.iMMia(j)
        elif self.kinetic_model[j] == GbaReactionType.RMM:
            self.rMM(j)
    
    def diMM( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for an irreversible
        Michaelis-Menten reaction with respect to metabolite concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        kcatf = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            term1             = self.KM_f[y,j]/np.power(self.c[i], 2.0)
            term2             = np.prod(1+self.KM_f[indices,j]/self.xc[indices])
            self.ditau_j[j,i] = -term1*term2/kcatf

    def diMMi( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for an irreversible
        Michaelis-Menten reaction with inhibition with respect to metabolite
        concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        KI_prod = np.prod(1+self.xc*self.rKI[:,j])
        KM_prod = np.prod(1+self.KM_f[:,j]/self.xc)
        kcatf   = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            rKI               = self.rKI[y,j]
            term1             = self.KM_f[y,j]/np.power(self.c[i], 2)
            term2             = np.prod(1+self.KM_f[indices,j]/self.xc[indices])
            self.ditau_j[j,i] = (rKI*KM_prod - KI_prod*term1*term2)/kcatf

    def diMMa( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for an irreversible
        Michaelis-Menten reaction with activation with respect to metabolite
        concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        KA_prod = np.prod(1+self.KA[:,j]/self.xc)
        KM_prod = np.prod(1+self.KM_f[:,j]/self.xc)
        kcatf   = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            term1             = self.KA[y,j]/np.power(self.c[i], 2.0)
            term2             = self.KM_f[y,j]/np.power(self.c[i], 2.0)
            term3             = np.prod(1+self.KM_f[indices,j]/self.xc[indices])
            self.ditau_j[j,i] = -(term1*KM_prod + term2*KA_prod*term3)/kcatf
    
    def diMMia( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for an irreversible
        Michaelis-Menten reaction with activation and inhibition with respect to
        metabolite concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        KI_prod = np.prod(1.0+self.c*self.rKI[:,j])
        KA_prod = np.prod(1.0+self.KA[:,j]/self.c)
        KM_prod = np.prod(1.0+self.KM_f[:,j]/self.c)
        kcatf   = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            rKI               = self.rKI[y,j]
            term2             = -self.KA[y,j]/np.power(self.c[i], 2.0)
            term3             = -self.KM_f[y,j]/np.power(self.c[i], 2.0)
            term4             = np.prod(1+self.KM_f[indices,j]/self.c[indices])
            self.ditau_j[j,i] = (rKI*KA_prod*KM_prod + KI_prod*term2*KM_prod + KI_prod*KA_prod*term3*term4)/kcatf 

    def drMM( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for a reversible
        Michaelis-Menten reaction with respect to metabolite concentrations.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        KMf_prod = np.prod(1+self.KM_f[:,j]/self.xc)
        KMb_prod = np.prod(1+self.KM_b[:,j]/self.xc)
        kcatf    = self.kcat_f[j]
        kcatb    = self.kcat_b[j]
        tau_j    = 1.0/(kcatf/KMf_prod-kcatb/KMb_prod)
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            prodf             = np.prod(1 + self.KM_f[indices,j]/self.xc[indices])
            prodb             = np.prod(1 + self.KM_b[indices,j]/self.xc[indices])
            term1             = self.KM_f[y,j] / np.power(self.c[i] + self.KM_f[y,j], 2.0)
            term2             = self.KM_b[y,j] / np.power(self.c[i] + self.KM_b[y,j], 2.0)
            ditauj            = (kcatf/prodf)*term1 - (kcatb/prodb)*term2
            self.ditau_j[j,i] = -ditauj*np.power(tau_j, 2.0)

    def compute_dtau( self, j: int ) -> None:
        """
        Compute the derivative of the turnover time tau for a reaction j.

        Parameters
        ----------
        j : int
            Reaction index.
        """
        if self.kinetic_model[j] == GbaReactionType.IMM:
            self.diMM(j)
        elif self.kinetic_model[j] == GbaReactionType.IMMI:
            self.diMMi(j)
        elif self.kinetic_model[j] == GbaReactionType.IMMA:
            self.diMMa(j)
        elif self.kinetic_model[j] == GbaReactionType.IMMIA:
            self.diMMia(j)
        elif self.kinetic_model[j] == GbaReactionType.RMM:
            self.drMM(j)
    
    def compute_mu( self ) -> None:
        """
        Compute the growth rate mu.
        """
        self.mu            = self.M[self.a,self.r]*self.q[self.r]/(self.tau_j.dot(self.q))
        self.doubling_time = np.log(2)/np.log(1+self.mu)

    def compute_v( self ) -> None:
        """
        Compute the fluxes v.
        """
        self.v = self.mu*self.rho*self.q

    def compute_p( self ) -> None:
        """
        Compute the protein concentrations p.
        """
        self.p = self.tau_j*self.v

    def compute_b( self ) -> None:
        """
        Compute the biomass fractions b.
        """
        self.b = self.M.dot(self.q)

    def compute_density( self ) -> None:
        """
        Compute the cell density (should be equal to 1).
        """
        self.density = self.sM.dot(self.q)

    def compute_dmu_dq( self ) -> None:
        """
        Compute the local growth rate gradient with respect to q.
        """
        term1       = np.power(self.mu, 2)/self.b[self.a]
        term2       = self.M[self.a,:]/self.mu
        term3       = self.q.T.dot(self.rho*self.ditau_j.dot(self.M))
        term4       = self.tau_j
        self.dmu_dq = term1*(term2-term3-term4)

    def compute_Gamma( self ) -> None:
        """
        Compute the local growth control coefficients with respect to q.
        """
        self.Gamma = self.dmu_dq-self.dmu_dq[0]*(self.sM/self.sM[0])
    
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
        self.compute_dmu_dq()
        self.compute_Gamma()
    
    def calculate( self ) -> None:
        """
        Calculate the model state.
        """
        self.calculate_first_order_terms()
        self.calculate_second_order_terms()

    def check_consistency( self ) -> None:
        """
        Check the model state's consistency.
        """
        test1           = (np.abs(self.density-1.0) < GbaConstants.TOL.value)
        test2           = (sum(1 for x in self.c if x < 0.0) == 0)
        test3           = (sum(1 for x in self.p if x < 0.0) == 0)
        self.consistent = True
        if not (test1 and test2 and test3):
            self.consistent = False

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 5) Generation of initial solutions #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def solve_local_linear_problem_deprecated( self, max_flux_fraction: Optional[float] = 50.0, rhs_factor: Optional[float] = 1000.0 ) -> bool:
        throw_message(MessageType.ERROR, "Deprecated function")
        """
        Solve the local linear problem to find the initial solution.

        Description
        -----------
        The local linear problem consists in finding the maximal ribosome flux
        fraction q^r, with a minimal production of each metabolite. The
        constraints are mass conservation (M*q = b) and surface flux balance
        (sM*q = 1).

        Parameters
        ----------
        max_flux_fraction : Optional[float], default=50.0
            Maximal flux fraction.
        rhs_factor : Optional[float], default=1000.0
            Factor dividing the rhs of the mass conservation constraint.
        """
        assert max_flux_fraction > GbaConstants.TOL.value, throw_message(MessageType.ERROR, f"Maximal flux fraction must be greater than {GbaConstants.TOL.value}.")
        assert rhs_factor > 0.0, throw_message(MessageType.ERROR, "RHS factor must be positive.")
        lb_vec = []
        for j in range(self.nj):
            if self.reversible[j]:
                lb_vec.append(-gp.GRB.INFINITY)
                #lb_vec.append(-max_flux_fraction)
            else:
                lb_vec.append(0.0)
        ub_vec = [gp.GRB.INFINITY]*self.nj
        #ub_vec = [max_flux_fraction]*self.nj
        for item in self.constant_reactions.items():
           r_index         = self.reaction_ids.index(item[0])
           lb_vec[r_index] = item[1]
           ub_vec[r_index] = item[1]
        gpmodel = gp.Model(env=env)
        v       = gpmodel.addMVar(self.nj, lb=lb_vec, ub=ub_vec)
        min_b   = 1/rhs_factor
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
            throw_message(MessageType.ERROR, "Local linear problem could not be solved.")
            return False

    def find_initial_solution_deprecated( self, condition_id: Optional[str] = "1", max_flux_fraction: Optional[float] = 50.0, rhs_factor: Optional[float] = 1000.0 ) -> None:
        throw_message(MessageType.ERROR, "Deprecated function")
        """
        Generate an initial solution using a linear program.

        Parameters
        ----------
        condition_id : Optional[str], default="1"
            Condition identifier.
        max_flux_fraction : Optional[float], default=50.0
            Maximal flux fraction.
        rhs_factor : Optional[float], default=1000.0
            Factor dividing the rhs of the mass conservation constraint.
        """
        solved = self.solve_local_linear_problem(max_flux_fraction=max_flux_fraction, rhs_factor=rhs_factor)
        if solved:
            self.set_condition(condition_id)
            self.set_q0(self.initial_solution)
            self.calculate()
            self.check_consistency()
            if self.consistent:
                throw_message(MessageType.INFO, f"Model is consistent with mu = {self.mu}.")
            else:
                throw_message(MessageType.INFO, "Model is inconsistent.")
        else:
            throw_message(MessageType.WARNING, "Impossible to find an initial solution.")

    ######################################

    def delete_reaction( self, reaction_id: str ) -> None:
        """
        Delete a reaction from the model.

        Parameters
        ----------
        reaction_id : str
            Identifier of the reaction to delete.
        """
        position    = self.reaction_ids.index(reaction_id)
        self.Mx     = np.delete(self.Mx, position, axis=1)
        self.M      = np.delete(self.M, position, axis=1)
        self.kcat_f = np.delete(self.kcat_f, position)
        self.kcat_b = np.delete(self.kcat_b, position)
        self.K      = np.delete(self.K, position, axis=1)
        self.KM_f   = np.delete(self.KM_f, position, axis=1)
        self.KM_b   = np.delete(self.KM_b, position, axis=1)
        self.KA     = np.delete(self.KA, position, axis=1)
        self.KI     = np.delete(self.KI, position, axis=1)
        self.rKI    = np.delete(self.rKI, position, axis=1)
        self.reaction_ids.pop(position)
        self.reversible.pop(position)
        self.kinetic_model.pop(position)
        self.directions.pop(position)
        if reaction_id in self.constant_reactions:
            self.constant_reactions.pop(reaction_id)
        self.initialize_model_mathematical_variables()
        
    def solve_q0_linear_problem( self, min_bp: Optional[float] = 0.2, sat_act: Optional[float] = 1.0, slack: Optional[float] = 2.0, verbose: Optional[bool] = False ) -> bool:
        """
        Find an initial solution.
        
        Description
        -----------
        Solution updated from Hugo Dourado.
        
        Parameters
        ----------
        min_bp : Optional[float], default=0.2
            Minimal protein production.
        sat_act : Optional[float], default=1.0
            Saturation of the metabolic enzymes.
        slack : Optional[float], default=2.0
            Slack variable for the minimal metabolite production.
        verbose : Optional[bool], default=False
            Verbose mode.
        
        Returns
        -------
        bool
            True if a consistent solution is found, False otherwise.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Define q boundaries     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        lb_vec = []
        for j in range(self.nj):
            if self.reversible[j]:
                lb_vec.append(-gp.GRB.INFINITY)
            else:
                lb_vec.append(GbaConstants.TOL.value)
        lb_vec = [GbaConstants.TOL.value]*self.nj    
        ub_vec = [gp.GRB.INFINITY]*self.nj
        for item in self.constant_reactions.items():
           r_index         = self.reaction_ids.index(item[0])
           lb_vec[r_index] = item[1]
           ub_vec[r_index] = item[1]
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Define RHS term         #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        aKm             = self.K[self.nx:, :].sum(axis=1).astype(float)
        aKm[aKm < 1e-4] = 1e-4
        w               = aKm / aKm.sum()
        min_bm          = (1.0-min_bp)*w[:-1]/slack
        act             = self.KA[self.nx:, :].sum(axis=1)*(sat_act/self.rho)
        rhs             = np.concatenate([min_bm + act[:-1], [min_bp]])
        for m_id, value in self.constant_rhs.items():
            rhs[self.c_ids.index(m_id)] = value
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Run optimization        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        gpmodel = gp.Model(env=env)
        q       = gpmodel.addMVar(self.nj, lb=lb_vec, ub=ub_vec, name="q")
        obj     = (1.0/self.kcat_f) @ q
        gpmodel.setObjective(obj, gp.GRB.MINIMIZE)
        gpmodel.addConstr(self.M @ q >= rhs, name="flux_balance")
        gpmodel.addConstr(self.sM @ q == 1.0, name="density")
        gpmodel.optimize()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Get the solution if any #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        try:
            self.initial_solution = np.copy(q.X)
            return True
        except:
            if verbose:
                throw_message(MessageType.ERROR, "Local linear problem could not be solved.")
            return False
                
    def find_q0( self, condition_id: Optional[str] = "1", param_exploration: Optional[bool] = True, min_bp: Optional[float] = 0.2, sat_act: Optional[float] = 1.0, slack: Optional[float] = 2.0, verbose: Optional[bool] = False ) -> bool:
        """
        Find an initial solution.
        
        Description
        -----------
        Inspired from the solution developed by Hugo Dourado.
        
        Parameters
        ----------
        condition_id : Optional[str], default="1"
            Condition identifier.
        min_bp : Optional[float], default=0.2
            Minimal protein production.
        param_exploration : Optional[bool], default=True
            Parameter exploration mode.
        sat_act : Optional[float], default=1.0
            Saturation of the metabolic enzymes.
        slack : Optional[float], default=2.0
            Slack variable for the minimal metabolite production.
        verbose : Optional[bool], default=False
            Verbose mode.
        
        Returns
        -------
        bool
            True if a consistent solution is found, False otherwise.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Explore saturation and slack if required #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if param_exploration:
            sat_act = 1.0
            slack   = 1.0
            solved  = False
            while slack < 100.0:
                solved = self.solve_q0_linear_problem(min_bp=min_bp, sat_act=sat_act, slack=slack)
                if solved:
                    break
                else:
                    sat_act /= 1.1
                    slack   *= 1.1
            if solved:
                self.set_condition(condition_id)
                self.set_q0(self.initial_solution)
                self.calculate()
                self.check_consistency()
                if self.consistent:
                    if verbose:
                        throw_message(MessageType.INFO, f"Model is consistent with mu = {self.mu}.")
                    return True
                else:
                    if verbose:
                        throw_message(MessageType.INFO, "Model is inconsistent.")
                    return False
            else:
                if verbose:
                    throw_message(MessageType.WARNING, "Impossible to find an initial solution.")
                return False
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Else direct optimization                 #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        else:
            solved = self.solve_q0_linear_problem(min_bp=min_bp, sat_act=sat_act, slack=slack)
            if solved:
                self.set_condition(condition_id)
                self.set_q0(self.initial_solution)
                self.calculate()
                self.check_consistency()
                if self.consistent:
                    if verbose:
                        throw_message(MessageType.INFO, f"Model is consistent with mu = {self.mu}.")
                    return True
                else:
                    if verbose:
                        throw_message(MessageType.INFO, "Model is inconsistent.")
                    return False
            else:
                if verbose:
                    throw_message(MessageType.WARNING, "Impossible to find an initial solution.")
                return False
    
    def find_initial_solution( self, condition_id: Optional[str] = "1", verbose: Optional[bool] = False ) -> bool:
        """
        Generate the best initial solution by scanning the minimal protein production.
        
        Description
        -----------
        Solution developed by Hugo Dourado.

        Parameters
        ----------
        condition_id : Optional[str], default="1"
            Condition identifier.
        verbose : Optional[bool], default=False
            Verbose mode.
        
        Returns
        -------
        bool
            True if a consistent solution is found, False otherwise.
        """
        min_bp     = 0.01
        step       = 0.01
        mu_max     = 0.0
        min_bp_max = min_bp
        max_found  = False
        while not max_found and min_bp < 1.0:
            found = self.find_q0(condition_id=condition_id, min_bp=min_bp, verbose=False)
            if found:
                if self.mu > mu_max:
                    mu_max     = self.mu
                    min_bp_max = min_bp
                    min_bp     = min_bp + step
                else:
                    max_found = True
            else:
                min_bp = min_bp + step
        if max_found:
            self.find_q0(condition_id=condition_id, min_bp=min_bp_max, verbose=verbose)
            return True
        else:
            if verbose:
                throw_message(MessageType.WARNING, "Impossible to find an initial solution.")
            return False

    def detect_inactive_reactions( self, threshold: Optional[float] = GbaConstants.TOL.value ) -> list[str]:
        """
        Detect inactive reactions in the initial solution.

        Parameters
        ----------
        threshold : Optional[float], default=1e-6
            Threshold below which a reaction is considered inactive.

        Returns
        -------
        list[str]
            List of inactive reaction identifiers.
        """
        self.find_initial_solution()
        inactive_reactions = []
        for j in range(self.nj):
            if abs(self.q0[j]) <= threshold:
                inactive_reactions.append(self.reaction_ids[j])
        return inactive_reactions

    def detect_non_essential_reactions( self, min_bp: Optional[float] = None, verbose: Optional[bool] = False ) -> dict[str, float]:
        """
        Detect non-essential reactions in the model.

        Parameters
        ----------
        min_bp : Optional[float], default=None
            Minimal protein production to consider when finding the initial
            solution. If None, the standard initial solution method is used.
        verbose : Optional[bool], default=False
            Verbose mode.
        Returns
        -------
        dict[str, float]
            Dictionary of non-essential reactions with their corresponding mu.
        """
        non_essential_reactions = {}
        for j in range(self.nj-1):
            reaction_id = self.reaction_ids[j]
            #if verbose:
            #    throw_message(MessageType.PLAIN, f"> Testing reaction {reaction_id}... ({j+1}/{self.nj-1})...")
            my_model = copy.deepcopy(self)
            my_model.delete_reaction(reaction_id)
            solution_exists = False
            if min_bp is None:
                solution_exists = my_model.find_initial_solution()
            else:
                solution_exists = my_model.find_q0(min_bp=min_bp)
            if solution_exists and my_model.mu > GbaConstants.TOL.value:
                if verbose:
                    throw_message(MessageType.INFO, f"Reaction <code>{reaction_id}</code> is non-essential (mu = {my_model.mu})")
                non_essential_reactions[self.reaction_ids[j]] = my_model.mu
            del(my_model)
        if verbose and len(non_essential_reactions) == 0:
            throw_message(MessageType.INFO, "No non-essential reaction was found.")
        return non_essential_reactions
    
    def generate_random_initial_solutions( self, condition_id: str, nb_solutions: int, max_trials: int, max_flux_fraction: Optional[float] = 10.0, min_mu: Optional[float] = 1e-3, verbose: Optional[bool] = False ) -> None:
        throw_message(MessageType.ERROR, "Deprecated function")
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
        assert condition_id in self.condition_ids, throw_message(MessageType.ERROR, f"Unknown condition identifier (<code>{condition_id}</code>).")
        assert nb_solutions > 0, throw_message(MessageType.ERROR, f"Number of solutions must be greater than 0.")
        assert max_trials >= nb_solutions, throw_message(MessageType.ERROR, f"Number of trials must be greater than the number of solutions.")
        assert max_flux_fraction > GbaConstants.TOL.value, throw_message(MessageType.ERROR, f"Maximal flux fraction must be greater than {GbaConstants.TOL.value}.")
        assert min_mu >= 0.0, throw_message(MessageType.ERROR, f"Minimal growth rate must be positive.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the random data frame #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        columns          = ["condition"] + self.reaction_ids + ["mu"]
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
                self.q_trunc = np.random.rand(self.nj-1)
                self.q_trunc = self.q_trunc*(max_flux_fraction-GbaConstants.TOL)+GbaConstants.TOL
                self.set_q_from_q_trunc()
                if self.q[0] >= 0.0:
                    negative_term = False
            self.calculate_state()
            self.check_consistency()
            if self.consistent and np.isfinite(self.mu) and self.mu > min_mu:
                solutions += 1
                data_dict  = {"condition": condition_id, "mu": self.mu}
                for reaction_id, fluxfraction in zip(self.reaction_ids, self.q):
                    data_dict[reaction_id] = fluxfraction
                data_row                         = pd.Series(data=data_dict)
                self.random_data                 = pd.concat([self.random_data, data_row.to_frame().T], ignore_index=True)
                self.random_solutions[solutions] = np.copy(self.q)
                if verbose:
                    throw_message(MessageType.PLAIN, f"{solutions} solutions were found after {trials} trials (last mu = {round(self.mu,5)}).")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 6) Optimization functions          #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def build_gbacpp_command_line( self, temporary_name: str, condition_id: str, use_previous_sol: bool, tol: float, mutol: float, convergence_count: int, max_iter: float ) -> str:
        """
        Build the command line for the C++ solver gbacpp.

        Parameters
        ----------
        temporary_name : str
            Name of the temporary model.
        condition_id : str
            Condition identifier.
        use_previous_sol : bool
            Use the previous solution as initial solution.
        tol : float
            Tolerance value.
        mutol : float
            Tolerance value for mu relative change.
        convergence_count : int
            Number of iteration with no significant mu change to assume
            convergence.
        max_iter : float
            Maximum number of iterations.
        
        Returns
        -------
        cmdline
            The solver command line
        """
        cmdline  = "find_model_optimum "
        cmdline += "-path . "
        cmdline += "-name "+str(temporary_name)+" "
        cmdline += "-condition "+condition_id+" "
        cmdline += "-output "+str(temporary_name)+" "
        cmdline += "-tol "+str(tol)+" "
        cmdline += "-mutol "+str(mutol)+" "
        cmdline += "-conv "+str(convergence_count)+" "
        cmdline += "-max "+str(max_iter)+" "
        if use_previous_sol:
            cmdline += "-previous "
        cmdline += "-optimum\n"
        return(cmdline)
    
    def read_solver_output( self, temporary_name: str, condition_id: str ) -> None:
        """
        Read the output of the solver and update the model state.

        Parameters
        ----------
        solver_output : str
            Output of the solver.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Prepare and check filenames #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        state_filename = ""
        q_filename     = ""
        v_filename     = ""
        p_filename     = ""
        b_filename     = ""
        c_filename     = ""
        if condition_id == "all":
            state_filename = "./"+temporary_name+"/"+temporary_name+"_all_state_optimum.csv"
            q_filename     = "./"+temporary_name+"/"+temporary_name+"_all_q_optimum.csv"
            v_filename     = "./"+temporary_name+"/"+temporary_name+"_all_v_optimum.csv"
            p_filename     = "./"+temporary_name+"/"+temporary_name+"_all_p_optimum.csv"
            b_filename     = "./"+temporary_name+"/"+temporary_name+"_all_b_optimum.csv"
            c_filename     = "./"+temporary_name+"/"+temporary_name+"_all_c_optimum.csv"
        else:
            state_filename = "./"+temporary_name+"/"+temporary_name+"_"+condition_id+"_state_optimum.csv"
            q_filename     = "./"+temporary_name+"/"+temporary_name+"_"+condition_id+"_q_optimum.csv"
            v_filename     = "./"+temporary_name+"/"+temporary_name+"_"+condition_id+"_v_optimum.csv"
            p_filename     = "./"+temporary_name+"/"+temporary_name+"_"+condition_id+"_p_optimum.csv"
            b_filename     = "./"+temporary_name+"/"+temporary_name+"_"+condition_id+"_b_optimum.csv"
            c_filename     = "./"+temporary_name+"/"+temporary_name+"_"+condition_id+"_c_optimum.csv"
        assert os.path.exists(state_filename), throw_message(MessageType.ERROR, f"Solver state output file <code>{state_filename}</code> not found.")
        assert os.path.exists(q_filename), throw_message(MessageType.ERROR, f"Solver q output file <code>{q_filename}</code> not found.")
        assert os.path.exists(v_filename), throw_message(MessageType.ERROR, f"Solver v output file <code>{v_filename}</code> not found.")
        assert os.path.exists(p_filename), throw_message(MessageType.ERROR, f"Solver p output file <code>{p_filename}</code> not found.")
        assert os.path.exists(b_filename), throw_message(MessageType.ERROR, f"Solver b output file <code>{b_filename}</code> not found.")
        assert os.path.exists(c_filename), throw_message(MessageType.ERROR, f"Solver c output file <code>{c_filename}</code> not found.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Read the optimal q vector   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f      = open(q_filename, "r")
        header = f.readline().strip("\n").split(";")
        l      = f.readline()
        while l:
            l         = l.strip("\n").split(";")
            condition = l[0]
            values    = l[1:]
            self.optimal_solutions[condition] = np.array([float(v) for v in values])
            self.q = self.optimal_solutions[condition]
            l      = f.readline()
        f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Load optimization data      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.data              = pd.DataFrame(self.conditions.T, columns=self.condition_params)
        self.data["condition"] = self.condition_ids
        data_files             = [state_filename, q_filename, v_filename, p_filename, b_filename, c_filename]
        suffixes               = ["state", "q", "v", "p", "b", "c"]
        for i in range(len(data_files)):
            file   = data_files[i]
            suffix = suffixes[i]
            df     = pd.read_csv(file, sep=";")
            if suffix != "state":
                new_columns = {"condition": "condition"}
                for col in df.columns:
                    if col != "condition":
                        new_columns[col] = col+"_"+suffix
                df = df.rename(columns=new_columns)
            self.data["condition"] = self.data["condition"].astype(str)
            df["condition"]        = df["condition"].astype(str)
            self.data              = pd.merge(self.data, df, on="condition")
    
    def find_optimum( self, tol: Optional[float] = 1e-10, mutol: Optional[float] = 1e-10, convergence_count: Optional[int] = 10000, max_iter: Optional[int] = 100000000, delete: Optional[bool] = True, verbose: Optional[bool] = False ) -> None:
        """
        Find the optimum of the model using the gbacpp solver.

        Parameters
        ----------
        tol : Optional[float], default=1e-10
            Tolerance value for the solver.
        mutol : Optional[float], default=1e-10
            Tolerance value for the growth rate relative difference.
        convergence_count : Optional[int], default=10000
            Number of iterations with no significant mu change to
            assume convergence.
        max_iter : Optional[int], default=100000000
            Maximum number of iterations for the solver.
        delete : Optional[bool], default=True
            Delete temporary files.
        verbose : Optional[bool], default=True
            Verbose mode.
        
        Returns
        -------
        consistent : bool
            True if the model is consistent, False otherwise.
        """
        assert os.path.exists("find_model_optimum"), throw_message(MessageType.ERROR, "The gbacpp solver 'find_model_optimum' is not available. Please check your installation, or your PATH variable.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Write the model in a temporary file with a unique key #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        random.seed(int(time.time()))
        random_number  = random.randint(0, 100000000)
        temporary_name = "temp_"+str(int(random_number))
        self.write_to_csv(name=temporary_name)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Run gbacpp solver                                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        cmdline        = self.build_gbacpp_command_line(temporary_name, self.condition, False, tol, mutol, convergence_count, max_iter)
        solver_process = subprocess.Popen([cmdline], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        solver_process.wait()
        #self.read_solver_output(temporary_name=temporary_name, condition_id=self.condition)
        try:
            self.read_solver_output(temporary_name=temporary_name, condition_id=self.condition)
        except:
            throw_message(MessageType.WARNING, "Impossible to read the solver output.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Remove the temporary files and folder                 #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if delete:
            assert temporary_name.startswith("temp_"), throw_message(MessageType.ERROR, "Temporary folder name must start with 'temp_'.")
            assert temporary_name.split("_")[1].isdigit(), throw_message(MessageType.ERROR, "Temporary folder name must contain a timestamp.")
            while os.path.exists(temporary_name):
                files = os.listdir(temporary_name)
                for file in files:
                    file_path = os.path.join(temporary_name, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(temporary_name)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Update the model                                      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.set_q0(self.optimal_solutions[self.condition])
        self.calculate()
        self.check_consistency()
        if self.consistent:
            if verbose:
                throw_message(MessageType.INFO, f"Condition {self.condition}: model converged with mu = {self.mu}.")
        else:
            if verbose:
                throw_message(MessageType.INFO, f"Condition {self.condition}: model did not converge.")
        
    def find_optimum_by_condition( self, use_previous_sol: Optional[bool] = True, tol: Optional[float] = 1e-10, mutol: Optional[float] = 1e-10, convergence_count: Optional[int] = 10000, max_iter: Optional[int] = 10000000, delete: Optional[bool] = True, verbose: Optional[bool] = False ) -> None:
        """
        Find optimums for all conditions of the model using the gbacpp solver.

        Parameters
        ----------
        use_previous_sol : Optional[bool], default=False
            Use the previous solution as initial solution for the next
            condition.
        tol : Optional[float], default=1e-10
            Tolerance value for the solver.
        mutol : Optional[float], default=1e-10
            Tolerance value for the growth rate relative difference.
        convergence_count : Optional[int], default=10000
            Number of iterations with no significant mu change to
            assume convergence.
        max_iter : Optional[int], default=1000000
            Maximum number of iterations for the solver.
        delete : Optional[bool], default=True
            Delete temporary files.
        verbose : Optional[bool], default=True
            Verbose mode.
        
        Returns
        -------
        consistent : bool
            True if the model is consistent, False otherwise.
        """
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Write the model in a temporary file with a unique key #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        random.seed(int(time.time()))
        random_number  = random.randint(0, 100000000)
        temporary_name = "temp_"+str(int(random_number))
        self.write_to_csv(name=temporary_name)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Run gbacpp solver                                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        cmdline        = self.build_gbacpp_command_line(temporary_name, "all", use_previous_sol, tol, mutol, convergence_count, max_iter)
        solver_process = subprocess.Popen([cmdline], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        solver_process.wait()
        #self.read_solver_output(temporary_name=temporary_name, condition_id="all")
        try:
            self.read_solver_output(temporary_name=temporary_name, condition_id="all")
        except:
            throw_message(MessageType.WARNING, "Impossible to read the solver output.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Remove the temporary files and folder                 #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if delete:
            assert temporary_name.startswith("temp_"), throw_message(MessageType.ERROR, "Temporary folder name must start with 'temp_'.")
            assert temporary_name.split("_")[1].isdigit(), throw_message(MessageType.ERROR, "Temporary folder name must contain a timestamp.")
            while os.path.exists(temporary_name):
                files = os.listdir(temporary_name)
                for file in files:
                    file_path = os.path.join(temporary_name, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(temporary_name)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Update the model                                      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for cond_id in self.optimal_solutions.keys():
            self.set_q0(self.optimal_solutions[cond_id])
            self.set_condition(cond_id)
            self.calculate()
            self.check_consistency()
            if self.consistent:
                if verbose:
                    throw_message(MessageType.INFO, f"Condition {cond_id}: model converged with mu = {self.mu}.")
            else:
                if verbose:
                    throw_message(MessageType.INFO, f"Condition {cond_id}: model did not converge.")
    
    def score( self, p1, p2 ) -> float:
        return self.mu*(1.0+np.sum([p1 for c in self.c if c < 0.0]))*(1.0+np.sum([p2 for p in self.p if p < 0.0]))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 7) Plotting functions              #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def plot( self, x: str, y: str, logx: Optional[bool] = False, logy: Optional[bool] = False, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None ) -> None:
        """
        Plot two parameters from the model data.

        Parameters
        ----------
        x : str
            Name of the x parameter.
        y : str
            Name of the y parameter.
        logx : Optional[bool], default=False
            Use a logarithmic scale for the x axis.
        logy : Optional[bool], default=False
            Use a logarithmic scale for the y axis.
        title : Optional[str], default=None
            Title of the plot.
        """
        assert x in self.data.columns, throw_message(MessageType.ERROR, f"Unknown x parameter <code>{x}</code>.")
        assert y in self.data.columns, throw_message(MessageType.ERROR, f"Unknown y parameter <code>{y}</code>.")
        if title is None:
            title = y+" vs "+x
        if xlabel is None:
            xlabel = x
        if ylabel is None:
            ylabel = y
        fig = px.line(self.data, x=x, y=y, title=title, labels={x: xlabel, y: ylabel}, template="plotly_white")
        if logx:
            fig.update_xaxes(type="log")
        if logy:
            fig.update_yaxes(type="log")
        fig.show()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 8) Summary functions               #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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
            html_str += df.to_html(escape=False, index=False)
            html_str += "</td></tr>"
            html_str += "</table>"
        display_html(html_str,raw=True)

    def summary( self ) -> None:
        """
        Print a summary of the model.
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
            "Category": ["Nb reactions", "Nb transporters", "Nb internal reactions"],
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
        html_str  = "<h1>Model "+self.name+" summary</h1>"
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

def read_csv_model( name: str, path: Optional[str] = "." ) -> Model:
    """
    Read a model from CSV files.

    Parameters
    ----------
    name : str
        Name of the model.
    path : Optional[str], default="."
        Path to the model folder.

    Returns
    -------
    Model
        The loaded model.
    """
    assert os.path.exists(path+"/"+name), throw_message(MessageType.ERROR, "The folder "+path+"/"+name+" does not exist.")
    model = Model(name)
    model.read_from_csv(path=path)
    return model

def read_ods_model( name: str, path: Optional[str] = "." ) -> Model:
    """
    Read a model from ODS files.

    Parameters
    ----------
    name : str
        Name of the model.
    path : Optional[str], default="."
        Path to the model folder.

    Returns
    -------
    Model
        The loaded model.
    """
    assert os.path.exists(path+"/"+name+".ods"), throw_message(MessageType.ERROR, "The folder "+path+"/"+name+".ods does not exist.")
    model = Model(name)
    model.read_from_ods(path=path)
    return model

def backup_model( model: Model, name: Optional[str] = "", path: Optional[str] = "." ) -> None:
    """
    Backup a model in binary format (extension .gba).

    Parameters
    ----------
    model : Model
        Model to backup.
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
    pickle.dump(model, ofile)
    ofile.close()
    assert os.path.isfile(filename), throw_message(MessageType.ERROR, ".gba file creation failed.")

def load_model( path: str ) -> Model:
    """
    Load a model from a binary file.

    Parameters
    ----------
    path : str
        Path to the model file.
    """
    assert path.endswith(".gba"), throw_message(MessageType.ERROR, "Model file extension is missing.")
    assert os.path.isfile(path), throw_message(MessageType.ERROR, "Model file not found.")
    ifile = open(path, "rb")
    model = pickle.load(ifile)
    ifile.close()
    return model

