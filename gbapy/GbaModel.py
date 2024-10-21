#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# gbapy
# -----
# Growth balance analysis in Python.
# 
# Copyright © 2023-2024 Charles Rocabert, Furkan Mert
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
import matplotlib.pyplot as plt

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

### Define constant and tolerance thresholds ###
MIN_CONCENTRATION            = 1e-10 # Minimum concentration value
MIN_FLUX_FRACTION            = 1e-10 # Minimum flux fraction value
MAX_FLUX_FRACTION            = 2.0   # Maximum flux fraction value
DENSITY_TOL                  = 1e-10 # Density tolerance threshold (|1-rho| < ε)
NEGATIVE_C_TOL               = 1e-10 # Negative C tolerance threshold (C > -ε)
NEGATIVE_P_TOL               = 1e-10 # Negative P tolerance threshold (P > -ε)
TRAJECTORY_CONVERGENCE_COUNT = 1000  # Number of iterations with equal mu values to consider the trajectory stable
TRAJECTORY_CONVERGENCE_TOL   = 1e-10 # Mu threshold below which growth rates are considered equal
DECREASING_DT_FACTOR         = 5.0   # Factor by which the time step is decreased when the trajectory is unstable
INCREASING_DT_FACTOR         = 2.0   # Factor by which the time step is increased when the trajectory is stable
INCREASING_DT_COUNT          = 100   # Number of iterations with equal mu values to increase the time step
EXPORT_DATA_COUNT            = 1     # Frequency of data export

### Backup a binary .gba model ###
def backup_gba_model( model, path = "" ):
    filename = model.name+".gba"
    if path != "":
        filename = path+"/"+model.name+".gba"
    ofile = open(filename, "wb")
    dill.dump(model, ofile)
    ofile.close()
    assert os.path.isfile(filename), "> ERROR: .gba model creation failed."

### Create a GBA model from CSV files ###
def create_gba_model( csv_model, gba_path = "", save_LP = False, save_optimums = False ):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Create and load the model from CSV files #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    model = GbaModel()
    model.read_csv_model(csv_model)
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
            model.save_f0()
        else:
            raise Exception("> ERROR: model is inconsistent with condition 1. f0 vector cannot be saved.")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 3) Compute and save optimums if requested   #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if save_optimums:
        print("> Computing optimums for model "+model.name+"...")
        if not save_LP:
            model.read_LP_from_csv()
        model.compute_optimums(max_time=10000, initial_dt=0.01)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 4) Clean model and dump binary backup       #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    model.reset_variables()
    backup_gba_model(model, gba_path)
    del model

### Read the GBA model from a binary file ###
def read_csv_model( csv_model ):
    assert os.path.exists(csv_model), "> ERROR: folder "+csv_model+" does not exist."
    model = GbaModel()
    model.read_from_csv(csv_model)
    return model

### Read the GBA model from a binary file ###
def read_gba_model( gba_model ):
    assert os.path.isfile(gba_model), "> ERROR: .gba model not found."
    assert gba_model.endswith(".gba"), "> ERROR: .gba model file extension is missing."
    ifile = open(gba_model, "rb")
    model = dill.load(ifile)
    ifile.close()
    return model

### Get the path of a GBA toy model included in the Python package as CSV files ###
def get_toy_model_path( model_name ):
    model_dir  = Path(pkgutil.resolve_name("gbapy.data").__file__).parent
    model_path = Path(model_dir , "toy_models/"+model_name)
    return str(model_path)

### Read a GBA toy model included in the Python package as CSV files ###
def read_toy_model( model_name ):
    model_dir  = Path(pkgutil.resolve_name("gbapy.data").__file__).parent
    model_path = Path(model_dir , "toy_models/"+model_name)
    model      = read_csv_model(str(model_path))
    return model


class GbaModel:
    
    ### Class constructor ###
    def __init__( self ):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) GBA model                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        ### Model folder, name and informations ###
        self.csv_model = ""             # Model folder
        self.name      = ""             # Model name
        self.infos     = pd.DataFrame() # Model informations

        ### Identifier lists ###
        self.metabolite_ids   = [] # List of all metabolite ids 
        self.x_ids            = [] # List of external metabolite ids
        self.c_ids            = [] # List of internal metabolite ids
        self.reaction_ids     = [] # List of reaction ids
        self.condition_ids    = [] # List of condition ids
        self.condition_params = [] # List of condition parameter ids

        ### Model structure ###
        self.Mx            = np.array([]) # Total mass fraction matrix
        self.M             = np.array([]) # Internal mass fraction matrix
        self.KM_f          = np.array([]) # Forward KM matrix
        self.KM_b          = np.array([]) # Backward KM matrix
        self.KI            = np.array([]) # KI matrix
        self.rKI           = np.array([]) # 1/KI matrix
        self.KA            = np.array([]) # KA matrix
        self.kcat_f        = np.array([]) # Forward kcat vector
        self.kcat_b        = np.array([]) # Backward kcat vector
        self.reversible    = []           # Indicates if the reaction is reversible
        self.kinetic_model = []           # Indicates the kinetic model of the reaction
        self.direction     = []           # Indicates the direction of the reaction
        self.conditions    = np.array([]) # List of conditions

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
        self.column_rank = 0 # Column rank of M

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Solutions                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.LP_solution       = np.array([]) # Linear programming solution
        self.optimum_solutions = {}           # Optimum f vectors for all conditions
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
        self.optimum_data = pd.DataFrame() # Optimum dataframe for all conditions
        self.GA_tracker   = pd.DataFrame() # Gradient ascent trajectory tracker
        self.MC_tracker   = pd.DataFrame() # Monte Carlo with genetic drift tracker
        self.MCMC_tracker = pd.DataFrame() # MCMC trajectory tracker
        
    #############################
    #   Model loading methods   #
    #############################
    
    ### Read information from CSV ###
    def read_Infos_from_csv( self ):
        variables      = []
        infos_columns  = ['Type', 'Content']
        self.infos     = pd.DataFrame(columns=infos_columns)
        Infos_filename = self.csv_model+"/Infos.csv"
        assert os.path.exists(Infos_filename), "> ERROR: file "+Infos_filename+" does not exist."
        f = open(Infos_filename, "r")
        l = f.readline()
        while l:
            l = l.strip().split(";")
            variables.append(l[0])
            data_row = [l[0], l[1]]
            self.infos.loc[len(self.infos)] = data_row
            if l[0] == "Name":
                self.name = l[1]
            l = f.readline()
        f.close()
        assert "Name" in variables, "> ERROR: Name not found in Infos.csv."

    ### Read the mass fraction matrix M from CSV ###
    def read_Mx_from_csv( self ):
        Mx_filename = self.csv_model+"/M.csv"
        assert os.path.exists(Mx_filename), "> File "+Mx_filename+" does not exist."
        df                  = pd.read_csv(Mx_filename, sep=";")
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
        del(df)

    ### Read the forward Michaelis constant matrix KM from CSV ###
    def read_KM_f_from_csv( self ):
        KM_f_filename = self.csv_model+"/KM_forward.csv"
        assert os.path.exists(KM_f_filename), "> ERROR: file "+KM_f_filename+" does not exist."
        df        = pd.read_csv(KM_f_filename, sep=";")
        df        = df.drop(["Unnamed: 0"], axis=1)
        df.index  = self.metabolite_ids
        self.KM_f = np.array(df)
        self.KM_f = self.KM_f.astype(float)
        del(df)

    ### Read the backward Michaelis constant matrix KM from CSV ###
    def read_KM_b_from_csv( self ):
        self.KM_b     = np.zeros(self.KM_f.shape)
        KM_b_filename = self.csv_model+"/KM_backward.csv"
        if os.path.exists(KM_b_filename):
            df        = pd.read_csv(KM_b_filename, sep=";")
            df        = df.drop(["Unnamed: 0"], axis=1)
            df.index  = self.metabolite_ids
            self.KM_b = np.array(df)
            self.KM_b = self.KM_b.astype(float)
            del(df)

    ### Read kcat forward and backward constant vectors from CSV ###
    def read_kcat_from_csv( self ):
        kcat_filename = self.csv_model+"/kcat.csv"
        assert os.path.exists(kcat_filename), "> ERROR: file "+kcat_filename+" does not exist."
        df          = pd.read_csv(kcat_filename, sep=";")
        df          = df.drop(["Unnamed: 0"], axis=1)
        kcat        = np.array(df)
        kcat        = kcat.astype(float)
        self.kcat_f = np.array(kcat[0,:])
        self.kcat_b = np.array(kcat[1,:])
        del(df)
        for j in range(len(self.kcat_b)):
            if self.kcat_b[j] > 0.0:
                self.reversible.append(True)
            else:
                self.reversible.append(False)

    ### Read the list of conditions from CSV ###
    def read_conditions_from_csv( self ):
        conditions_filename = self.csv_model+"/conditions.csv"
        assert os.path.exists(conditions_filename), "> ERROR: file "+conditions_filename+" does not exist."
        df                    = pd.read_csv(conditions_filename, sep=";")
        self.condition_params = list(df["Unnamed: 0"])
        self.condition_ids    = list(df.columns)[1:df.shape[1]]
        self.condition_ids    = [str(int(name)) for name in self.condition_ids]
        df                    = df.drop(["Unnamed: 0"], axis=1)
        df.index              = self.condition_params
        self.conditions       = np.array(df)
        del(df)

    ### Read the inhibition constants matrix KI from CSV ###
    def read_KI_from_csv( self ):
        self.KI     = np.zeros(self.Mx.shape)
        KI_filename = self.csv_model+"/KI.csv"
        if os.path.exists(KI_filename):
            df          = pd.read_csv(KI_filename, sep=";")
            metabolites = list(df["Unnamed: 0"])
            df          = df.drop(["Unnamed: 0"], axis=1)
            df.index    = metabolites
            self.KI     = np.array(df)
            self.KI     = self.KI.astype(float)
            del(df)

    ### Read the activation constants matrix KA from CSV ###
    def read_KA_from_csv( self ):
        self.KA     = np.zeros(self.Mx.shape)
        KA_filename = self.csv_model+"/KA.csv"
        if os.path.exists(KA_filename):
            df          = pd.read_csv(KA_filename, sep=";")
            metabolites = list(df["Unnamed: 0"])
            df          = df.drop(["Unnamed: 0"], axis=1)
            df.index    = metabolites
            self.KA     = np.array(df)
            self.KA     = self.KA.astype(float)
            del(df)

    ### Read the LP solution from CSV on request ###
    def read_LP_from_csv( self ):
        LP_filename = self.csv_model+"/f0.csv"
        assert os.path.exists(LP_filename), "> ERROR: file "+LP_filename+" does not exist."
        df               = pd.read_csv(LP_filename, sep=";")
        self.LP_solution = np.array(df["f0"])
        del(df)

    ### Initialize model mathematical variables ###
    def initialize_model_mathematical_variables( self ):
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
                self.kinetic_model.append("iMM")
                self.direction.append("forward")
            elif (self.kcat_b[j] == 0 and self.KI[:,j].sum() > 0 and self.KA[:,j].sum() == 0):
                self.kinetic_model.append("iMMi")
                self.direction.append("forward")
            elif (self.kcat_b[j] == 0 and self.KI[:,j].sum() == 0 and self.KA[:,j].sum() > 0):
                self.kinetic_model.append("iMMa")
                self.direction.append("forward")
            elif (self.kcat_b[j] == 0 and self.KI[:,j].sum() > 0 and self.KA[:,j].sum() > 0):
                self.kinetic_model.append("iMMia")
            elif (self.kcat_b[j] > 0):
                assert self.KI[:,j].sum() == 0
                assert self.KA[:,j].sum() == 0
                self.kinetic_model.append("rMM")
                self.direction.append("reversible")
    
    ### Read the GBA model from CSV files ###
    def read_from_csv( self, csv_model ):
        assert os.path.exists(csv_model), "> ERROR: folder "+csv_model+" does not exist."
        self.csv_model = csv_model
        self.read_Infos_from_csv()
        self.read_Mx_from_csv()
        self.read_KM_f_from_csv()
        self.read_KM_b_from_csv()
        self.read_kcat_from_csv()
        self.read_conditions_from_csv()
        self.read_KI_from_csv()
        self.read_KA_from_csv()
        self.initialize_model_mathematical_variables()

    ### Reset model variables (used before binary export) ###
    def reset_variables( self ):
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
    
    #######################
    #   Print functions   #
    #######################

    ### Print model summary ###
    def summary( self ):
        raise NotImplementedError("> ERROR: summary() method not implemented.")
    
    ###############
    #   Getters   #
    ###############
    
    ### Get a condition parameter value ###
    def get_condition( self, condition_id, condition_param ):
        assert condition_id in self.condition_ids, "> ERROR: unknown condition identifier "+condition_id+"."
        assert condition_param in self.condition_params, "> ERROR: unknown condition parameter "+condition_param+"."
        i = self.condition_params.index(condition_param)
        j = self.condition_ids.index(condition_id)
        return self.conditions[i,j]

    ###################
    #   Print model   #
    ###################

    ### Model str report function ###
    def __str__( self ):
        header  = " -------- Model report: " + self.name + " --------\n"
        report  = "\n"
        report += header
        report += "| • Nb metabolites          = " + str(self.ni) + "\n"
        report += "| • Nb external metabolites = " + str(self.nx) + "\n"
        report += "| • Nb internal metabolites = " + str(self.nc) + "\n"
        report += " " + "".join(["-"]*(len(header)-2))
        report += "\n"
        report += "| • Nb reactions          = " + str(self.nj) + "\n"
        report += "| • Nb exchange reactions = " + str(self.ns) + "\n"
        report += "| • Nb internal reactions = " + str(self.ne) + "\n"
        report += "| • Column rank           = " + str(self.column_rank) + "\n"
        report += " " + "".join(["-"]*(len(header)-2))
        report += "\n"
        return report

    ### Model print report function ###
    def __repr__( self ):
        return self.__str__()

    ##########################
    #   Analytical methods   #
    ##########################

    ### Set external conditions                       ###
    ### (Minimal values bounded to MIN_CONCENTRATION) ###
    def set_condition( self, condition ):
        assert condition in self.condition_ids, "> ERROR: unknown condition identifier "+condition_id+"."
        self.condition = condition
        self.rho       = self.get_condition(self.condition, "rho")
        for i in range(self.nx):
            x_name    = self.x_ids[i]
            x_value   = self.get_condition(self.condition, x_name)
            self.x[i] = x_value
            if self.adjust_concentrations and self.x[i] < MIN_CONCENTRATION:
                self.x[i] = MIN_CONCENTRATION

    ### Set f0 ###
    def set_f0( self, f0 ):
        assert len(f0) == self.nj, "> ERROR: incorrect f0 length."
        self.f0      = np.copy(f0)
        self.f_trunc = np.copy(self.f0[1:self.nj])
        self.f       = np.copy(self.f0)
    
    ### Compute f from truncated vector f_trunc ###
    def set_f( self ):
        term1  = (1-self.sM[1:].dot(self.f_trunc))/self.sM[0]
        self.f = np.copy(np.concatenate([np.array([term1]), self.f_trunc]))
    
    ### Compute internal concentrations ###
    def compute_c( self ):
        self.c = self.rho*self.M.dot(self.f)
        if self.adjust_concentrations:
            self.c[self.c < MIN_CONCENTRATION] = MIN_CONCENTRATION
        self.xc = np.concatenate([self.x, self.c])
        
    ### Irreversible Michaelis-Menten kinetics ###
    def iMM( self, j ):
        term1         = np.prod(1.0+self.KM_f[:,j]/self.xc)
        term2         = self.kcat_f[j]
        self.tau_j[j] = term1/term2

    ### Irreversible Michaelis-Menten kinetics + inhibition (only one inhibitor per reaction) ###
    def iMMi( self, j ):
        term1         = np.prod(1.0+self.xc*self.rKI[:,j])
        term2         = np.prod(1.0+self.KM_f[:,j]/self.xc)
        term3         = self.kcat_f[j]
        self.tau_j[j] = term1*term2/term3
    
    ### Irreversible Michaelis-Menten kinetics + activation (only one activator per reaction) ###
    def iMMa( self, j ):
        term1         = np.prod(1.0+self.KA[:,j]/self.xc)
        term2         = np.prod(1.0+self.KM_f[:,j]/self.xc)
        term3         = self.kcat_f[j]
        self.tau_j[j] = term1*term2/term3

    ### Irreversible Michaelis-Menten kinetics + inhibition + activation ###
    def iMMia( self, j ):
        term1         = np.prod(1.0+self.xc*self.rKI[:,j])
        term2         = np.prod(1.0+self.KA[:,j]/self.xc)
        term3         = np.prod(1.0+self.KM_f[:,j]/self.xc)
        term4         = self.kcat_f[j]
        self.tau_j[j] = term1*term2*term3/term4

    ### Reversible Michaelis-Menten kinetics ###
    def rMM( self, j ):
        term1         = self.kcat_f[j]
        term2         = np.prod(1+self.KM_f[:,j]/self.xc)
        term3         = self.kcat_b[j]
        term4         = np.prod(1+self.KM_b[:,j]/self.xc)
        self.tau_j[j] = 1.0/(term1/term2-term3/term4)
    
    ### Compute tau ###
    def compute_tau( self, j ):
        if self.kinetic_model[j] == "iMM":
            self.iMM(j)
        elif self.kinetic_model[j] == "iMMi":
            self.iMMi(j)
        elif self.kinetic_model[j] == "iMMa":
            self.iMMa(j)
        elif self.kinetic_model[j] == "iMMia":
            self.iMMia(j)
        elif self.kinetic_model[j] == "rMM":
            self.rMM(j)
    
    ### derivative of iMM with respect to metabolite concentrations ###
    def diMM( self, j ):
        constant1 = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            term1             = self.KM_f[y,j]/np.power(self.c[i], 2.0)
            term2             = np.prod(1+self.KM_f[indices,j]/self.xc[indices])
            self.ditau_j[j,i] = -term1*term2/constant1

    ### derivative of iMMi with respect to metabolite concentrations ###
    def diMMi( self, j ):
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
    
    ### derivative of iMMa with respect to metabolite concentrations ###
    def diMMa( self, j ):
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

    ### derivative of iMMia with respect to metabolite concentrations ###
    def diMMia( self, j ):
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

    ### Derivative of rMM with respect to metabolite concentrations ###
    def drMM( self, j ):
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
    
    ### Compute dtau ###
    def compute_dtau( self, j ):
        if self.kinetic_model[j] == "iMM":
            self.diMM(j)
        elif self.kinetic_model[j] == "iMMi":
            self.diMMi(j)
        elif self.kinetic_model[j] == "iMMa":
            self.diMMa(j)
        elif self.kinetic_model[j] == "iMMia":
            self.diMMia(j)
        elif self.kinetic_model[j] == "rMM":
            self.drMM(j)
    
    ### Compute the growth rate ###
    def compute_mu( self ):
        self.mu = self.M[self.a,self.r]*self.f[self.r]/(self.tau_j.dot(self.f))

    ### Compute fluxes ###
    def compute_v( self ):
        self.v = self.mu*self.rho*self.f

    ### Compute protein concentrations ###
    def compute_p( self ):
        self.p = self.tau_j*self.v

    ### Compute biomass fractions ###
    def compute_b( self ):
        self.b = self.M.dot(self.f)

    ### Compute cell density (should be == 1) ###
    def compute_density( self ):
        self.density = self.sM.dot(self.f)

    ### Compute local mu gradient with respect to f ###
    def compute_dmu_f( self ):
        term1      = np.power(self.mu, 2)/self.b[self.a]
        term2      = self.M[self.a,:]/self.mu
        term3      = self.f.T.dot(self.rho*self.ditau_j.dot(self.M))
        term4      = self.tau_j
        self.dmu_f = term1*(term2-term3-term4)

    ### Compute local growth control coefficients with respect to f ###
    def compute_GCC_f( self ):
        self.GCC_f = self.dmu_f-self.dmu_f[0]*(self.sM/self.sM[0])
    
    ### Calculate the model state ###
    def calculate_state( self ):
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

    ### Check model state's consistency ###
    def check_model_consistency( self ):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Test density constraint                 #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        test1 = (np.abs(self.density-1.0) < DENSITY_TOL)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Test negative concentrations constraint #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        test2 = (sum(1 for x in self.c if x < -NEGATIVE_C_TOL) == 0)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Test negative proteins constraint       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        test3 = (sum(1 for x in self.p if x < -NEGATIVE_P_TOL) == 0)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Print error message if inconsistent     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.consistent = True
        if not (test1 and test2 and test3):
            self.consistent = False

    ###################################
    # Generation of initial solutions #
    ###################################

    ### Initial value subproblem: linear optimization to find maximal ###
    # Ribosome flux fraction f^r, with a minimal production of each
    # metabolite. The constraints are mass conservation (M*f = b)
    # and surface flux balance (sM*f = 1).
    def solve_local_linear_problem( self ):
        gpmodel = gp.Model(env=env)
        x       = gpmodel.addMVar(self.nj, lb=MIN_FLUX_FRACTION, ub=MAX_FLUX_FRACTION)
        min_b   = 1/self.nc/10
        rhs     = np.repeat(min_b, self.nc)
        gpmodel.setObjective(x[-1], gp.GRB.MAXIMIZE)
        gpmodel.addConstr(self.M @ x >= rhs, name="c1")
        gpmodel.addConstr(self.sM @ x == 1, name="c2")
        gpmodel.optimize()
        self.LP_solution = np.copy(x.X)

    ### Generate random initial solutions ###
    def generate_random_initial_solutions( self, condition, nb_solutions, max_trials, min_mu ):
        assert condition in self.condition_ids, "> ERROR: unknown condition identifier."
        assert nb_solutions > 0, "> ERROR: number of solutions must be greater than 0."
        assert max_trials >= nb_solutions, "> ERROR: number of trials must be greater than the number of solutions."
        assert min_mu >= 0.0, "> ERROR: minimal growth rate must be positive."
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the optimums data frame #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        columns          = ['condition', 'mu', 'density']
        columns          = columns + self.reaction_ids
        self.random_data = pd.DataFrame(columns=columns)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Find the random solutions          #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.set_condition(condition)
        self.random_solutions.clear()
        solutions = 0
        trials    = 0
        while solutions < nb_solutions and trials < max_trials:
            trials       += 1
            negative_term = True
            while negative_term:
                self.f_trunc = np.random.rand(self.nj-1)
                self.f_trunc = self.f_trunc*(MAX_FLUX_FRACTION-MIN_FLUX_FRACTION)+MIN_FLUX_FRACTION
                self.set_f()
                if self.f[0] >= 0.0:
                    negative_term = False
            self.calculate_state()
            self.check_model_consistency()
            if self.consistent and np.isfinite(self.mu) and self.mu > min_mu:
                print("> ", solutions, " solutions was found after ", trials, " trials")
                solutions += 1
                data_dict  = {"condition": condition, "mu": self.mu, "density": self.density}
                for reaction_id, fluxfraction in zip(self.reaction_ids, self.f):
                    data_dict[reaction_id] = fluxfraction
                data_row                         = pd.Series(data=data_dict)
                self.random_data                 = pd.concat([self.random_data, data_row.to_frame().T], ignore_index=True)
                self.random_solutions[solutions] = np.copy(self.f)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Save the dataset                   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.random_data.to_csv("./csv_models/"+self.name+"/random_solutions.csv", sep=';', index=False)
        print("> ", solutions, " solutions was found after ", trials, " trials")

    ########################
    # Optimization Methods #
    ########################

    ### Draw a random normal vector with std 'sigma' and length 'n' ###
    def draw_gaussian_noise( self, sigma, n ):
        epsilon = np.random.normal(0.0, sigma, size=n)
        return epsilon
    
    ### Mutate an element of f with Gaussian std 'sigma' ###
    def mutate_f( self, index, sigma ):
        non_mutated_f        = np.copy(self.f_trunc)
        epsilon              = self.draw_gaussian_noise(sigma, 1)
        self.f_trunc[index] += epsilon 
        self.f_trunc[self.f_trunc < MIN_FLUX_FRACTION] = MIN_FLUX_FRACTION
        self.set_f()
        return non_mutated_f
    
    ### Calculate the selection coefficient for MCMC mutation fixation ###
    def calculate_selection_coefficient( self, mu, mutated_mu ):
        return 1.0-mu/mutated_mu
    
    ### Calculate fixation probability pi for MCMC ###
    def calculate_pi( self, selection_coefficient, N_e ):
        pi = 0.0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if selection_coefficient == 0.0:
                pi = 1.0/N_e
            else:
                pi = (1-np.exp(-2*selection_coefficient)) / (1-np.exp(-2*N_e*selection_coefficient))
        return pi

    ### Simulate fixation for MCMC ###
    def simulate_fixation( self, pi ):
        return np.random.rand() < pi
    
    ### Block reactions tending to zero ###
    # f values tending towards zero are bounded to min value.
    # Corresponding derivative values are set to zero depending
    # on the direction:
    # - f -> 0+ and gcc < 0: f = min_f, gcc = 0
    # - f -> 0- and gcc > 0: f = -min_f, gcc = 0
    def block_reactions( self ):
        for j in range(self.nj-1):
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 1) Reaction is irreversible and positive #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            if self.direction[j+1] == "forward" and self.f_trunc[j] <= MIN_FLUX_FRACTION:
                self.f_trunc[j] = MIN_FLUX_FRACTION
                if self.GCC_f[(j+1)] < 0.0:
                    self.GCC_f[(j+1)] = 0.0
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 2) Reaction is irreversible and negative #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            elif self.direction[j+1] == "backward" and self.f_trunc[j] >= -MIN_FLUX_FRACTION:
                self.f_trunc[j] = -MIN_FLUX_FRACTION
                if self.GCC_f[(j+1)] > 0.0:
                    self.GCC_f[(j+1)] = 0.0
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # 3) Reaction is reversible                #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # elif self.direction[j+1] == "reversible" and np.abs(self.f_trunc[j]) <= MIN_FLUX_FRACTION:
            #     self.GCC_f[(j+1)] = 0.0
            #     if self.f_trunc[j] >= 0.0:
            #         self.f_trunc[j] = MIN_FLUX_FRACTION
            #     elif self.f_trunc[j] < 0.0:
            #         self.f_trunc[j] = -MIN_FLUX_FRACTION
    ### Data tracking function for all algorithms
    def track_values(self, saved_values, data_dict, algorithm="MC", dt=None):

        if algorithm == "GA":
            tracker = self.GA_tracker
            if dt is not None:
                data_dict["dt"] = dt
        elif algorithm == "MCMC":
            tracker = self.MCMC_tracker
        else:
            tracker = self.MC_tracker

        # Fluxfractions    
        if 'f' in saved_values:
            for reaction_id, fluxfraction in zip([rid + ".f" for rid in self.reaction_ids], self.f):
                data_dict[reaction_id] = fluxfraction
        # Fluxes
        if 'v' in saved_values:
            for reaction_id, flux in zip([rid + ".v" for rid in self.reaction_ids], self.v):
                data_dict[reaction_id] = flux
        # Metabolites
        if 'b' in saved_values:
            for reaction_id, metabolite in zip([rid + ".b" for rid in self.reaction_ids], self.b):
                data_dict[reaction_id] = metabolite
        # c
        if 'c' in saved_values:
            for reaction_id, c in zip([rid + ".c" for rid in self.reaction_ids], self.c):
                data_dict[reaction_id] = c
        # p
        if 'p' in saved_values:
            for reaction_id, p in zip([rid + ".p" for rid in self.reaction_ids], self.p):
                data_dict[reaction_id] = p
        
        # Save the data row to the right tracker
        data_row = pd.Series(data=data_dict)
        tracker = pd.concat([tracker, data_row.to_frame().T], ignore_index=True)

        # Update the right tracker back to the class attribute
        if algorithm == "GA":
            self.GA_tracker = tracker
        elif algorithm == "MCMC":
            self.MCMC_tracker = tracker
        else:
            self.MC_tracker = tracker



    ### Run a gradient ascent ###
    def gradient_ascent( self, condition = "1", max_time = 5.0, initial_dt = 0.01, track = False, saved_values = ['f'], label = 1, verbose = False ):
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.adjust_concentrations = False
        self.set_condition(condition)
        self.calculate_state()
        self.check_model_consistency()
        assert self.consistent, "> ERROR: initial model is not consistent."
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize tracker       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if track:
            if self.GA_tracker.empty:
                columns = ['label', 'condition', 'dt','t', 'mu', 'fixed']
                self.GA_tracker = pd.DataFrame(columns=columns)

            data_dict = {"label": label, "condition": condition, "t": 0.0, "mu": self.mu, "fixed": 0}
            self.track_values(saved_values, data_dict, algorithm = "GA", dt = 1)
            
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
            # if nb_iterations%5000 == 0:
            #    print("> Iteration: ",nb_iterations, " Time: ",t, " mu: ",self.mu, " dt: ",dt)
            ### 4.1) Test trajectory convergence ###
            if mu_alteration_counter >= TRAJECTORY_CONVERGENCE_COUNT:
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
                if track and nb_iterations%EXPORT_DATA_COUNT == 0:
                    data_dict = {"label": label, "condition": condition, "t": t, "mu": self.mu, "fixed": nb_fixed}
                    self.track_values(saved_values, data_dict, algorithm = "GA", dt = 1)
                ### Check if mu changes significantly ###
                if np.abs(self.mu - previous_mu) < TRAJECTORY_CONVERGENCE_TOL:
                    mu_alteration_counter += 1
                else:
                    mu_alteration_counter = 0
                ### Check if dt is never changing, and possibly increase it ###
                if dt_counter == INCREASING_DT_COUNT:
                    dt         = dt*INCREASING_DT_FACTOR
                    dt_counter = 0
            ### 4.4) If the model is inconsistent: ###
            else:
                self.f_trunc = np.copy(previous_f)
                self.set_f()
                self.calculate_state()
                self.check_model_consistency()
                assert self.consistent, "> ERROR: previous model is not consistent"
                if (dt > 1e-100):
                    dt         = dt/DECREASING_DT_FACTOR
                    dt_counter = 0
                else:
                    raise AssertionError("> ERROR: adaptative timestep < 1e-100.")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if t >= max_time:
            if verbose:
                print("> Gradient ascent: maximum time reached (condition="+str(condition)+",\tmu="+str(round(self.mu, 5))+",\tnb iterations="+str(nb_iterations)+",\tnb fixed="+str(nb_fixed)+")")
            return False, run_time
        else:
            if verbose:
                print("> Gradient ascent: convergence reached (condition="+str(condition)+",\tmu="+str(round(self.mu, 5))+",\tnb iterations="+str(nb_iterations)+",\tnb fixed="+str(nb_fixed)+")")
            return True, run_time

    ### Compute all the optimums ###
    def compute_optimums( self, max_time = 5, initial_dt = 0.01, verbose = False ):
        start = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the optimums data frame #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        columns           = ['condition', 'mu', 'density', 'converged', 'run_time']
        columns           = columns + self.reaction_ids
        self.optimum_data = pd.DataFrame(columns=columns)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Calculate the optimums             #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.optimum_solutions.clear()
        for condition in self.condition_ids:
            self.set_f0(self.LP_solution)
            converged, run_time = self.gradient_ascent(condition=condition, max_time=max_time, initial_dt=initial_dt)
            data_dict           = {"condition": condition, "mu": self.mu, "density": self.density, "converged": int(converged), "run_time": run_time}
            for reaction_id, fluxfraction in zip(self.reaction_ids, self.f):
                data_dict[reaction_id] = fluxfraction
            data_row                          = pd.Series(data=data_dict)
            self.optimum_data                 = pd.concat([self.optimum_data, data_row.to_frame().T], ignore_index=True)
            self.optimum_solutions[condition] = np.copy(self.f)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Save the dataset                   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.optimum_data.to_csv(self.csv_model+"/optimums.csv", sep=';', index=False)
        end = time.time()
        if verbose:
            print("> All optimums were computed in "+str(end-start)+" seconds")
    
    ### Run a Monte Carlo simulation with genetic drift (Pál & Miklós, 1998) ###
    # f(t+1) = f(t) + sigma * dmu/df + epsilon.
    def MC_simulation( self, condition = "1", max_time = 100000, max_iter = 100000, sigma = 0.1, N_e = 2.5e7, track = False, saved_values = ['f'], label = 1 ):
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.set_condition(condition)
        self.calculate_state()
        self.check_model_consistency()
        assert self.consistent, "> Error: initial model is not consistent."
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize tracker       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if track:
            if self.MC_tracker.empty:
                columns = ['label', 'condition', 't', 'mu', 'fixed']
                self.MC_tracker = pd.DataFrame(columns=columns)

            data_dict = {"label": label, "condition": condition, "t": 0.0, "mu": self.mu, "fixed": 0}
            self.track_values(saved_values, data_dict, algorithm = 'MC')

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
            if nb_iterations >= max_iter:
                print("> Condition "+condition+": MAXITER reached")
                break
            ### 4.1) Calculate the next step ###
            epsilon      = self.draw_gaussian_noise(np.sqrt(sigma/(2.0*N_e)), self.nj-1)
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
                if track and nb_iterations % EXPORT_DATA_COUNT == 0:
                    data_dict = {"label": label, "condition": condition, "t": t, "mu": self.mu, "fixed": nb_fixed}
                    
                    # Call the track_values function to save data during iterations
                    self.track_values(saved_values, data_dict)
            ### 4.3) If the model is inconsistent: ###
            else:
                self.f_trunc = np.copy(previous_f)
                self.set_f()
                self.calculate_state()
                self.check_model_consistency()
                assert self.consistent, "> ERROR: previous model is not consistent."
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if nb_fixed == 0 and nb_iterations < max_iter:
            print("> MC simulation completed with no fixed mutation (condition="+condition+",\tmu="+str(round(self.mu, 5))+",\tnb iterations="+str(nb_iterations)+",\tnb fixed="+str(nb_fixed)+").")
            return False, run_time
        elif nb_fixed > 0 and nb_iterations < max_iter:
            print("> MC simulation completed (condition="+condition+",\tmu="+str(round(self.mu, 5))+",\tnb iterations="+str(nb_iterations)+",\tnb fixed="+str(nb_fixed)+").")
            return True, run_time
        elif nb_iterations >= max_iter:
            print("> MC simulation: maximum iterations reached (condition="+condition+",\tmu="+str(round(self.mu, 5))+",\tnb iterations="+str(nb_iterations)+",\tnb fixed="+str(nb_fixed)+").")
            return False, run_time

    ### Run a Markov chain Monte Carlo simulation ###
    # Standard MCMC formulation (Gillespie, 1983).
    def MCMC_simulation(self, condition = "1", max_iter = 100000, sigma = 0.01, N_e = 2.5e7, track = False,saved_values=['f'], label = 1 ):
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.set_condition(condition)
        self.calculate_state()
        self.check_model_consistency()
        assert self.consistent, "> ERROR: initial model is not consistent"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize trackers       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if track:
            if self.MC_tracker.empty:
                columns = ['label', 'condition', 't', 'mu', 'fixed']
                self.MC_tracker = pd.DataFrame(columns=columns)

            data_dict = {"label": label, "condition": condition, "t": 0.0, "mu": self.mu, "fixed": 0}
            self.track_values(saved_values, data_dict, algorithm = "MCMC")    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Initialize the algorithm  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        current_mu = self.mu
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Start the MCMC            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        nb_iterations = 0
        nb_fixed      = 0
        while nb_iterations < max_iter:
            nb_iterations += 1
            ### 4.1) Draw reaction to mutate at random ###
            reaction_index = np.random.randint(len(self.f_trunc))
            current_mu     = self.mu
            non_mutated_f  = self.mutate_f(reaction_index, sigma)
            self.calculate_state()
            self.check_model_consistency()
            ### 4.2) Check model consistency and simulate fixation ###
            if self.consistent:
                mutated_mu = self.mu
                s          = self.calculate_selection_coefficient(current_mu, mutated_mu)
                pi         = self.calculate_pi(s, N_e)
                ### 4.3) Undo Mutation if no fixation occurs ###
                if self.simulate_fixation(pi) == False:
                    self.f_trunc = np.copy(non_mutated_f)
                    self.set_f()
                ### 4.4) Save Mutation for trajectory if fixation occurs ###
                else:
                    nb_fixed  += 1
                    if track and nb_iterations % EXPORT_DATA_COUNT == 0:
                        data_dict = {"label": label, "condition": condition, "t": nb_iterations, "mu": self.mu, "fixed": nb_fixed}
                        self.track_values(saved_values, data_dict, algorithm="MCMC")
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
        if nb_fixed == 0 and nb_iterations < max_iter:
            print("> MCMC: simulation completed with no fixed mutation (condition="+condition+",\tmu="+str(round(self.mu, 5))+",\tnb iterations="+str(nb_iterations)+",\tnb fixed="+str(nb_fixed)+").")
            return False, run_time, nb_fixed
        elif nb_fixed > 0 and nb_iterations < max_iter:
            print("> MCMC: simulation completed (condition="+condition+",\tmu="+str(round(self.mu, 5))+",\tnb iterations="+str(nb_iterations)+",\tnb fixed="+str(nb_fixed)+").")
            return True, run_time, nb_fixed
        elif nb_iterations >= max_iter:
            print("> MCMC: maximum iterations reached (condition="+condition+",\tmu="+str(round(self.mu, 5))+",\tnb iterations="+str(nb_iterations)+",\tnb fixed="+str(nb_fixed)+").")
            return False, run_time, nb_fixed

    ### Save f0 vector ###
    def save_f0( self ):
        ### Add the reaction name as header ###
        f = open(self.csv_model+"/f0.csv", "w")
        f.write("reaction;f0\n")
        for i in range(self.nj):
            f.write(self.reaction_ids[i]+";"+str(self.f0[i])+"\n")
        f.close()

    ### Save the gradient ascent trajectory to csv ###
    def save_gradient_ascent_trajectory( self, label = "" ):
        header = "./output/"+self.name
        if label != "":
            header += "_"+str(label)
        if not self.GA_tracker.empty:
            self.GA_tracker.to_csv(header+"_gradient_ascent_trajectory.csv", sep=';', index=False)

    ### Save MC trajectory to csv ###
    def save_MC_trajectory( self, label = "" ):
        header = "./output/"+self.name
        if label != "":
            header += "_"+str(label)
        if not self.MC_tracker.empty:
            self.MC_tracker.to_csv(header+"_MC_trajectory.csv", sep=';', index=False)
    
    ### Save MCMC trajectory to csv ###
    def save_MCMC_trajectory( self, label = "" ):
        header = "./output/"+self.name
        if label != "":
            header += "_"+str(label)
        if not self.MCMC_tracker.empty:
            self.MCMC_tracker.to_csv(header+"_MCMC_trajectory.csv", sep=';', index=False)
    
    ### Save all trajectories to csv ###
    def save_all_trajectories( self, label = "" ):
        self.save_gradient_ascent_trajectory(label)
        self.save_MC_tracker_trajectory(label) 
        self.save_MCMC_trajectory(label)

    ### Clear gradient ascent trajectory ###
    def clear_gradient_ascent_trajectory( self ):
        self.GA_tracker = pd.DataFrame()
    
    ### Clear MC trajectory ###
    def clear_MC_trajectory( self ):
        self.MC_tracker = pd.DataFrame()
    
    ### Clear MCMC trajectory ###
    def clear_MCMC_trajectory( self ):
        self.MCMC_tracker = pd.DataFrame()
    
    ### Clear all trajectories ###
    def clear_all_trajectories( self ):
        self.clear_gradient_ascent_trajectory()
        self.clear_MC_trajectory()
        self.clear_MCMC_trajectory()

