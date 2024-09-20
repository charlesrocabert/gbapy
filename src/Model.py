#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# Copyright © 2023-2024 Charles Rocabert, Furkan Mert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# GBA_model.py
# ------------
# Implementation of the GBA analytical formalism for a given model.
# (LOCAL SCRIPT)
#***********************************************************************

import os
import sys
import dill
import time
import numpy as np
import pandas as pd
import gurobipy as gp
import matplotlib.pyplot as plt

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

sys.path.append('./src/')

### Define constant and tolerance thresholds ###
MIN_CONCENTRATION          = 1e-10 # Minimum concentration value
MIN_FLUX_FRACTION          = 1e-10 # Minimum flux fraction value
MAX_FLUX_FRACTION          = 10.0  # Maximum flux fraction value
DENSITY_TOL                = 1e-10 # Density tolerance threshold (|1-rho| < ε)
NEGATIVE_C_TOL             = 1e-10 # Negative C tolerance threshold (C > -ε)
NEGATIVE_P_TOL             = 1e-10 # Negative P tolerance threshold (P > -ε)
TRAJECTORY_STABLE_MU_COUNT = 1000  # Number of iterations with equal mu values to consider the trajectory stable
TRAJECTORY_CONVERGENCE_TOL = 1e-10 # Mu threshold below which growth rates are considered equal
DECREASING_DT_FACTOR       = 5.0   # Factor by which the time step is decreased when the trajectory is unstable
INCREASING_DT_FACTOR       = 2.0   # Factor by which the time step is increased when the trajectory is stable
INCREASING_DT_COUNT        = 100   # Number of iterations with equal mu values to increase the time step
MCMC_CONVERGENCE_TOL       = 1e-5  # MCMC trajectory convergence tolerance
POPLEVEL_CONVERGENCE_TOL   = 1e-5  # Population-level trajectory convergence tolerance
EFM_TOL                    = 1e-5  # Tolerance threshold below which EFM values are considered to be zero

### Dump the model in a binary file ###
def dump_model( model, model_name ):
    filename = "./binary_models/"+model_name+".gba"
    ofile = open(filename, "wb")
    dill.dump(model, ofile)
    ofile.close()
    assert os.path.isfile(filename), "ERROR: dump_model: model dump failed."

### Load a model and dump the binary backup ###
def load_and_backup_model( model_name, save_LP = False, save_optimums = False ):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Create and load the model from CSV files #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    model = Model()
    model.load_model("./csv_models/", model_name)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Compute and save f0 if requested         #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if save_LP:
        print("> Computing LP solution for model "+model_name+"...")
        model.solve_local_linear_problem()
        model.set_f0(model.LP_solution)
        model.set_condition("1")
        model.calculate()
        model.check_model_consistency()
        if model.consistent:
            model.write_f0()
        else:
            print("> ERROR: Model is inconsistent with condition 1. f0 vector cannot be saved.")
            sys.exit(1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 3) Compute and save optimums if requested   #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if save_optimums:
        print("> Computing optimums for model "+model_name+"...")
        if not save_LP:
            model.load_LP()
        model.compute_optimums(max_time=10000, initial_dt=0.01)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 4) Clean model and dump binary backup       #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    model.reset_variables()
    dump_model(model, model_name)
    del model

### Load the model from a binary file ###
def load_model( model_name ):
    filename = "./binary_models/"+model_name+".gba"
    assert os.path.isfile(filename), "ERROR: model not found."
    ifile = open(filename, "rb")
    model = dill.load(ifile)
    ifile.close()
    return model


class Model:

    # Mathematical formalism may differ from original ODS files and R scripts,
    # in particular with ni = nx + nc, and not ni = nc.
    # ------------------------------------------------------------------------
    # x:       External metabolite concentrations
    # c:       Internal metabolite concentrations
    # v:       Fluxes vector
    # f:       Flux fractions vector
    # p:       Protein concentrations vector
    # b:       Biomass fractions vector
    # Mx:      Total mass fraction matrix
    # M:       Internal mass fraction matrix
    # KM:      Km matrix
    # KI:      Inhibition constants matrix
    # KA:      Activation constants matrix
    # kcat_f:  Forward kcat vector
    # kcat_b:  Backward kcat vector
    # tau_j:   Turnover times vector
    # ditau_j: Turnover times derivative vector
    # rho:     Total density
    # mu:      Growth rate
    # dmu_f:   Growth rate derivative with respect to f
    # GCC_f:   Growh control coefficient vector with respect to f

    ### Class constructor ###
    def __init__( self ):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) GBA model                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        ### Model path and name ###
        self.model_path = "" # Model path
        self.model_name = "" # Model name

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
        # 2) GBA model constant variables  #
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
        self.column_rank = 0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Solutions                     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.LP_solution       = np.array([])   # Linear programming solution
        self.optimum_solutions = {}             # Optimum f vectors for all conditions
        self.random_solutions  = {}             # Random f vectors for all conditions
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) GBA model dynamical variables #
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
        # 5) Evolutionary variables        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.current_condition = ""           # Current environmental condition
        self.current_rho       = 0.0          # Current total density
        self.f0                = np.array([]) # Initial LP solution
        self.dmu_f             = np.array([]) # Local mu derivatives with respect to f
        self.GCC_f             = np.array([]) # Local growth control coefficients with respect to f
        self.f_trunc           = np.array([]) # Truncated f vector (first element is removed)
        self.f                 = np.array([]) # Flux fractions vector

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 6) Trackers                      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.optimum_data = pd.DataFrame() # Optimum dataframe for all conditions
        self.trajectory   = pd.DataFrame() # Trajectory dataframe

    #############################
    #   Model loading methods   #
    #############################
        
    ### Load the mass fraction matrix M ###
    def load_Mx( self ):
        Mx_filename = self.model_path+"/"+self.model_name+"/M.csv"
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

    ### Load the forward Michaelis constant matrix KM ###
    def load_KM_f( self ):
        KM_f_filename = self.model_path+"/"+self.model_name+"/KM_forward.csv"
        assert os.path.exists(KM_f_filename), "> File "+KM_f_filename+" does not exist."
        df        = pd.read_csv(KM_f_filename, sep=";")
        df        = df.drop(["Unnamed: 0"], axis=1)
        df.index  = self.metabolite_ids
        self.KM_f = np.array(df)
        self.KM_f = self.KM_f.astype(float)
        del(df)

    ### Load the backward Michaelis constant matrix KM ###
    def load_KM_b( self ):
        self.KM_b     = np.zeros(self.KM_f.shape)
        KM_b_filename = self.model_path+"/"+self.model_name+"/KM_backward.csv"
        if os.path.exists(KM_b_filename):
            df        = pd.read_csv(KM_b_filename, sep=";")
            df        = df.drop(["Unnamed: 0"], axis=1)
            df.index  = self.metabolite_ids
            self.KM_b = np.array(df)
            self.KM_b = self.KM_b.astype(float)
            del(df)

    ### Load kcat forward and backward constant vectors ###
    def load_kcat( self ):
        kcat_filename = self.model_path+"/"+self.model_name+"/kcat.csv"
        assert os.path.exists(kcat_filename), "> File "+kcat_filename+" does not exist."
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

    ### Load the list of conditions ###
    def load_conditions( self ):
        conditions_filename = self.model_path+"/"+self.model_name+"/conditions.csv"
        assert os.path.exists(conditions_filename), "> File "+conditions_filename+" does not exist."
        df                    = pd.read_csv(conditions_filename, sep=";")
        self.condition_params = list(df["Unnamed: 0"])
        self.condition_ids    = list(df.columns)[1:df.shape[1]]
        self.condition_ids    = [str(int(name)) for name in self.condition_ids]
        df                    = df.drop(["Unnamed: 0"], axis=1)
        df.index              = self.condition_params
        self.conditions       = np.array(df)
        del(df)

    ### Load the inhibition constants matrix KI ###
    def load_KI( self ):
        self.KI     = np.zeros(self.Mx.shape)
        KI_filename = self.model_path+"/"+self.model_name+"/KI.csv"
        if os.path.exists(KI_filename):
            df          = pd.read_csv(KI_filename, sep=";")
            metabolites = list(df["Unnamed: 0"])
            df          = df.drop(["Unnamed: 0"], axis=1)
            df.index    = metabolites
            self.KI     = np.array(df)
            self.KI     = self.KI.astype(float)
            del(df)

    ### Load the activation constants matrix KA ###
    def load_KA( self ):
        self.KA     = np.zeros(self.Mx.shape)
        KA_filename = self.model_path+"/"+self.model_name+"/KA.csv"
        if os.path.exists(KA_filename):
            df          = pd.read_csv(KA_filename, sep=";")
            metabolites = list(df["Unnamed: 0"])
            df          = df.drop(["Unnamed: 0"], axis=1)
            df.index    = metabolites
            self.KA     = np.array(df)
            self.KA     = self.KA.astype(float)
            del(df)

    ### Load the LP solution on request ###
    def load_LP( self ):
        LP_filename = self.model_path+"/"+self.model_name+"/f0.csv"
        assert os.path.exists(LP_filename), "> File "+LP_filename+" does not exist."
        df          = pd.read_csv(LP_filename, sep=";")
        self.LP_solution = np.array(df["f0"])
        del(df)

    ### Initialize model mathematical variables ###
    def initialize_model_mathematical_variables( self ):
        ### Inverse of KI ###
        with np.errstate(divide='ignore'):
            self.rKI = 1/self.KI
            self.rKI[np.isinf(self.rKI)] = 0.0
        ### Vector lengths ###
        self.nx = len(self.x_ids)
        self.nc = len(self.c_ids)
        self.ni = self.nx+self.nc
        self.nj = len(self.reaction_ids)
        self.x  = np.zeros(self.nx)
        self.c  = np.zeros(self.nc)
        self.xc = np.zeros(self.ni)
        ### Create M matrix ###
        self.M = np.zeros((self.nc, self.nj))
        for i in range(self.nc):
            met_id = self.c_ids[i]
            for j in range(self.nj):
                self.M[i,j] = self.Mx[self.metabolite_ids.index(met_id),j]
        ### Indices for reactions: s (transport), e (enzymatic), and ribosome r ###
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
        ### Indices: m (metabolite), a (all proteins) ###
        self.m = list(range(self.nc-1))
        self.a = self.nc-1
        ### Matrix column rank ###
        self.column_rank = np.linalg.matrix_rank(self.M)
        ### GBA model dynamical variables ###
        self.tau_j   = np.zeros(self.nj)
        self.ditau_j = np.zeros((self.nj, self.nc))
        self.x       = np.zeros(self.nx)
        self.c       = np.zeros(self.nc)
        self.xc      = np.zeros(self.ni)
        self.v       = np.zeros(self.nj)
        self.p       = np.zeros(self.nj)
        self.b       = np.zeros(self.nc)
        ### Evolutionary variables ###
        self.f0      = np.zeros(self.nj)
        self.dmu_f   = np.zeros(self.nj)
        self.GCC_f   = np.zeros(self.nj)
        self.f_trunc = np.zeros(self.nj-1)
        self.f       = np.zeros(self.nj)
    
    ### Load the GBA model (M, K and kcat matrices) ###
    def load_model( self, model_path, model_name ):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Save model path and name                               #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.model_path = model_path
        self.model_name = model_name
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Load model's kinetics                                  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.load_Mx()
        self.load_KM_f()
        self.load_KM_b()
        self.load_kcat()
        self.load_conditions()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Load inhibition and activation constants if they exist #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.load_KI()
        self.load_KA()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Initialize model mathematical variables                #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.initialize_model_mathematical_variables()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Define the kinetic model of each reaction              #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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

    ### Reset model variables (used before binary export) ###
    def reset_variables( self ):
        ### GBA model dynamical variables ###
        self.tau_j   = np.zeros(self.nj)
        self.ditau_j = np.zeros((self.nj, self.nc))
        self.x       = np.zeros(self.nx)
        self.c       = np.zeros(self.nc)
        self.xc      = np.zeros(self.ni)
        self.v       = np.zeros(self.nj)
        self.p       = np.zeros(self.nj)
        self.b       = np.zeros(self.nc)
        ### Evolutionary variables ###
        self.f0      = np.zeros(self.nj)
        self.dmu_f   = np.zeros(self.nj)
        self.GCC_f   = np.zeros(self.nj)
        self.f_trunc = np.zeros(self.nj-1)
        self.f       = np.zeros(self.nj)
    
    ###############
    #   Getters   #
    ###############
    
    ### Get a condition parameter value ###
    def get_condition( self, condition_id, condition_param ):
        assert condition_id in self.condition_ids
        assert condition_param in self.condition_params
        i = self.condition_params.index(condition_param)
        j = self.condition_ids.index(condition_id)
        return self.conditions[i,j]

    ###################
    #   Print model   #
    ###################

    ### Model str report function ###
    def __str__( self ):
        header  = " -------- Model report: " + self.model_name + " --------\n"
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
        report += "| • Model kinetics = " + self.model_kinetics + "\n"
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
        assert condition in self.condition_ids
        self.current_condition = condition
        self.current_rho       = self.get_condition(self.current_condition, "rho")
        for i in range(self.nx):
            x_name    = self.x_ids[i]
            x_value   = self.get_condition(self.current_condition, x_name)
            self.x[i] = x_value
            if self.adjust_concentrations and self.x[i] < MIN_CONCENTRATION:
                self.x[i] = MIN_CONCENTRATION

    ### Set f0 ###
    def set_f0( self, f0 ):
        assert len(f0) == self.nj
        self.f0      = np.copy(f0)
        self.f_trunc = np.copy(self.f0[1:self.nj])
        self.f       = np.copy(self.f0)
    
    ### Compute f from truncated vector f_trunc ###
    def set_f( self ):
        term1  = (1-self.sM[1:].dot(self.f_trunc))/self.sM[0]
        self.f = np.copy(np.concatenate([np.array([term1]), self.f_trunc]))
    
    ### Compute internal concentrations ###
    def compute_c( self ):
        self.c = self.current_rho*self.M.dot(self.f)
        if self.adjust_concentrations:
            self.c[self.c < MIN_CONCENTRATION] = MIN_CONCENTRATION
        self.xc = np.concatenate([self.x, self.c])
        
    ### Irreversible Michaelis-Menten kinetics ###
    def iMM( self, j ):
        self.tau_j[j] = np.prod(1+self.KM_f[:,j]/self.xc)/self.kcat_f[j]

    ### Irreversible Michaelis-Menten kinetics + inhibition (only one inhibitor per reaction) ###
    def iMMi( self, j ):
        self.tau_j[j] = np.prod(1+self.xc*self.rKI[:,j])*np.prod(1+self.KM_f[:,j]/self.xc)/self.kcat_f[j]
    
    ### Irreversible Michaelis-Menten kinetics + activation (only one activator per reaction) ###
    def iMMa( self, j ):
        term1 = np.prod(1+self.KA[:,j]/self.xc)
        term2 = np.prod(1+self.KM_f[:,j]/self.xc)
        term3 = self.kcat_f[j]
        self.tau_j[j] = term1*term2/term3

    ### Irreversible Michaelis-Menten kinetics + inhibition + activation ###
    def iMMia( self, j ):
        self.tau_j[j] = np.prod(1+self.xc*self.rKI[:,j])*np.prod(1+self.KA[:,j]/self.xc)*np.prod(1+self.KM_f[:,j]/self.xc)/self.kcat_f[j]

    ### Reversible Michaelis-Menten kinetics ###
    def rMM( self, j ):
        forward_term  = self.kcat_f[j]/np.prod(1+self.KM_f[:,j]/self.xc)
        backward_term = self.kcat_b[j]/np.prod(1+self.KM_b[:,j]/self.xc)
        self.tau_j[j] = 1/(forward_term-backward_term)
    
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
        term3 = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            indices           = np.arange(self.ni) != y
            term1             = self.KM_f[y,j]/np.power(self.c[i], 2)
            term2             = np.prod(1+self.KM_f[indices,j]/self.xc[indices])
            self.ditau_j[j,i] = -term1*term2/term3
            #if self.ditau_j[j,i] == -0.0:
            #    self.ditau_j[j,i] = 0.0

    ### derivative of iMMi with respect to metabolite concentrations ###
    def diMMi( self, j ):
        term5 = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            term1             = self.rKI[y,j]*np.prod(1+self.KM_f[:,j]/self.xc)
            term2             = np.prod(1+self.xc*self.rKI[:,j])
            term3             = self.KM_f[y,j]/self.c[i]**2
            term4             = np.prod(1+self.KM_f[np.arange(self.ni) != y,j]/self.xc[np.arange(self.ni) != y])
            self.ditau_j[j,i] = term1-term2*term3*term4/term5
    
    ### derivative of iMMa with respect to metabolite concentrations ###
    def diMMa( self, j ):
        term6 = self.kcat_f[j]
        for i in range(self.nc):
            y     = i+self.nx
            term1 = self.KA[y,j]/self.c[i]**2
            term2 = np.prod(1+self.KM_f[:,j]/self.xc)
            term3 = self.KM_f[y,j]/self.c[i]**2
            term4 = np.prod(1+self.KA[:,j]/self.xc)
            term5 = np.prod(1+self.KM_f[np.arange(self.ni) != y,j]/self.xc[np.arange(self.ni) != y])
            self.ditau_j[j,i] = -(term1*term2+term3*term4*term5)/term6

    ### derivative of iMMia with respect to metabolite concentrations ###
    def diMMia( self, j ):
        term9 = self.kcat_f[j]
        for i in range(self.nc):
            y                 = i+self.nx
            term1             = self.rKI[y,j]*np.prod(1+self.KA[:,j]/self.c)*np.prod(1+self.KM_f[:,j]/self.c)
            term2             = np.prod(1+self.c*self.rKI[:,j])
            term3             = -self.KA[y,j]/self.c[i]**2
            term4             = np.prod(1+self.KM_f[:,j]/self.c)
            term5             = np.prod(1+self.c*self.rKI[:,j])
            term6             = np.prod(1+self.KA[:,j]/self.c)
            term7             = -self.KM_f[y,j]/self.c[i]**2
            term8             = np.prod(1+self.KM_f[np.arange(self.ni) != y,j]/self.c[np.arange(self.ni) != y])
            self.ditau_j[j,i] = term1+(term2*term3*term4)+(term5*term6*term7*term8)/term9

    ### Derivative of rMM with respect to metabolite concentrations ###
    def drMM( self, j ):
        for i in range(self.nc):
            y       = i+self.nx
            indices = np.arange(self.ni) != y
            term1   = (self.kcat_f[j]/np.prod(1 + self.KM_f[indices,j]/self.xc[indices]))
            term2   = self.KM_f[y,j]/((self.c[i] + self.KM_f[y,j])**2)
            term3   = (self.kcat_b[j]/np.prod(1 + self.KM_b[indices,j]/self.xc[indices]))
            term4   = self.KM_b[y,j]/((self.c[i] + self.KM_b[y,j])**2)
            self.ditau_j[j,i] = term1*term2-term3*term4
        self.ditau_j[j,:] *= -self.tau_j[j]
    
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
        self.v = self.mu*self.current_rho*self.f

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
        term3      = self.f.T.dot(self.current_rho*self.ditau_j.dot(self.M))
        term4      = self.tau_j
        self.dmu_f = term1*(term2-term3-term4)

    ### Compute local growth control coefficients with respect to f ###
    def compute_GCC_f( self ):
        self.GCC_f = self.dmu_f-self.dmu_f[0]*(self.sM/self.sM[0])
    
    ### Calculate all model variables from the f vector ###
    def calculate( self ):
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
    ### ribosome flux fraction f^r, with a minimal production of each ###
    ### metabolite. The constraints are mass conservation (M*f = b)   ###
    ### and surface flux balance (sM*f = 1).                          ###
    ### WARNING: this method assumes full irreversibility             ###
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
        assert condition in self.condition_ids, "> Condition not found"
        assert nb_solutions > 0, "> Number of solutions must be greater than 0"
        assert max_trials >= nb_solutions, "> Number of trials must be greater than the number of solutions"
        assert min_mu >= 0.0, "> Minimal growth rate must be positive"
        self.set_condition(condition)
        self.random_solutions.clear()
        solutions = 0
        trials    = 0
        while solutions < nb_solutions and trials < max_trials:
            trials       += 1
            negative_term = True
            while negative_term:
                f_trunc = np.random.rand(self.nj-1)
                f_trunc = f_trunc*MAX_FLUX_FRACTION
                self.set_f(f_trunc)
                if self.f[0] >= 0.0:
                    negative_term = False
            self.calculate()
            self.check_model_consistency()
            if self.consistent and np.isfinite(self.mu) and self.mu > min_mu:
                print("> ", solutions, " solutions was found after ", trials, " trials")
                solutions += 1
                self.random_solutions[solutions] = np.copy(self.f)
        print("> ", solutions, " solutions was found after ", trials, " trials")

    ########################
    # Optimization Methods #
    ########################

    ### Draw a random normal vector with std 'sigma' and length 'n' ###
    def draw_noise( self, sigma, n ):
        epsilon = np.random.normal(0.0, sigma, size=n)
        return epsilon
    
    ### Calculates the mutated flux fraction for each reaction ###
    def mutate_f( self, index, sigma ):
        non_mutated_f     = np.copy(self.f_trunc)
        mutated_f         = np.copy(self.f_trunc)
        epsilon           = self.draw_noise(sigma, 1)
        mutated_f[index] += epsilon 
        mutated_f[mutated_f < MIN_FLUX_FRACTION] = MIN_FLUX_FRACTION
        self.set_f(mutated_f)
        return non_mutated_f
    
    ### Calculate the selection coefficient for MCMC mutation fixation ###
    def calculate_selection_coefficient( self, mu, mutated_mu ):
        return 1.0 - mu / mutated_mu
    
    ### Simulate fixation for MCMC ###
    def simulate_fixation( self, pi ):
        return np.random.rand() < pi
        
    ### Calcutlate fixation probability pi for MCMC ###
    def calculate_pi( self, selection_coefficient, N_e ):
        if (selection_coefficient == 0):
            return 1/N_e
        else:
            return (1-np.exp(-2*selection_coefficient)) / (1-np.exp(-2*N_e*selection_coefficient))

    ### Bloc reactions tending to zero ###
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
            # 2) Reaction is reversible                #
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            elif self.direction[j+1] == "reversible" and np.abs(self.f_trunc[j]) <= MIN_FLUX_FRACTION:
                self.GCC_f[(j+1)] = 0.0
                if self.f_trunc[j] >= 0.0:
                    self.f_trunc[j] = MIN_FLUX_FRACTION
                elif self.f_trunc[j] < 0.0:
                    self.f_trunc[j] = -MIN_FLUX_FRACTION
    
    ### Compute the gradient ascent ###
    def gradient_ascent( self, condition = "1", max_time = 5.0, initial_dt = 0.01, index = 1, track = False, add = False ):
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.adjust_concentrations = False
        self.set_condition(condition)
        self.calculate()
        self.check_model_consistency()
        assert self.consistent, "> Initial model is not consistent"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize tracker        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if track:
            if not add or self.trajectory.empty:
                overview_columns = ['index', 'condition', 't','dt','mu','dmu']
                overview_columns = overview_columns + self.reaction_ids
                self.trajectory  = pd.DataFrame(columns=overview_columns)
            overview_dict = {"index": index, "condition": condition, "t": 0.0, "dt": initial_dt, "mu": self.mu, "dmu": 0.0}
            for reaction_id, value in zip(self.reaction_ids, self.f):
                overview_dict[reaction_id] = value
            overview_row                   = pd.Series(data=overview_dict)
            self.trajectory                = pd.concat([self.trajectory, overview_row.to_frame().T], ignore_index=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Initialize the algorithm  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        t                     = 0.0
        dt                    = initial_dt
        mu_alteration_counter = 0
        previous_f            = np.copy(self.f_trunc)
        previous_mu           = self.mu
        self.converged        = False
        nb_iterations         = 0
        dt_counter            = 0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Start the gradient ascent #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        while (t < max_time):
            nb_iterations += 1
            #if nb_iterations%1000 == 0:
            #    print("> Iteration: ",nb_iterations, " Time: ",t, " mu: ",self.mu, " dt: ",dt)
            ### 4.1) Test trajectory convergence ###
            if(mu_alteration_counter >= TRAJECTORY_STABLE_MU_COUNT):
                self.converged = True
                break
            ### 4.2) Calculate the next step ###
            previous_mu = self.mu
            self.block_reactions()
            self.f_trunc = self.f_trunc+self.GCC_f[1:]*dt
            #self.f_trunc[self.f_trunc < 0.0] = 0.0
            self.set_f()
            self.calculate()
            self.check_model_consistency()
            
            #print(pd.DataFrame(data={"f": self.f, "GCC_f": self.GCC_f}, index=self.reaction_ids))
            #sys.exit()
            ### 4.3) If the model is consistent: ###
            if self.consistent and self.mu >= previous_mu:
                previous_f  = np.copy(self.f_trunc)
                t           = t + dt
                dt_counter += 1
                if track:
                    overview_dict = {"index": index, "condition": condition, "t": t, "dt": dt, "mu": self.mu, "dmu": np.abs(self.mu-previous_mu)}
                    for reaction_id, value in zip(self.reaction_ids, self.f):
                        overview_dict[reaction_id] = value
                    overview_row = pd.Series(data=overview_dict)
                    self.trajectory = pd.concat([self.trajectory, overview_row.to_frame().T], ignore_index=True)
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
                self.calculate()
                self.check_model_consistency()
                assert self.consistent, "> Previous model is not consistent"
                if (dt > 1e-100):
                    dt         = dt/DECREASING_DT_FACTOR
                    dt_counter = 0
                else:
                    raise AssertionError("> Trajectory was stopped, because dt got too small")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if t >= max_time:
            print("> Condition "+condition+": MAXTIME reached")
            return False, run_time
        else:
            print("> Condition "+condition+": convergence reached (mu="+str(self.mu)+", nb iterations="+str(nb_iterations)+")")
            return True, run_time
        
    ### Compute all the optimums ###
    def compute_optimums( self, max_time = 5, initial_dt = 0.01 ):
        start = time.time()
        overview_columns  = ['condition', 'mu','density','converged', 'run_time']
        overview_columns  = overview_columns[:3] + self.reaction_ids + overview_columns[3:]
        self.optimum_data = pd.DataFrame(columns=overview_columns)
        self.optimum_solutions.clear()
        for condition in self.condition_ids:
            self.set_f0(self.LP_solution)
            converged, run_time = self.gradient_ascent(condition=condition, max_time=max_time, initial_dt=initial_dt, track=False, add=False)
            overview_dict = {"condition": condition, "mu": self.mu, "density": self.density, "converged": converged, "run_time": run_time}
            for reaction_id, fluxfraction in zip(self.reaction_ids, self.f):
                overview_dict[reaction_id] = fluxfraction
            overview_row                      = pd.Series(data=overview_dict)
            self.optimum_data                 = pd.concat([self.optimum_data, overview_row.to_frame().T], ignore_index=True)
            self.optimum_solutions[condition] = np.copy(self.f)
        self.optimum_data.to_csv("./csv_models/"+self.model_name+"/optimum.csv", sep=';', index=False)
        end = time.time()
        print("> All optimums were computed in ", end-start, " seconds")
    
    ### Compute the gradient ascent with noise ###
    def gradient_ascent_with_noise( self, condition = "1", max_time = 5, initial_dt = 0.01, sigma = 0.1, index = 1, track = False, add = False ):
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.set_condition(condition)
        self.calculate()
        self.check_model_consistency()
        assert self.consistent, "> Initial model is not consistent"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize tracker        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if track:
            if not add or self.trajectory.empty:
                overview_columns = ['index', 'condition', 't','dt','mu','dmu']
                overview_columns = overview_columns + self.reaction_ids
                self.trajectory  = pd.DataFrame(columns=overview_columns)
            overview_dict = {"index": index, "condition": condition, "t": 0.0, "dt": initial_dt, "mu": self.mu, "dmu": 0.0}
            for reaction_id, value in zip(self.reaction_ids, self.f):
                overview_dict[reaction_id] = value
            overview_row                   = pd.Series(data=overview_dict)
            self.trajectory                = pd.concat([self.trajectory, overview_row.to_frame().T], ignore_index=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Initialize the algorithm  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        t                     = 0.0
        dt                    = initial_dt
        mu_alteration_counter = 0
        previous_f            = np.copy(self.f_trunc)
        next_f                = np.copy(self.f_trunc)
        previous_mu           = self.mu
        self.converged        = False
        nb_iterations         = 0
        dt_counter            = 0
        epsilon               = self.draw_noise(sigma, self.nj-1)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Start the gradient ascent #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        while (t < max_time):
            nb_iterations += 1
            ### 4.1) Test trajectory convergence ###
            if(mu_alteration_counter >= TRAJECTORY_STABLE_MU_COUNT):
                self.converged = True
                break
            ### 4.2) Calculate the next step ###
            previous_mu          = self.mu
            next_f               = next_f+self.GCC_f[1:]*dt+epsilon*dt
            next_f[next_f < 0.0] = 0.0
            self.set_f(next_f)
            self.calculate()
            self.check_model_consistency()
            ### 4.3) If the model is consistent: ###
            if self.consistent and self.mu >= previous_mu:
                previous_f  = np.copy(next_f)
                epsilon     = self.draw_noise(sigma, self.nj-1)
                t           = t + dt
                dt_counter += 1
                if track:
                    overview_dict = {"index": index, "condition": condition, "t": t, "dt": dt, "mu": self.mu, "dmu": np.abs(self.mu-previous_mu)}
                    for reaction_id, value in zip(self.reaction_ids, self.f):
                        overview_dict[reaction_id] = value
                    overview_row = pd.Series(data=overview_dict)
                    self.trajectory = pd.concat([self.trajectory, overview_row.to_frame().T], ignore_index=True)
                ### Check if mu changes significantly ###
                if np.abs(self.mu - previous_mu) <= TRAJECTORY_CONVERGENCE_TOL:
                    mu_alteration_counter += 1
                else:
                    mu_alteration_counter = 0
                ### Check if dt is never changing, and possibly increase it ###
                if dt_counter == INCREASING_DT_COUNT:
                    dt         = dt*INCREASING_DT_FACTOR
                    dt_counter = 0
            ### 4.4) If the model is inconsistent: ###
            else:
                next_f = np.copy(previous_f)
                self.set_f(previous_f)
                self.calculate()
                self.check_model_consistency()
                assert self.consistent, "> Previous model is not consistent"
                if (dt > 1e-100):
                    dt         = dt/DECREASING_DT_FACTOR
                    dt_counter = 0
                else:
                    raise AssertionError("> Trajectory was stopped, because dt got too small")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if t >= max_time:
            print("> Max time was reached, Model is consistent for condition: ",condition)
            return False, run_time
        else:
            print("> Maximum was found, Model is consistent for condition: ",condition)
            return True, run_time

    ### Compute Markov chain Monte Carlo ###    
    def MCMC(self, condition = "1", max_time = 100000, sigma = 0.01, N_e = 2.5e7, index = 1, track = False, add = False ):
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.set_condition(condition)
        self.calculate()
        self.check_model_consistency()
        assert self.consistent, "> Initial model is not consistent"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize trackers       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if track:
            if not add or self.trajectory.empty:
                overview_columns = ['index', 'condition', 't','mu']
                overview_columns = overview_columns + self.reaction_ids
                self.trajectory  = pd.DataFrame(columns=overview_columns)
            overview_dict = {"index": index, "condition": condition, "t": 0.0, "mu": self.mu}
            for reaction_id, value in zip(self.reaction_ids, self.f):
                overview_dict[reaction_id] = value
            overview_row                   = pd.Series(data=overview_dict)
            self.trajectory                = pd.concat([self.trajectory, overview_row.to_frame().T], ignore_index=True)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Initialize the algorithm  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        current_mu = self.mu
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Start the MCMC            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        fixed = 0
        t     = 0
        while t < max_time:
            t += 1
            ### 4.1) Draw reaction to mutate at random ###
            reaction_index = np.random.randint(len(self.f_trunc))
            current_mu     = self.mu
            non_mutated_f  = self.mutate_f(reaction_index, sigma)
            self.calculate()
            self.check_model_consistency()
            ### 4.2) Check model consistency and simulate fixation ###
            if self.consistent:
                mutated_mu = self.mu
                s          = self.calculate_selection_coefficient(current_mu, mutated_mu)
                pi         = self.calculate_pi(s, N_e)
                ### 4.3) Undo Mutation if no fixation occurs ###
                if self.simulate_fixation(pi) == False:
                    self.set_f(non_mutated_f)
                ### 4.4) Save Mutation for trajectory if fixation occurs ###
                else:
                    fixed         += 1
                    overview_dict  = {"index": index, "condition": condition, "t": t, "mu": self.mu}
                    for reaction_id, value in zip(self.reaction_ids, self.f):
                        overview_dict[reaction_id] = value
                    overview_row                   = pd.Series(data=overview_dict)
                    self.trajectory                = pd.concat([self.trajectory, overview_row.to_frame().T], ignore_index=True)
            ### 4.5) Undo Mutation if model is inconsistent ###
            else:
                self.set_f(non_mutated_f)
            self.calculate()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        end_time = time.time()
        run_time = end_time-start_time
        if fixed == 0:
            print("> MCMC simulation was completed. No mutation was fixed")
            return False, run_time
        else:
            print("> MCMC simulation was completed")
            return True, run_time

    ### Save trajectory to csv ###
    def save_trajectory( self, label = "" ):
        filename = "./output/"+self.model_name+"_"
        if label == "":
            filename += "trajectory.csv"
        else:
            filename += label+"_trajectory.csv"
        self.trajectory.to_csv(filename, sep=';', index=False)

    ### Clear trajectory ###
    def clear_trajectory( self ):
        self.trajectory = pd.DataFrame()
    
    ######################
    #   Export methods   #
    ######################

    ### Write the model state in a file ###
    def write_model_state( self, file, header, identifier, additional_variables ):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Write the header      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~#
        if header:
            header = "identifier;mu;density;consistent"
            for f_id in self.reaction_ids:
                header += ";"+f_id
            for c_id in self.c_ids:
                header += ";"+c_id
            for f_id in self.reaction_ids:
                header += ";protein_"+f_id
            for f_id in self.reaction_ids:
                header += ";GCC_"+f_id
            for var in additional_variables.keys():
                header += ";"+var
            file.write(header+"\n")
            file.flush()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Write the model state #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~#
        line = str(identifier)+";"+str(self.mu)+";"+str(self.density)+";"+str(int(self.consistent))
        for val in self.f:
            line += ";"+str(val)
        for val in self.c:
            line += ";"+str(val)
        for val in self.p:
            line += ";"+str(val)
        for val in self.GCC_f:
            line += ";"+str(val)
        for var in additional_variables.keys():
            line += ";"+str(additional_variables[var])
        file.write(line+"\n")
        file.flush()
    
    ### Write the list of model variables ###
    def write_model_variables( self ):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Write metabolite names #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open("./output/"+self.model_name+"_external_metabolite_ids.csv", "w")
        for id in self.x_ids:
            f.write(id+"\n")
        f.close()
        f = open("./output/"+self.model_name+"_internal_metabolite_ids.csv", "w")
        for id in self.c_ids:
            f.write(id+"\n")
        f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Write reaction names   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open("./output/"+self.model_name+"_reaction_ids.csv", "w")
        for id in self.reaction_ids:
            f.write(id+"\n")
        f.close()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Write protein names    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        f = open("./output/"+self.model_name+"_protein_ids.csv", "w")
        for id in self.reaction_ids:
            f.write("protein_"+id+"\n")
        f.close()

    ### Write f0 vector in a file ###
    def write_f0( self ):
        ### Add the reaction name as header ###
        f = open("./csv_models/"+self.model_name+"/f0.csv", "w")
        f.write("reaction;f0\n")
        for i in range(self.nj):
            f.write(self.reaction_ids[i]+";"+str(self.f0[i])+"\n")
        f.close()

