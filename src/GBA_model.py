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
import numpy as np
import pandas as pd
import gurobipy as gp

env = gp.Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

sys.path.append('./src/')

from GBA_tol import *

### Dump the model in a binary file ###
def dump_model( gba_model, model_name ):
    filename = "./binary_models/"+model_name+".gba"
    ofile = open(filename, "wb")
    dill.dump(gba_model, ofile)
    ofile.close()
    assert os.path.isfile(filename), "ERROR: dump_model: model dump failed."

### Load a model and dump the binary backup ###
def load_and_backup_model( model_name, save_f0, save_optimums ):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 1) Create and load the model from CSV files #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    model = GBA_model()
    model.load_model("./csv_models/", model_name)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # 2) Compute and save f0 if requested         #
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    model.solve_local_linear_problem()
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
    algo.compute_optimum_for_all_conditions(max_time=200, initial_dt=0.01)
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


class GBA_model:

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
        self.model_path     = "" # Model path
        self.model_name     = "" # Model name
        self.model_kinetics = "" # Kinetics model

        ### Identifier lists ###
        self.metabolite_ids   = [] # List of all metabolite ids 
        self.x_ids            = [] # List of external metabolite ids
        self.c_ids            = [] # List of internal metabolite ids
        self.reaction_ids     = [] # List of reaction ids
        self.condition_ids    = [] # List of condition ids
        self.condition_params = [] # List of condition parameter ids

        ### Model structure ###
        self.Mx         = np.array([]) # Total mass fraction matrix
        self.M          = np.array([]) # Internal mass fraction matrix
        self.KM_f       = np.array([]) # Forward KM matrix
        self.KM_b       = np.array([]) # Backward KM matrix
        self.KI         = np.array([]) # KI matrix
        self.KA         = np.array([]) # KA matrix
        self.kcat_f     = np.array([]) # Forward kcat vector
        self.kcat_b     = np.array([]) # Backward kcat vector
        self.reversible = []           # Indicates if the reaction is reversible
        self.conditions = np.array([]) # List of conditions

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
        # 3) GBA model dynamical variables #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.tau_j      = np.array([]) # Tau values (turnover times)
        self.ditau_j    = np.array([]) # Tau derivative values
        self.x          = np.array([]) # External metabolite concentrations
        self.c          = np.array([]) # Internal metabolite concentrations
        self.xc         = np.array([]) # Metabolite concentrations
        self.v          = np.array([]) # Fluxes vector
        self.p          = np.array([]) # Protein concentrations vector
        self.b          = np.array([]) # Biomass fractions vector
        self.density    = 0.0          # Cell's relative density
        self.mu         = 0.0          # Growth rate
        self.consistent = False        # Is the model consistent?

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Evolutionary variables        #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.current_condition = ""           # Current environmental condition
        self.current_rho       = 0.0          # Current total density
        self.f0                = np.array([]) # Initial state
        self.dmu_f             = np.array([]) # Local mu derivatives with respect to f
        self.GCC_f             = np.array([]) # Local growth control coefficients with respect to f
        self.f_trunc           = np.array([]) # Truncated f vector (first element is removed)
        self.f                 = np.array([]) # Flux fractions vector

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

    ### Initialize model mathematical variables ###
    def initialize_model_mathematical_variables( self ):
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
        self.sM                  = np.sum(self.M, axis=0)
        self.sM[self.sM < 1e-10] = 0
        is_e                     = [self.sM[0:(self.nj-1)] == 0]
        self.e                   = []
        self.s                   = []
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
        ### Matrix column rank ###
        self.column_rank = np.linalg.matrix_rank(self.M)
    
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
        # 5) Define the kinetics model to be used                   #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.model_kinetics = ""
        if (np.sum(self.kcat_b) == 0 and np.sum(self.KI) == 0 and np.sum(self.KA) == 0):
            self.model_kinetics = "iMM"
        elif (np.sum(self.kcat_b) == 0 and np.sum(self.KI) > 0  and np.sum(self.KA) == 0):
            self.model_kinetics = "iMMi"
        elif (np.sum(self.kcat_b) == 0 and np.sum(self.KI) == 0  and np.sum(self.KA) > 0):
            self.model_kinetics = "iMMa"
        elif (np.sum(self.kcat_b) == 0 and np.sum(self.KI) > 0  and np.sum(self.KA) > 0):
            self.model_kinetics = "iMMia"
        elif (np.sum(self.kcat_b) > 0):
            assert np.sum(self.KI) == 0
            assert np.sum(self.KA) == 0
            self.model_kinetics = "rMM"
        assert self.model_kinetics != "", "> ERROR: load_model: kinetics model not found."

    ###############
    #   Getters   #
    ###############
    
    # ### Get a condition parameter value ###
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
            if self.x[i] < MIN_CONCENTRATION:
                self.x[i] = MIN_CONCENTRATION

    ### Set f0 ###
    def set_f0( self, f0 ):
        assert len(f0) == self.nj
        self.f0      = np.copy(f0)
        self.f_trunc = np.copy(self.f0[1:self.nj])
        self.f       = np.copy(self.f0)
    
    ### Compute f from truncated vector f_trunc ###
    def set_f( self, f_trunc ):
        self.f_trunc = np.copy(f_trunc)
        term1        = (1-self.sM[1:self.nj].dot(self.f_trunc))/self.sM[0]
        self.f       = np.copy(np.concatenate([np.array([term1]), self.f_trunc]))
    
    ### Initial value subproblem: linear optimization to find maximal ###
    ### ribosome flux fraction f^r, with a minimal production of each ###
    ### metabolite. The constraints are mass conservation (M*f = b)   ###
    ### and surface flux balance (sM*f = 1).                          ###
    ### WARNING: this method assumes full irreversibility             ###
    def solve_local_linear_problem( self ):
        gpmodel = gp.Model(env=env)
        x       = gpmodel.addMVar(self.nj, lb=0.0, ub=FLUX_BOUNDARY)
        min_b   = 1/self.nc/10
        rhs     = np.repeat(min_b, self.nc)
        gpmodel.setObjective(x[-1], gp.GRB.MAXIMIZE)
        gpmodel.addConstr(self.M @ x >= rhs, name="c1")
        gpmodel.addConstr(self.sM @ x == 1, name="c2")
        gpmodel.optimize()
        self.set_f0(x.X)

    ### Compute internal concentrations ###
    def compute_c( self ):
        self.c = self.current_rho*self.M.dot(self.f)
        self.c[self.c < MIN_CONCENTRATION] = MIN_CONCENTRATION
        self.xc = np.concatenate([self.x, self.c])
        
    ### Irreversible Michaelis-Menten kinetics ###
    def iMM( self ):
        for j in range(self.nj):
            self.tau_j[j] = np.prod(1+self.KM_f[:,j]/self.xc)/self.kcat_f[j]

    ### Irreversible Michaelis-Menten kinetics + inhibition (only one inhibitor per reaction) ###
    def iMMi( self ):
        rKI                = 1/self.KI
        rKI[np.isinf(rKI)] = 0.0
        for j in range(self.nj):
            self.tau_j[j] = np.prod(1+self.xc*rKI[:,j])*np.prod(1+self.KM_f[:,j]/self.xc)/self.kcat_f[j]
    
    ### Irreversible Michaelis-Menten kinetics + activation (only one activator per reaction) ###
    def iMMa( self ):
        for j in range(self.nj):
            term1 = np.prod(1+self.KA[:,j]/self.xc)
            term2 = np.prod(1+self.KM_f[:,j]/self.xc)
            term3 = self.kcat_f[j]
            self.tau_j[j] = term1*term2/term3

    ### Irreversible Michaelis-Menten kinetics + inhibition + activation ###
    def iMMia( self ):
        rKI                = 1/self.KI
        rKI[np.isinf(rKI)] = 0
        for j in range(self.nj):
            self.tau_j[j] = np.prod(1+self.xc*rKI[:,j])*np.prod(1+self.KA[:,j]/self.xc)*np.prod(1+self.KM_f[:,j]/self.xc)/self.kcat_f[j]

    ### Reversible Michaelis-Menten kinetics ###
    def rMM( self ):
        for j in range(self.nj):
            forward_term  = self.kcat_f[j]/np.prod(1+self.KM_f[:,j]/self.xc)
            backward_term = self.kcat_b[j]/np.prod(1+self.KM_b[:,j]/self.xc)
            self.tau_j[j] = 1/(forward_term-backward_term)
    
    ### Compute tau ###
    def compute_tau( self ):
        if self.model_kinetics == "iMM":
            self.iMM()
        elif self.model_kinetics == "iMMi":
            self.iMMi()
        elif self.model_kinetics == "iMMa":
            self.iMMa()
        elif self.model_kinetics == "iMMia":
            self.iMMia()
        elif self.model_kinetics == "rMM":
            self.rMM()
    
    ### derivative of iMM with respect to metabolite concentrations ###
    def diMM( self ):
        for j in range(self.nj):
            for i in range(self.nc):
                y       = i+self.nx
                indices = np.arange(self.ni) != y
                term1   = (self.KM_f[y,j]/self.c[i]**2)
                term2   = np.prod(1+self.KM_f[indices,j]/self.xc[indices])
                term3   = self.kcat_f[j]
                self.ditau_j[j,i] = -term1*term2/term3

    ### derivative of iMMi with respect to metabolite concentrations ###
    def diMMi( self ):
        rKI                = 1/self.KI
        rKI[np.isinf(rKI)] = 0
        for j in range(self.nj):
            for i in range(self.nc):
                y                 = i+self.nx
                term1             = rKI[y,j]*np.prod(1+self.KM_f[:,j]/self.xc)
                term2             = np.prod(1+self.xc*rKI[:,j])
                term3             = self.KM_f[y,j]/self.c[i]**2
                term4             = np.prod(1+self.KM_f[np.arange(self.ni) != y,j]/self.xc[np.arange(self.ni) != y])
                term5             = self.kcat_f[j]
                self.ditau_j[j,i] = term1-term2*term3*term4/term5
    
    ### derivative of iMMa with respect to metabolite concentrations ###
    def diMMa( self ):
        for j in range(self.nj):
            for i in range(self.nc):
                y     = i+self.nx
                term1 = self.KA[y,j]/self.c[i]**2
                term2 = np.prod(1+self.KM_f[:,j]/self.xc)
                term3 = self.KM_f[y,j]/self.c[i]**2
                term4 = np.prod(1+self.KA[:,j]/self.xc)
                term5 = np.prod(1+self.KM_f[np.arange(self.ni) != y,j]/self.xc[np.arange(self.ni) != y])
                term6 = self.kcat_f[j]
                self.ditau_j[j,i] = -(term1*term2+term3*term4*term5)/term6

    ### derivative of iMMia with respect to metabolite concentrations ###
    def diMMia( self ):
        rKI                = 1/self.KI
        rKI[np.isinf(rKI)] = 0
        for j in range(self.nj):
            for i in range(self.nc):
                y                 = i+self.nx
                term1             = rKI[y,j]*np.prod(1+self.KA[:,j]/self.c)*np.prod(1+self.KM_f[:,j]/self.c)
                term2             = np.prod(1+self.c*rKI[:,j])
                term3             = -self.KA[y,j]/self.c[i]**2
                term4             = np.prod(1+self.KM_f[:,j]/self.c)
                term5             = np.prod(1+self.c*rKI[:,j])
                term6             = np.prod(1+self.KA[:,j]/self.c)
                term7             = -self.KM_f[y,j]/self.c[i]**2
                term8             = np.prod(1+self.KM_f[np.arange(self.ni) != y,j]/self.c[np.arange(self.ni) != y])
                term9             = self.kcat_f[j]
                self.ditau_j[j,i] = term1+(term2*term3*term4)+(term5*term6*term7*term8)/term9

    ### Derivative of rMM with respect to metabolite concentrations ###
    def drMM( self ):
        for j in range(self.nj):
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
    def compute_dtau( self ):
        if self.model_kinetics == "iMM":
            self.diMM()
        elif self.model_kinetics == "iMMi":
            self.diMMi()
        elif self.model_kinetics == "iMMa":
            self.diMMa()
        elif self.model_kinetics == "iMMia":
            self.diMMia()
        elif self.model_kinetics == "rMM":
            self.drMM()
    
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
        term1      = self.mu**2/self.b[self.a]
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
        self.compute_tau()
        self.compute_dtau()
        self.compute_mu()
        self.compute_v()
        self.compute_p()
        self.compute_b()
        self.compute_density()
        self.compute_dmu_f()
        self.compute_GCC_f()

    ### Check model state's consistency ###
    def check_model_consistency( self ):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Test density constraint                 #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        test1 = (np.abs(self.density-1.0) < DENSITY_CONSTRAINT_TOL)
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
            
    ######################
    # Trajectory Methods #
    ######################
    
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

