#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# Copyright © 2024 Furkan Mert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# GBA_algorithms.py
# -----------------
# Implementation of GBA evolutionary algorithms.
# (LOCAL SCRIPT)
#***********************************************************************

import os
import sys
import dill
from matplotlib.pylab import f
import matplotlib.pyplot as plt
import time

# Add the local src directory to the path
sys.path.append('./src/')

from GBA_tol import *
from GBA_model import *


class GBA_algorithms:

    ### Class constructor ###
    def __init__( self, model_name ):
        assert os.path.exists("./binary_models/"+model_name+".gba"), "> Model not found"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Main model parameters      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.model_name = model_name
        self.gba_model  = load_model(self.model_name)
        self.condition  = ""
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Gradient ascent parameters #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.converged  = False
        self.run_time   = 0.0
        self.optimums   = None

    ### Plot trajectory ###
    def plot_trajectory( self, t_vec, dt_vec, mu_vec, mu_diff_vec ):
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 2, 1)
        plt.plot(t_vec, mu_vec, label='mu(t)')
        plt.xlabel('Time')
        plt.ylabel('mu')
        plt.title('Plot of mu against Time')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(dt_vec, label='dt')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('dt')
        plt.title('Dt')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(mu_diff_vec, label='mu_diff')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Mu diff')
        plt.title('Mu diff')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    ### Plot mu to condition ###
    def plot_mu_to_condition(self ):
        plt.plot(self.optimums['condition'], self.optimums['mu'], label='MaxGrowthrate at condition')
        plt.xlabel('conditions')
        plt.ylabel('Max-Grotwthrate')
        plt.title('Max-Growthrates over different conditions')
        plt.legend()
        plt.grid(False)

    ### Plot f to condition ###
    def plot_f_to_condition( self ):
        f_to_condition = self.optimums.iloc[:, 3:3+self.gba_model.nj].to_numpy()
        conditions = self.optimums['condition'].to_numpy()
        for i in range(self.gba_model.nj):
            plt.plot(conditions, f_to_condition[:, i], label = self.gba_model.reaction_ids[i])
        plt.xlabel('Conditions')
        plt.ylabel('Flux fraction')
        plt.title('Flux fractions over different conditions')
        plt.legend()
        plt.grid(False)

    ### Compute the gradient ascent without noise ###
    def compute_gradient_ascent( self, condition = "1", max_time = 5.0, initial_dt = 0.01 ):
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.gba_model.solve_local_linear_problem()
        self.gba_model.set_condition(condition)
        self.gba_model.calculate()
        self.gba_model.check_model_consistency()
        assert self.gba_model.consistent, "> Initial model is not consistent"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize trackers       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        t_vec       = []
        dt_vec      = []
        mu_vec      = []
        mu_diff_vec = []
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Initialize the algorithm  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        t                     = 0.0
        dt                    = initial_dt
        mu_alteration_counter = 0
        previous_f            = np.copy(self.gba_model.f_trunc)
        next_f                = np.copy(self.gba_model.f_trunc)
        previous_mu           = self.gba_model.mu
        self.converged        = False
        nb_iterations         = 0
        dt_counter            = 0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Start the gradient ascent #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        while (t < max_time):
            nb_iterations += 1
            #if (nb_iterations % 100 == 0):
            #    print("> Iteration: ", nb_iterations, " mu: ", mu_vec[-1], "mu diff: ", (mu_diff_vec[-1]), "dt: ", dt_vec[-1])
            ### 4.1) Test trajectory convergence ###
            if(mu_alteration_counter >= TRAJECTORY_STABLE_MU_COUNT):
                self.converged = True
                break
            ### 4.2) Calculate the next step ###
            previous_mu = self.gba_model.mu
            next_f      = next_f+self.gba_model.GCC_f[1:]*dt
            next_f[next_f < 0.0] = 0.0
            self.gba_model.set_f(next_f)
            self.gba_model.calculate()
            self.gba_model.check_model_consistency()
            ### 4.3) If the model is consistent: ###
            if self.gba_model.consistent and self.gba_model.mu >= previous_mu:
                previous_f  = np.copy(next_f)
                t           = t + dt
                dt_counter += 1
                t_vec.append(t)
                dt_vec.append(dt)
                mu_vec.append(self.gba_model.mu)
                mu_diff_vec.append(np.abs(self.gba_model.mu-previous_mu))
                ### Check if mu changes significantly ###
                if np.abs(self.gba_model.mu - previous_mu) <= TRAJECTORY_CONVERGENCE_TOL:
                    mu_alteration_counter += 1
                else:
                    mu_alteration_counter = 0
                ### Check if dt is never changing, and possibly increase it ###
                if dt_counter == 1000:
                    dt         = dt*INCREASING_DT_FACTOR
                    dt_counter = 0
            ### 4.4) If the model is inconsistent: ###
            else:
                next_f = np.copy(previous_f)
                self.gba_model.set_f(previous_f)
                self.gba_model.calculate()
                self.gba_model.check_model_consistency()
                assert self.gba_model.consistent, "> Previous model is not consistent"
                if (dt > 1e-100):
                    dt         = dt/DECREASING_DT_FACTOR
                    dt_counter = 0
                else:
                    raise AssertionError("trajectory was stopped, because dt got too small")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #self.plot_trajectory(t_vec, dt_vec, mu_vec, mu_diff_vec)
        if (t>=max_time):
            print("> Max time was reached, Model is consistent for condition: ",condition)
        else:
            print("> Maximum was found, Model is consistent for condition: ",condition)
        end_time      = time.time()
        self.run_time = end_time-start_time

    ### Compute the optimum for all conditions ###
    def compute_optimum_for_all_conditions( self, max_time = 5, initial_dt = 0.01 ):
        overview_columns = ['condition', 'mu','density','converged', 'run_time']
        overview_columns = overview_columns[:3] + self.gba_model.reaction_ids + overview_columns[3:]
        self.optimums    = pd.DataFrame(columns=overview_columns)
        for condition in self.gba_model.condition_ids:
            self.compute_gradient_ascent(condition=condition, max_time=max_time, initial_dt=initial_dt)
            overview_dict = {
                "condition": condition,
                "mu": self.gba_model.mu,
                "density": self.gba_model.density,
                "converged": self.converged,
                "run_time": self.run_time
            }
            for reaction_id, fluxfraction in zip(self.gba_model.reaction_ids, self.gba_model.f):
                overview_dict[reaction_id] = fluxfraction
            overview_row  = pd.Series(data=overview_dict)
            self.optimums = pd.concat([self.optimums, overview_row.to_frame().T], ignore_index=True)
        self.optimums.to_csv("./output/"+self.model_name+"_optimum.csv", sep=';', index=False)
    
    ### Draw a random normal vector with std 'sigma' and length 'n' ###
    def draw_noise( self, sigma ):
        epsilon = np.random.normal(0.0, sigma, size=self.gba_model.nj-1)
        return epsilon

    ### Compute the gradient ascent with noise ###
    def compute_gradient_ascent_with_noise( self, condition = "1", max_time = 5, initial_dt = 0.01, sigma = 0.1, nameOfCSV = None ):
        start_time = time.time()
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.gba_model.solve_local_linear_problem()
        self.gba_model.set_condition(condition)
        self.gba_model.calculate()
        self.gba_model.check_model_consistency()
        assert self.gba_model.consistent, "> Initial model is not consistent"
        print(self.gba_model.mu)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize trackers       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        t_vec       = []
        dt_vec      = []
        mu_vec      = []
        mu_diff_vec = []
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Initialize the algorithm  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        t                     = 0.0
        dt                    = initial_dt
        mu_alteration_counter = 0
        previous_f            = np.copy(self.gba_model.f_trunc)
        next_f                = np.copy(self.gba_model.f_trunc)
        previous_mu           = self.gba_model.mu
        self.converged        = False
        nb_iterations         = 0
        dt_counter            = 0
        epsilon               = self.draw_noise(sigma)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Start the gradient ascent #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        while (t < max_time):
            nb_iterations += 1
            if (nb_iterations % 100 == 0):
                print("> Iteration: ", nb_iterations, " mu: ", mu_vec[-1], "mu diff: ", (mu_diff_vec[-1]), "dt: ", dt_vec[-1])
            ### 4.1) Test trajectory convergence ###
            if(mu_alteration_counter >= TRAJECTORY_STABLE_MU_COUNT):
                self.converged = True
                break
            ### 4.2) Calculate the next step ###
            previous_mu = self.gba_model.mu
            next_f      = next_f+self.gba_model.GCC_f[1:]*dt+epsilon*dt
            next_f[next_f < 0.0] = 0.0
            self.gba_model.set_f(next_f)
            self.gba_model.calculate()
            self.gba_model.check_model_consistency()
            ### 4.3) If the model is consistent: ###
            if self.gba_model.consistent and self.gba_model.mu >= previous_mu:
                previous_f  = np.copy(next_f)
                epsilon     = self.draw_noise(sigma)
                t           = t + dt
                dt_counter += 1
                t_vec.append(t)
                dt_vec.append(dt)
                mu_vec.append(self.gba_model.mu)
                mu_diff_vec.append(np.abs(self.gba_model.mu-previous_mu))
                ### Check if mu changes significantly ###
                if np.abs(self.gba_model.mu - previous_mu) <= TRAJECTORY_CONVERGENCE_TOL:
                    mu_alteration_counter += 1
                else:
                    mu_alteration_counter = 0
                ### Check if dt is never changing, and possibly increase it ###
                if dt_counter == 1000:
                    dt         = dt*INCREASING_DT_FACTOR
                    dt_counter = 0
            ### 4.4) If the model is inconsistent: ###
            else:
                next_f = np.copy(previous_f)
                self.gba_model.set_f(previous_f)
                self.gba_model.calculate()
                self.gba_model.check_model_consistency()
                assert self.gba_model.consistent, "> Previous model is not consistent"
                if (dt > 1e-100):
                    dt         = dt/DECREASING_DT_FACTOR
                    dt_counter = 0
                else:
                    raise AssertionError("trajectory was stopped, because dt got too small")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 5) Final algorithm steps     #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.plot_trajectory(t_vec, dt_vec, mu_vec, mu_diff_vec)
        if (t>=max_time):
            print("> Max time was reached, Model is consistent for condition: ",condition)
        else:
            print("> Maximum was found, Model is consistent for condition: ",condition)
        end_time      = time.time()
        self.run_time = end_time-start_time


# ####### save Values of model in CSV-Files #######
# def saveValues(model,condition,nameOfCSV=None):
#   dict_arrays = {
#     "Max_growthrate ": model.mu,
#     "F-Vector ": model.f,
#     "Protein_concentrations vector " : model.p,
#     "GCC_F ": model.GCC_f,
#     "Fluxes_vector " : model.v,
#     "Internal_Metabolite_concentrations ": model.c,
#   }
#   dict_arrays_str = {k: [str(v)] for k, v in dict_arrays.items()}
#   df = pd.DataFrame(dict_arrays_str)                                                           # create dataframe

#   if(nameOfCSV is None):
#       df.to_csv(model.model_name +" "+ condition + " Values_at_Max.csv", sep=',', index=False)       # save CSV with autom. File name
#   else:
#     df.to_csv( nameOfCSV , sep=',', index=False)                                           # save CSV with own File name
#   #print(df)
#   return

