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

# Load the GBA_model class
from GBA_model import *


class GBA_algorithms:

    ### Class constructor ###
    def __init__( self, model_name ):
        assert os.path.exists("./binary_models/"+model_name+".gba"), "> Model not found"
        self.model_name = model_name
        self.gba_model  = load_model(self.model_name)
        self.condition  = ""
        self.converged  = False
        self.run_time   = 0.0
    
    ### Draw a random normal vector with std 'sigma' and length 'n' ###
    def draw_noise( self ):
        epsilon = np.random.normal(0.0, self.sigma, size=len(self.f_trunc))
        return epsilon

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
    
    ### Compute the gradient ascent without noise ###
    def compute_gradient_ascent( self, condition = "1", max_time = 5, initial_dt = 0.01, dt_changeRate = 0.1, nameOfCSV = None ):
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
            next_f      = np.add(next_f, self.gba_model.GCC_f[1:] * dt)
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
                    dt_counter = 0
                    dt = dt *2.0
            ### 4.4) If the model is inconsistent: ###
            else:
                next_f = np.copy(previous_f)
                self.gba_model.set_f(previous_f)
                self.gba_model.calculate()
                self.gba_model.check_model_consistency()
                assert self.gba_model.consistent, "> Previous model is not consistent"
                if (dt > 1e-100):
                    dt = dt / 5.0
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

# def plot_FluxFractions_to_condition(overview, reaction_ids):
#   plt.figure(figsize=(8, 6))
#   n_reactions = len(reaction_ids)
#   fluxfractions_to_condition = overview.iloc[:, 3:3+ n_reactions].to_numpy()
#   conditions = overview['Cond.'].to_numpy()
#   for i in range(n_reactions):
#         #print(flux_rate)
#         plt.plot(conditions, fluxfractions_to_condition[:, i], label = reaction_ids[i])
#   # Plot each flux rate curve as a line graph
  
#   plt.xlabel('conditions')
#   plt.ylabel('Fluxfraction Rate')
#   plt.title('Fluxfraction Rates over different conditions')
#   plt.legend()
#   plt.grid(False)
#   return

# def plot_Mu_to_condition(overview_df):
#   plt.plot(overview_df['Cond.'], overview_df['mu'], label = 'MaxGrowthrate at condition')
#   plt.xlabel('conditions')
#   plt.ylabel('Max-Grotwthrate')
#   plt.title('Max-Growthrates over different conditions')
#   plt.legend()
#   plt.grid(False)
#   return

# def trajectory_each_condition(model_name = "A", max_time=5, first_dt = 0.01, dt_changeRate = 0.1, nameOfCSV = None):
#   model = load_model(model_name)

#   overview_columns = ['Cond.', 'mu','density','converged?', 'time_to_execute']
#   overview_columns = overview_columns[:3] + model.reaction_ids + overview_columns[3:]
#   overview_df = pd.DataFrame(columns = overview_columns)

#   for condition in model.condition_ids :
#     condition , max_mu, density, fluxfractions, converged, time_to_execute  = trajectory(model_name = model_name , condition = condition, max_time=5, first_dt = 0.01, dt_changeRate=0.1, nameOfCSV = None)
#     overview_dict ={
#       "Cond.": condition,
#       "mu": max_mu,
#       "density": density,
#       "converged?": converged,
#       "time_to_execute": time_to_execute
#     }
#     for reaction_id, fluxfraction in zip(model.reaction_ids, fluxfractions):
#       overview_dict[reaction_id] = fluxfraction

#     overview_row = pd.Series(data = overview_dict)
#     overview_df = pd.concat([overview_df, overview_row.to_frame().T], ignore_index=True)
#   plot_FluxFractions_to_condition(overview_df, model.reaction_ids)
#   plot_Mu_to_condition(overview_df)


#   print(overview_df)
#   overview_df.to_csv(model.model_name + " All conditions.csv", sep=',', index = False)
#   return

# ###### Trajectory with Noise #################
# def trajectoryWithNoise(model_name = "A", condition = "1", max_time = 10000, first_dt = 0.01, dt_changeRate = 0.1, sigma = 0.1, nameOfCSV=None):
#   model = load_model(model_name)      # load and run model
#   model.set_condition(condition)      # set condition of model
#   model.solve_local_linear_problem()  # solve first linear problem
#   model.calculate()                   # calc for the first time (maybe not needed)

#   dt = first_dt
#   t = 0                              # time
#   previous_mu = model.mu
#   mu_alterationCounter = 0              # setup counter for error criteria
#   consistent_f = np.copy(model.f_trunc) # saves consistent_f
#   next_f = np.copy(model.f_trunc)     # the f_trunc, that we are going to change
#   epsilon = drawNoise(sigma, len(next_f) )

#   y_muRates = []                      # save muRates for plotting
#   timestamps = []                     # save timeStamps for plotting

#   while (t < max_time):                                                                 # end loop if time is up
#     print(" current mu-Rate",model.mu)
#     print("time :",t)
#     previous_mu = model.mu
    
#     if(model.mu - previous_mu <= TRAJECTORY_CONVERGENCE_TOL):                                            # check if mu changes significantly
#       mu_alterationCounter = mu_alterationCounter + 1

#     else:
#         mu_alterationCounter = 0

#     if(mu_alterationCounter >= TRAJECTORY_STABLE_MU_COUNT and model.consistent):                         # terminate if mu doesnt change anymore and model is consistent
#         plotTrajectory(timestamps, y_muRates)
#         saveValues(model, condition, model_name + condition + " with noise.csv")
#         print("epsilon", epsilon)
#         raise AssertionError("trajectory was stopped, because the model is consistent and the growthrate did not increase significantly for " + str(TRAJECTORY_STABLE_MU_COUNT) + " tries. ")
        
    
#     next_f = np.add(next_f, (np.add(model.GCC_f[1:] ,epsilon) * dt))                                      # add next_f with noisy Gcc_f
    
                                                                
       
#     next_f[next_f < 0] = MIN_FLUX_FRACTION                                         #negative value correction

#     model.set_f(next_f)
#     model.calculate()                                                             #calculate everything
#     model.check_model_consistency()                                               #check consistency

#     if (model.consistent):
#       timestamps = np.append(timestamps, t)
#       y_muRates = np.append(y_muRates, model.mu)
#       consistent_f = next_f
#       epsilon = drawNoise(sigma, len(next_f) )
#                                                        # saves new f as the consistent f

#       if(mu_alterationCounter >= POSSIBLY_STABLE_MU_ATTEMPTS):                     # if mu doesnt change it might mean, that the optimum is reached. So to terminate this function more quickly by time we increase t by 1.
#         t = t + 1
#       else:
#         t = t + dt                                                                  # else we add dt, which allows us to figure out the optimum.
     

#     else:
#       next_f = consistent_f                                                       #resets next_f to last consistent_f
#       model.set_f(consistent_f)

#       if (dt > 1e-100):                                                           # make sure dt is not too small
#        dt = dt * dt_changeRate
#        t = t + dt                                                                 # calc. new t

#       else:
#           consistent_f = next_f

#   plotTrajectory(timestamps, y_muRates)
#   #saveValues(model, condition, model_name + condition + " with noise.csv")
#   if(model.consistent == True):
#     AssertionError("The trajecory stopped, because the model is consistent but max time ran out.")

#   return 


