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
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Main model parameters    #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.model_name = model_name
        self.gba_model  = load_model(self.model_name)
        self.condition  = ""
        self.optimum_df = None
        self.optimum_f  = {}
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Random initial solutions #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.random_f = {}
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Gradient ascent          #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.t_trajectory   = []
        self.dt_trajectory  = []
        self.mu_trajectory  = []
        self.dmu_trajectory = []
        self.converged      = False
        self.run_time       = 0.0
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) MCMC parameters          #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.fixationTime_trajectory = []
        self.t_trajectory = []
        self.f_trajectory = []
        self.mu_trajectory  = []
        self.population_N = 2.5e735

    ### Plot all fluxfractions over time and highlight fixation points ###
    def plot_MCMC_Fluxfractions(self):
        plt.figure(figsize=(8, 6))
        num_fluxes = len(self.f_trajectory[0])

        for i in range(num_fluxes):
            flux_rate = [row[i] for row in self.f_trajectory]
            plt.plot(self.t_trajectory, flux_rate, label = self.gba_model.reaction_ids[i])

            #for fixation in fixation_stamps:
                #plt.axvline(x=fixation, color='black', linestyle='--', linewidth=0.5)

        plt.xlabel('Time')
        plt.ylabel('Fluxfraction Rate')
        plt.title('Fluxfraction Rate over Time with Highlighted Mutations')
        plt.legend()
        plt.grid(False)
        plt.show()
        
    
    ### Draw mutation coefficient for mutating a single fluxfraction ###
    def draw_Mutation(self, sigma):
        alpha =  np.random.normal(0,sigma)
        return alpha

    ### Calculates the mutated fluxfraction for each reaction ###
    def mutate_f(self, index, sigma):
        non_mutated_f = np.copy(self.gba_model.f_trunc)
        mutated_f = np.copy(self.gba_model.f_trunc) 

        alpha = self.draw_Mutation(sigma)

        mutated_f[index] += alpha 
        mutated_f[mutated_f < MIN_FLUX_FRACTION] = MIN_FLUX_FRACTION

        self.gba_model.set_f(mutated_f)
        return non_mutated_f 
    
    ### Calculate the selection coefficient for MCMC mutation fixation ###
    def calc_selection_coefficient(self, mu, mutated_mu):
        return 1 - mu / mutated_mu
    
    ### Simulate fixation for MCMC ###
    def simulate_fixation(self, pi):
        if np.random.rand() < pi:
                return True
        else:
                return False
        
    ### Calcutlate fixation probability pi for MCMC ###
    def calc_pi (self, selection_coefficient, N_e):
        if (selection_coefficient == 0):
                    return 1/N_e
        else:
                    return (1-np.exp(-2*selection_coefficient)) / (1-np.exp(-2*N_e*selection_coefficient))

    ### Load optimums for all conditions ###
    def load_optimums( self ):
        self.optimum_f.clear()
        optimum_df = pd.read_csv("./csv_models/"+self.model_name+"/optimum.csv", sep=';')
        for i in range(optimum_df.shape[0]):
            condition = optimum_df.iloc[i]['condition']
            self.optimum_f[str(condition)] = optimum_df.iloc[i][3:3+self.gba_model.nj].to_numpy()
    
    ### Generate random initial solutions ###
    def generate_random_initial_solutions( self, condition, nb_solutions ):
        assert condition in self.gba_model.condition_ids, "> Condition not found"
        self.gba_model.set_condition(condition)
        self.random_f.clear()
        solutions  = 0
        trials     = 0
        max_trials = 10000
        while solutions < nb_solutions and trials < max_trials:
            if trials%100 == 0:
                print("> ", solutions, " solutions was found after ", trials, " trials")
            trials += 1
            negative_term = True
            while negative_term:
                f_trunc = np.random.rand(self.gba_model.nj-1)
                f_trunc = f_trunc*FLUX_BOUNDARY
                self.gba_model.set_f(f_trunc)
                if self.gba_model.f[0] >= 0.0:
                    negative_term = False
            self.gba_model.calculate()
            self.gba_model.check_model_consistency()
            if self.gba_model.consistent and np.isfinite(self.gba_model.mu) and self.gba_model.mu > 1e-5:
                solutions += 1
                self.random_f[solutions] = np.copy(self.gba_model.f)
            
    ### Initialize the model with the LP problem ###
    def load_LP_initial_solution( self ):
        self.gba_model.solve_local_linear_problem()
    
    ### Initialize the model with a random initial solution ###
    def load_random_initial_solution( self, solution ):
        assert solution in self.random_f.keys(), "> Solution not found"
        self.gba_model.set_f(self.random_f[solution])

    ### Initialize the model with an optimum solution ###
    def load_optimum_solution( self, condition ):
        assert condition in self.optimum_f.keys(), "> Optimum not found"
        self.gba_model.set_f0(self.optimum_f[condition])
    
    ### Save trajectory into a csv file ###
    def save_trajectory( self, filename ):
        trajectory_df = pd.DataFrame({
            "t": self.t_trajectory,
            "dt": self.dt_trajectory,
            "mu": self.mu_trajectory,
            "dmu": self.dmu_trajectory
        })
        trajectory_df.to_csv("./output/"+filename+".csv", sep=';', index=False)

    ### Plot trajectory ###
    def plot_trajectory( self ):
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 2, 1)
        plt.plot(self.t_trajectory, self.mu_trajectory, label='mu(t)')
        plt.xlabel('Time')
        plt.ylabel('mu')
        plt.title('Plot of mu against Time')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(self.dt_trajectory, label='dt')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('dt')
        plt.title('Dt')
        plt.grid(True)
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(self.dmu_trajectory, label='mu_diff')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Mu diff')
        plt.title('Mu diff')
        plt.grid(True)
        plt.legend()

    ### Plot mu to condition ###
    def plot_mu_to_condition( self ):
        plt.plot(self.optimum_df['condition'], self.optimum_df['mu'], label='MaxGrowthrate at condition')
        plt.xlabel('conditions')
        plt.ylabel('Max-Grotwthrate')
        plt.title('Max-Growthrates over different conditions')
        plt.legend()
        plt.grid(False)

    ### Plot f to condition ###
    def plot_f_to_condition( self ):
        f_to_condition = self.optimum_df.iloc[:, 3:3+self.gba_model.nj].to_numpy()
        conditions     = self.optimum_df['condition'].to_numpy()
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
        self.gba_model.set_condition(condition)
        self.gba_model.calculate()
        self.gba_model.check_model_consistency()
        assert self.gba_model.consistent, "> Initial model is not consistent"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize trackers       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.t_trajectory.clear()
        self.dt_trajectory.clear()
        self.mu_trajectory.clear()
        self.dmu_trajectory.clear()
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
                self.t_trajectory.append(t)
                self.dt_trajectory.append(dt)
                self.mu_trajectory.append(self.gba_model.mu)
                self.dmu_trajectory.append(np.abs(self.gba_model.mu-previous_mu))
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
        self.optimum_df  = pd.DataFrame(columns=overview_columns)
        self.optimum_f   = {}
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
            overview_row    = pd.Series(data=overview_dict)
            self.optimum_df = pd.concat([self.optimum_df, overview_row.to_frame().T], ignore_index=True)
            self.optimum_f[condition] = np.copy(self.gba_model.f)
        self.optimum_df.to_csv("./csv_models/"+self.model_name+"/optimum.csv", sep=';', index=False)
    
    ### Draw a random normal vector with std 'sigma' and length 'n' ###
    def draw_noise( self, sigma ):
        epsilon = np.random.normal(0.0, sigma, size=self.gba_model.nj-1)
        return epsilon

    ### Compute the gradient ascent with noise ###
    def compute_gradient_ascent_with_noise( self, condition = "1", max_time = 5, initial_dt = 0.01, sigma = 0.1 ):
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
        self.t_trajectory.clear()
        self.dt_trajectory.clear()
        self.mu_trajectory.clear()
        self.dmu_trajectory.clear()
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
                self.t_trajectory.append(t)
                self.dt_trajectory.append(dt)
                self.mu_trajectory.append(self.gba_model.mu)
                self.dmu_trajectory.append(np.abs(self.gba_model.mu-previous_mu))
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
        if (t>=max_time):
            print("> Max time was reached, Model is consistent for condition: ",condition)
        else:
            print("> Maximum was found, Model is consistent for condition: ",condition)
        end_time      = time.time()
        self.run_time = end_time-start_time

    ### Compute Markov chain Monte Carlo ###    
    def MCMC(self, condition = "1", max_time = 1e8, sigma = 0.01, population_N = 2.5e7, nameOfCSV = None ):
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 1) Initialize the model      #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        self.gba_model.set_condition(condition)
        self.gba_model.calculate()
        self.gba_model.check_model_consistency()
        assert self.gba_model.consistent, "> Initial model is not consistent"
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 2) Initialize trackers       #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        fluxFractions = np.copy(self.gba_model.f)   
        timestamps = [0]                     
        fixationstamps = []               
        muRates = [self.gba_model.mu]                      
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 3) Initialize the algorithm  #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        fluxFractions = np.copy(self.gba_model.f)   
        N_e = population_N
        current_mu = self.gba_model.mu
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # 4) Start the MCMC            #
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        for t in range(max_time):
            ### 4.1 Draw reaction to mutate at random ###
            reaction_index = np.random.randint(len(self.gba_model.f_trunc))                
            current_mu = self.gba_model.mu
            non_mutated_f = self.mutate_f(reaction_index, sigma) 
            self.gba_model.calculate()                                      
            self.gba_model.check_model_consistency()                                               
            ### 4.2 Check model consistency and simulate fixation ###
            if (self.gba_model.consistent):
                mutated_mu = self.gba_model.mu                 
                s = self.calc_selection_coefficient(current_mu, mutated_mu)       
                pi = self.calc_pi(s,N_e)                                          
            ### 4.3 Undo Mutation if no fixation occurs ###
                if ( self.simulate_fixation(pi) == False ):
                    self.gba_model.set_f(non_mutated_f) 
                    muRates = np.append(muRates, current_mu)
                    timestamps = np.append(timestamps, t)
            ### 4.4 Save Mutation for trajectory if fixation occurs ###
                else :
                    timestamps = np.append(timestamps, t)
                    muRates = np.append(muRates, mutated_mu)
                    fixationstamps = np.append(fixationstamps, t)
            ### 4.5 Undo Mutation if model is inconsistent ###
            else:
                self.gba_model.set_f(non_mutated_f)
                muRates = np.append(muRates, current_mu)
                timestamps = np.append(timestamps, t)

            self.gba_model.calculate()                                              
            fluxFractions = np.vstack((fluxFractions, self.gba_model.f))   
           
        if(len(fixationstamps)> 1):
            self.fixationTime_trajectory = np.copy(fixationstamps)
            self.t_trajectory = np.copy(timestamps)
            self.f_trajectory = np.copy(fluxFractions)
            self.mu_trajectory  = np.copy(muRates)

        else:
            AssertionError("no Mutation got fixated")

         
