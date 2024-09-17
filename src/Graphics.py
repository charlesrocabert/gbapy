#!/usr/bin/env python3
# coding: utf-8

#***********************************************************************
# Copyright © 2024 Charles Rocabert, Furkan Mert
# Web: https://github.com/charlesrocabert/GBA_Evolution
#
# Graphics.py
# -----------
# Definition of various plot functions.
# (LOCAL SCRIPT)
#***********************************************************************

import os
import sys
from matplotlib.pylab import f
import matplotlib.pyplot as plt


# Add the local src directory to the path
sys.path.append('./src/')

from Model import *


### Plot the gradient ascent trajectory ###
def plot_gradient_ascent_trajectory( model ):

    plt.figure(figsize=(12, 8))

    # Plot 1: mu vs Time
    plt.subplot(2, 2, 1)
    plt.plot(model.trajectory["t"], model.trajectory["mu"], color='b', label='mu(t)', linestyle='-', marker='o')
    plt.xlabel('Time')
    plt.ylabel('mu')
    plt.title('Plot of mu Against Time')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend(loc='best')
    plt.tight_layout()

    # Plot 2: dt vs Iteration (Log Scale)
    plt.subplot(2, 2, 2)
    plt.plot(model.trajectory["dt"], color='r', label='dt', linestyle='-', marker='x')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('dt (Log Scale)')
    plt.title('Dt Over Iterations')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend(loc='best')
    plt.tight_layout()

    # Plot 3: mu_diff vs Iteration (Log Scale)
    plt.subplot(2, 2, 3)
    plt.plot(model.trajectory["dmu"], color='g', label='mu_diff', linestyle='-', marker='s')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Mu diff (Log Scale)')
    plt.title('Mu Difference Over Iterations')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend(loc='best')
    plt.tight_layout()

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
### PLot EFM trajectory ###
def plot_EFM_trajectory( model ):
    for i in range(model.nj):
        r_id = model.reaction_ids[i]
        plt.plot(model.trajectory["t"], model.trajectory[r_id], label=r_id)
    plt.xlabel('Time')
    plt.ylabel('EFM proportions')
    plt.legend()
    plt.grid(False)
    plt.show()

    
    
    # Get firt column of f_trajectory
    # x = [row[0] for row in self.f_trajectory]
    # y = [row[1] for row in self.f_trajectory]
    # z = [row[2] for row in self.f_trajectory]
    # #z = self.mu_trajectory
    # ax.plot(x, y, z)

### Plot mu to condition ###
def plot_mu_to_condition( model, path = None ):
    plt.figure(figsize=(12, 6))
    plt.plot(model.optimum_data['condition'], model.optimum_data['mu'], label='MaxGrowthrate at condition')
    plt.xlabel('conditions')
    plt.ylabel('Max-Grotwthrate')
    plt.title('Model  '+ model.model_name+ 'max-growthrates over different conditions')
    plt.legend()
    plt.grid(False)
    plt.gcf()
    plt.show()
    # auto_path = "./output/Model "+model.model_name+" output"
    # if path == None:
    #     if not os.path.exists(auto_path):
    #         os.makedirs(auto_path)
    #         plt.savefig(auto_path+"/"+model.model_name+" Growthrate to all model conditions.png")
    #     else:
    #         plt.savefig(auto_path+"/"+model.model_name+" Growthrate to all model conditions.png")
    # else:
    #         plt.savefig(path)

### Plot f to condition ###
def plot_f_to_condition( model, path = None ):
    f_to_condition = model.optimum_data.iloc[:, 3:3+model.nj].to_numpy()
    conditions     = model.optimum_data['condition'].to_numpy()
    plt.figure(figsize=(12, 6))
    plt.suptitle("Model "+model.model_name+" fluxfractions at optimum to conditions", fontsize=16)

    for i in range(model.nj):
        plt.plot(conditions, f_to_condition[:, i], label = model.reaction_ids[i])

    plt.xlabel('Conditions')
    plt.ylabel('Flux fraction at optimum')
    plt.title('Model ' + model.model_name+ ' flux fractions over different conditions')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.grid(False)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.gcf()
    plt.show()
    # auto_path = "./output/Model "+model.model_name+" output"
    # if path == None:
    #     if not os.path.exists(auto_path):
    #         os.makedirs(auto_path)
    #         plt.savefig(auto_path+"/"+model.model_name+" Flux fractions over all model conditions.png")
    #     else:
    #         plt.savefig(auto_path+"/"+model.model_name+" Flux fractions over all model conditions.png")
    # else:
    #         plt.savefig(path)

### Plot the MCMC trajectory and highlight fixation points ###
def plot_MCMC_trajectory( model, path = None ):
    plt.figure(figsize=(10, 10))
    plt.suptitle("Model "+model.model_name +" condition "+ model.current_condition + " MCMC fluxfraction trajectory", fontsize=16)
    plt.subplot(2, 1, 1)
    for i in range(model.nj):
        r_id = model.reaction_ids[i]
        plt.step(model.trajectory["t"], model.trajectory[r_id], label=r_id)
    plt.xlabel('Time', fontsize = 12)
    plt.ylabel('Fluxfraction rate', fontsize = 12)
    plt.title('Model ' + model.model_name+ ' fluxfraction rate over time with highlighted mutations', fontsize = 14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.step(model.trajectory["t"], model.trajectory["mu"], label='mu(t)',linewidth = 2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Mu(t)', fontsize=12)
    plt.title('Model ' + model.model_name+ 'condition' +model.current_condition+ 'Mu trajectory over time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.gcf()
    plt.show()
    # auto_path = "./output/Model "+model.model_name+" output"
    # if path == None:
    #     if not os.path.exists(auto_path):
    #         os.makedirs(auto_path)
    #         plt.savefig(auto_path+"/"+"condition "+model.current_condition+" MCMC trajectory.png")
    #     else:
    #         plt.savefig(auto_path+"/"+"condition "+model.current_condition+" MCMC trajectory.png")
    # else:
    #         plt.savefig(path)

