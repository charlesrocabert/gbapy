import os
import sys
import dill
import matplotlib.pyplot as plt
# Add the local src directory to the path
sys.path.append('./src/')

# Load the GBA_model class
from GBA_model import *

####### Load the model from a binary file #######
def load_model( model_name ):
    filename = "./binary_models/"+model_name+".gba"
    assert os.path.isfile(filename), "ERROR: model not found."
    ifile = open(filename, "rb")
    model = dill.load(ifile)
    ifile.close()
    return model

#draws mutation coefficient for mutating Kcat(f and b)
def draw_Mutation():
  mu_log = np.log(3/2)
  sigma_log = 0.3  

  alpha = np.random.lognormal(mean=mu_log, sigma=sigma_log)

  return alpha

#calculates the mutated Kcat for each reaction
def mutate_kcat(model, index):
  last_kcat_f = model.kcat_f[index] # save non mutated kcat at index
  last_kcat_b = model.kcat_b[index] # save non mutated kcat at index

  alpha = draw_Mutation()

  model.kcat_f = model.kcat_f[index] * alpha #mutate_kcat
  model.kcat_b = model.kcat_b[index] * alpha #mutate_kcat

  return last_kcat_f, last_kcat_b

#calculates the selection coefficient
def calc_selection_coefficient(mu, mutated_mu):
   
   return 1 - mu / mutated_mu

def simulate_fixation(pi):
  if np.random.rand() < pi:
        # Fixation occurs
        return True
  else:
        # No fixation, keep last kcat
        return False


def MCMC(model_name = "A", condition = "1", max_time=10^8, population_N = 2.5e735 ):
  model = load_model(model_name)      # load and run model
  model.set_condition(condition)      # set condition of model
  model.solve_local_linear_problem()  # solve first linear problem
  model.calculate()                   # calc for the first time (maybe not needed)
  N_e = population_N
  current_mu = model.mu               # save current mu

  for t in range(max_time):
      reaction_index = np.random(len(model.kcat_f))                # generate index to draw kcat of a random reaction
      last_kcat_f, last_kcat_b = mutate_kcat(model,reaction_index) # mutates kcat temporarily and saves backed up kcat, for the case if it doesnt fixate.
      mutated_mu = model.calculate()                               # calculate mu with mutated kcats                 
      s = calc_selection_coefficient(current_mu, mutated_mu)       # calculate selectioncoefficient s

      if (s == 0):
        pi = 1/N_e
      else:
         pi = (1-np.exp(-2*s)) / (1-np.exp(-2*N_e*s))

      if ( simulate_fixation(pi) == False ):
         last_kcat_f = model.kcat_f[reaction_index] # undo  mutated kcat at index
         last_kcat_b = model.kcat_b[reaction_index] # undo  mutated kcat at index