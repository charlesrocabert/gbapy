import os
import sys
import dill
import matplotlib.pyplot as plt
# Add the local src directory to the path
sys.path.append('./src/')

# Load the GBA_model class
from GBA_model import *
from GBA_trajectory import *

####### Load the model from a binary file #######
def load_model( model_name ):
    filename = "./binary_models/"+model_name+".gba"
    assert os.path.isfile(filename), "ERROR: model not found."
    ifile = open(filename, "rb")
    model = dill.load(ifile)
    ifile.close()
    return model

#draws mutation coefficient for mutating f
def draw_Mutation(sigma):
   alpha =  np.random.normal(0,sigma)
   return alpha


#calculates the mutated f for each reaction
def mutate_f(model, index, sigma):
  non_mutated_f = np.copy(model.f_trunc) # save non mutated f at index
  mutated_f = np.copy(model.f_trunc) #save copy to of f to mutate.

  alpha = draw_Mutation(sigma)

  mutated_f[index] *= alpha #mutate_f
  mutated_f[mutated_f < 0] = MIN_FLUXFRACTION

  model.set_f(mutated_f)
  return non_mutated_f 

#calculates the selection coefficient
def calc_selection_coefficient(mu, mutated_mu):
   
   return 1 - mu / mutated_mu

def simulate_fixation(pi):
  if np.random.rand() < pi:
        # Fixation occurs
        return True
  else:
        # No fixation, keep last f
        return False

def calc_pi (selection_coefficient,N_e):
   if (selection_coefficient == 0):
            return 1/N_e
   else:
            return (1-np.exp(-2*selection_coefficient)) / (1-np.exp(-2*N_e*selection_coefficient))


def MCMC(model_name = "A", condition = "1", max_time = 1e8, sigma = 0.01, population_N = 2.5e735 ):
  model = load_model(model_name)      # load and run model
  model.set_condition(condition)      # set condition of model
  model.solve_local_linear_problem()  # solve first linear problem
  model.calculate()                   # calc for the first time (maybe not needed)
  N_e = population_N

  timestamps = []                     # save timeStamps for plotting
  y_muRates = []                      # save muRates for plotting
  
  current_mu = model.mu               # save current mu

  for t in range(max_time):
      reaction_index = np.random.randint(len(model.f_trunc))                # generate index to draw f of a random reaction
      #print("choose enzyme: ", reaction_index + 1)
      non_mutated_f = mutate_f(model, reaction_index, sigma) # mutates f temporarily and saves backed up f, for the case if it doesnt fixate.
      model.calculate()                                      # calculate mu with mutated f
      model.check_model_consistency()                                               #check consistency

      if (model.consistent):
         print("consistent")
         mutated_mu = model.mu                 
         s = calc_selection_coefficient(current_mu, mutated_mu)       # calculate selectioncoefficient s
         print("selection coefficient for this mutation: ", s)

         pi = calc_pi(s,N_e)                                          # calculate fixation-propability

         if ( simulate_fixation(pi) == False ):
            print("for pi = "+ str(pi) +" the mutation is not fixated")
            model.set_f(non_mutated_f) # undo  mutated f
         else :
            timestamps = np.append(timestamps, t)
            y_muRates = np.append(y_muRates, model.mu)
      else:
         model.set_f(non_mutated_f)
  #plotTrajectory(timestamps, y_muRates)
  print("y_muRates: ", y_muRates )
  return 
