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

def plot_MCMC_Fluxfractions(model, f_stamps, time_stamps, fixation_stamps ):
    plt.figure(figsize=(8, 6))
    num_fluxes = len(f_stamps[0])

    # Plot each flux rate curve as a line graph
    for i in range(num_fluxes):
        flux_rate = [row[i] for row in f_stamps]
        #print(flux_rate)
        plt.plot(time_stamps, flux_rate, label = model.reaction_ids[i])

        # Highlight the specific timestamps with vertical lines
        for fixation in fixation_stamps:
            plt.axvline(x=fixation, color='black', linestyle='--', linewidth=0.5)

    # Add labels, title, and legend
    plt.xlabel('Time')
    plt.ylabel('Fluxfraction Rate')
    plt.title('Fluxfraction Rate over Time with Highlighted Mutations')
    plt.legend()
    plt.grid(False)
    return




#draws mutation coefficient for mutating f
def draw_Mutation(sigma):
   alpha =  np.random.normal(0,sigma)
   return alpha


#calculates the mutated f for each reaction
def mutate_f(model, index, sigma):
  non_mutated_f = np.copy(model.f_trunc) # save non mutated f at index
  mutated_f = np.copy(model.f_trunc) #save copy of f to mutate.

  alpha = draw_Mutation(sigma)

  mutated_f[index] += alpha #mutate_f
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


def MCMC(model_name = "A", condition = "1", max_time = 1e8, sigma = 0.01, population_N = 2.5e735, nameOfCSV = None ):
  model = load_model(model_name)      # load and run model
  model.set_condition(condition)      # set condition of model
  model.solve_local_linear_problem()  # solve first linear problem+
  model.calculate()                   # calc for the first time (maybe not needed)
  N_e = population_N

  fluxFractions = np.copy(model.f)    # save fluxfractions for plotting
  timestamps = [0]                     # save timeStamps for plotting
  fixationstamps = []               # save timestamps of fixation for highlighting Mutation in plots
  muRates = [model.mu]                      # save muRates for plotting
  
  current_mu = model.mu               # save current mu

  for t in range(max_time):
      reaction_index = np.random.randint(len(model.f_trunc))                # generate index to draw a random reaction to mutate
      #print("choose enzyme: ", reaction_index + 1)
      current_mu = model.mu
      non_mutated_f = mutate_f(model, reaction_index, sigma) # mutates f temporarily and saves non_mutated f, for no fixation or inconsistency.
      model.calculate()                                      # calculate mu with mutated f
      model.check_model_consistency()                                               #check consistency

      if (model.consistent):
         #print("consistent")
         mutated_mu = model.mu                 
         s = calc_selection_coefficient(current_mu, mutated_mu)       # calculate selectioncoefficient s
         #print("selection coefficient for this mutation: ", s)

         pi = calc_pi(s,N_e)                                          # calculate fixation-propability pi

         if ( simulate_fixation(pi) == False ):
            #print("for pi = "+ str(pi) +" the mutation is not fixated")
            model.set_f(non_mutated_f) # undo  mutated f
            muRates = np.append(muRates, current_mu)
            timestamps = np.append(timestamps, t)

         else :
            timestamps = np.append(timestamps, t)
            muRates = np.append(muRates, mutated_mu)
            fixationstamps = np.append(fixationstamps, t)
      else:
         model.set_f(non_mutated_f)
         muRates = np.append(muRates, current_mu)
         timestamps = np.append(timestamps, t)

      model.calculate()                                              #calculate muRate again , if f didn't fixate
      fluxFractions = np.vstack((fluxFractions, model.f))   #save fluxfractions for plotting
      #print('Timestampscount',len(timestamps))
      #print('Fluxfractioncount',len(fluxFractions))


  if(len(fixationstamps)> 1):
   #fluxFractions = fluxFractions.T
   plotTrajectory(timestamps, muRates)
   plot_MCMC_Fluxfractions(model, fluxFractions, timestamps, fixationstamps)
   saveValues(model,condition,nameOfCSV)

  else:
      AssertionError("no Mutation got fixated")

  #print("y_muRates: ", y_muRates )
  return 
