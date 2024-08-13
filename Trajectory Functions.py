import os
import sys
import dill
import matplotlib.pyplot as plt
# Add the local src directory to the path
sys.path.append('./src/')

# Load the GBA_model class
from GBA_model import *

### Load the model from a binary file ###
def load_model( model_name ):
    filename = "./binary_models/"+model_name+".gba"
    assert os.path.isfile(filename), "ERROR: model not found."
    ifile = open(filename, "rb")
    model = dill.load(ifile)
    ifile.close()
    return model


### save Values of model in CSV-Files
def saveValues(model,condition,nameOfCSV=None):
  dict_arrays = {
    "Max_growthrate": model.mu,
    "F-Vector": model.f,
    "Protein_concentrations vector" : model.p,
    "GCC_F": model.GCC_f,
    "Fluxes_vector" : model.v,
    "Internal_Metabolite_concentrations": model.c,
    "External_metabolite_concentrations": model.x,
    "Metabolite_concentrations": model.xc,
  }
  dict_arrays_str = {k: [str(v)] for k, v in dict_arrays.items()}

  df = pd.DataFrame(dict_arrays_str)                                                           # Erstellen des DataFrames

  if(nameOfCSV is None):
      df.to_csv(model.model_name +" "+ condition + " Values_at_Max.csv", sep=',', index=False)       # Save CSV with autom. File name
  else:
    df.to_csv( nameOfCSV , sep=',', index=False)                                           # Save CSV with own File name
  print(df)

 
def plotTrajectory(timestamps, muRates):
  # Erstellung des Plots
  plt.figure(figsize=(8, 6))  # Größe des Diagramms festlegen
  plt.plot(timestamps, muRates, label='mu(t)')  # Plot der Daten
  plt.xlabel('Time')  # Beschriftung der X-Achse
  plt.ylabel('mu')    # Beschriftung der Y-Achse
  plt.title('Plot of mu against Time')  # Titel des Diagramms
  plt.grid(True)      # Gitterlinien aktivieren
  plt.legend()        # Legende hinzufügen
  plt.show()          # Diagramm anzeigen
  print("max μ rate :")
  print(np.max(muRates))
  return 

###### Trajectory without Noise
def trajectory(model_name="A",condition="1",max_time=5,first_dt = 0.01,dt_changeRate=0.1,nameOfCSV=None):
  model = load_model(model_name)      #load and run model
  model.set_condition(condition)      #set condition of model
  model.solve_local_linear_problem()  #solve first linear problem
  model.calculate()                   #calc for the first time (maybe not needed)

  dt = first_dt
  t = 0                              # time
  previous_mu = model.mu
  mu_alterationCounter = 0              #setup counter for error criteria
  consistent_f = np.copy(model.f_trunc) # saves consistent_f
  next_f = np.copy(model.f_trunc)     # the f_trunc, that we are going to change
  allGCC_F = [model.GCC_f[1:]]       # to collect all previous GCC_f (just for checking the change of GCC_f)

  y_muRates = []                      #save muRates for plotting
  timestamps = []                     #save timeStamps for plotting

  while (t < max_time):                                                                 # end loop if time is up
    print(" current mu-Rate",model.mu)
    print("time :",t)
    previous_mu = model.mu
    if( ( np.abs(model.GCC_f) <= TRAJECTORY_CONVERGENCE_TOL ).all() and model.consistent):               #(maybe use mu and look if it doesnt change for the next x steps , stop)
      
      break
    
    if(model.mu - previous_mu <= TRAJECTORY_CONVERGENCE_TOL):                                            #maybe decrease Trajectory_Convergence_Tol to 
      mu_alterationCounter = mu_alterationCounter + 1
      #print(mu_alterationCounter)
    else:
        mu_alterationCounter = 0

    if(mu_alterationCounter >= TRAJECTORY_STABLE_MU_COUNT and model.consistent):
        plotTrajectory(timestamps, y_muRates)
        saveValues(model,condition,nameOfCSV)
        raise AssertionError("trajectory was stopped, because the model is consistent and the growthrate did not increase significantly for " + str(TRAJECTORY_STABLE_MU_COUNT) + " tries. ")
    
    if np.any(next_f < 0):                                                            #negative value correction
       #print("next_f before neg.correction:", next_f)
       next_f[next_f < 0] = 1e-10
       #print("next_f after neg.correction:", next_f)

    #print("no Mu alterations: ",mu_alterationCounter)
    print("current gradient :", model.GCC_f)
    print("current protein",model.p)
    print("current Fluxvector ", model.v)
    print("current Tau for protein calc" , model.tau_j)

    #print("current Metabolite :",model.c)

    next_f = np.add(next_f, model.GCC_f[1:] * dt)                                      # add without first index of GCC_f

    model.set_f(next_f)
    model.calculate()
    model.v[ model.v < 0 ] = 1e-10                           # enforce positive flux
    model.p[ model.p < 0 ] = 1e-10                           # enforce positive protein


                                                             #calculate everything
    model.check_model_consistency()                                               #check consistency

    if (model.consistent):
      timestamps = np.append(timestamps,t)
      y_muRates = np.append(y_muRates,model.mu)

      t = t + dt                                                                  # calc. new t
      consistent_f = next_f                                                       # saves new f as the consistent f

    else:
      next_f = consistent_f                                                       #resets next_f to last consistent_f
      model.set_f(consistent_f)

      if (dt > 1e-100):                                                           # make sure dt is not too small
       dt = dt * dt_changeRate
       t = t + dt                                                                 # calc. new t

      else:
        raise AssertionError("trajectory was stopped, because dt got too small")
        break
      
  plotTrajectory(timestamps, y_muRates)
  saveValues(model,condition)
  
  print ("Maximum was found, Model is consistent")
  return

trajectory(model_name="B",condition="2",max_time=2000,first_dt = 0.01,dt_changeRate=0.1)


