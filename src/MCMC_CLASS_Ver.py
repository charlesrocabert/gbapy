import os
import dill
import sys
import numpy as np
import matplotlib.pyplot as plt
# Add the local src directory to the path
sys.path.append('./src/')

class MCMC:
    def __init__(self, model_name, condition="1", population_N=2.5e735, max_time=1e8, sigma=0.01):
        self.model_name = model_name
        self.condition = condition
        self.population_N = population_N
        self.max_time = int(max_time)
        self.sigma = sigma
        self.model = self._load_model(model_name)
        self.model.set_condition(condition)
        self.model.solve_local_linear_problem()
        self.model.calculate()
        self.N_e = population_N
        self.fluxFractions = np.copy(self.model.f)
        self.timestamps = [0]
        self.fixationstamps = []
        self.muRates = [self.model.mu]

    def _load_model(self, model_name):
        filename = f"./binary_models/{model_name}.gba"
        assert os.path.isfile(filename), "ERROR: model not found."
        with open(filename, "rb") as ifile:
            model = dill.load(ifile)
        return model

    def _draw_mutation(self):
        return np.random.normal(0, self.sigma)

    def _mutate_f(self, index):
        non_mutated_f = np.copy(self.model.f_trunc)
        mutated_f = np.copy(self.model.f_trunc)

        alpha = self._draw_mutation()
        mutated_f[index] += alpha
        mutated_f[mutated_f < 0] = 0

        self.model.set_f(mutated_f)
        return non_mutated_f

    def _calc_selection_coefficient(self, mu, mutated_mu):
        return 1 - mu / mutated_mu

    def _simulate_fixation(self, pi):
        return np.random.rand() < pi

    def _calc_pi(self, selection_coefficient):
        if selection_coefficient == 0:
            return 1 / self.N_e
        else:
            return (1 - np.exp(-2 * selection_coefficient)) / (1 - np.exp(-2 * self.N_e * selection_coefficient))

    def run_MCMC(self):
        for t in range(self.max_time):
            reaction_index = np.random.randint(len(self.model.f_trunc))
            current_mu = self.model.mu

            non_mutated_f = self._mutate_f(reaction_index)
            self.model.calculate()
            self.model.check_model_consistency()

            if self.model.consistent:
                mutated_mu = self.model.mu
                s = self._calc_selection_coefficient(current_mu, mutated_mu)
                pi = self._calc_pi(s)

                if not self._simulate_fixation(pi):
                    self.model.set_f(non_mutated_f)
                    self.muRates.append(current_mu)
                    self.timestamps.append(t)
                else:
                    self.timestamps.append(t)
                    self.muRates.append(mutated_mu)
                    self.fixationstamps.append(t)
            else:
                self.model.set_f(non_mutated_f)
                self.muRates.append(current_mu)
                self.timestamps.append(t)

            self.model.calculate()
            self.fluxFractions = np.vstack((self.fluxFractions, self.model.f))

        if len(self.fixationstamps) > 1:
            self.plot_trajectory()
            self.plot_MCMC_fluxfractions()
        else:
            raise AssertionError("No mutation got fixated")

    def plot_trajectory(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.timestamps, self.muRates, label="Mu Rate")
        plt.xlabel('Time')
        plt.ylabel('Mu Rate')
        plt.title('Mu Rate over Time')
        plt.grid(False)
        plt.show()

    def plot_MCMC_fluxfractions(self):
        plt.figure(figsize=(8, 6))
        num_fluxes = len(self.fluxFractions[0])

        for i in range(num_fluxes):
            flux_rate = [row[i] for row in self.fluxFractions]
            plt.plot(self.timestamps, flux_rate, label=self.model.reaction_ids[i])

            for fixation in self.fixationstamps:
                plt.axvline(x=fixation, color='black', linestyle='--', linewidth=0.5)

        plt.xlabel('Time')
        plt.ylabel('Fluxfraction Rate')
        plt.title('Fluxfraction Rate over Time with Highlighted Mutations')
        plt.legend()
        plt.grid(False)
        plt.show()
