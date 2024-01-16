"""
    The goal of Lab L7 is to define and simulate simple strategies 
    to  control an  epidemic (SIR) process through non pharmaceutical interventions
    (I.e. by introducing mobility restrictions).

    Consider a homogeneous population of 50M individuals.
    Fix R(0)=4 and gamma= 1/14 days (recovering rate).    
    Assume that  10% (6%) of the infected individuals  needs to be Hospitalized (H)  (undergo Intensive Treatments (IT).)
    
    Fix the fatality rate of the epidemic to 3%.
    H/IT places are limited (10k/50 k). Design some  non pharmaceutical intervention strategy that avoids H/IT overloads, 
    and limits the number of death in 1 year to 100K.
    To design your strategy you can use a mean-field SIR model.

    Then, once you have defined your strategy simulate both the stochastic SIR and its mean field.  Are there significant differences, why? 
    What happens if you scale down your population N to 10K (of course you have to scale down also other parameters, such as H and IT places)?


    For the plagiarism checks, please upload your code here: 
    https://www.dropbox.com/request/FSm4b6hTRu8qArbe9ImF
"""

import argparse
import os, datetime
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#-----------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
def get_input_parameters():
    
    parser = argparse.ArgumentParser(description='Input parameters for the simulation')
    
    # simulation parameters
    parser.add_argument('--initial_population', type=int, default=50_000_000,
                        help='Initial number of individual in the system')
    parser.add_argument('--repr_rate', type=float, default=.3,
                        help='Reproducibility rate (R(0)) for the simulation') 
    parser.add_argument('--recov_rate', type=float, default=.1,
                        help='Recovery rate')
    parser.add_argument('--fatality_rate', type=float, default=.3,
                        help='Fatality rate')
    parser.add_argument('--hospitalized_percentage', type=float, default=.1,
                        help='Percentage of infected individuals that needs to be Hospitalized')
    parser.add_argument('--hosp_places', type=int, default=[10_000, 50_000], nargs='+',
                        help='Number of places available in the hospital system')
    parser.add_argument('--death_limit', type=int, default=100_000,
                        help='Maximum number of death in one year')
    parser.add_argument('--sim_time', type=int, default=1000,
                        help='Period of time to simulate')

    # utility parameters
    parser.add_argument('--accuracy_threshold', type=float, default=.8,
                        help='Accuracy value for which we accept the result')
    parser.add_argument('--confidence_level', type=float, default=.8,
                        help='Value of confidence we want for the accuracy calculation')
    parser.add_argument('--verbose', action='store_true',
                        help='To see the consensus sum of the nodes')
    parser.add_argument('--seed', type=int, default=42, 
                        help='For reproducibility reasons')
    
    return parser.parse_args()

# CREATE OUTPUT FOLDER
#
def create_output_folder():
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'outputs',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    print(f'Output images will be saved in the folder: {folder_path}')
    
    return folder_path

#-----------------------------------------------------------------------------------------------------------#
# SIR MODEL POPULATION
# R(0) = 4 => base reproduction number
# gamma = 1/14 days => recovery rate
#
#
def initial_population(n):
    
    # since the population is a function of time
    # store the time also in a list to track the numbers in time
    population = {
        'time': [0],
        'susceptible':[n], # at t=0 the only variable is the susceptible one
        'infected':[n*.01],
        'removed':[0]
    }
    
    return population

#-----------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
def simulate(population: dict, max_time, lam, gamma):
    S = population['susceptible'][0]
    I = population['infected'][0]
    R = population['removed'][0]
    N = S+I+R
    
    for t in range(1,max_time):
        I += (lam*population['susceptible'][t-1]*population['infected'][t-1])/N -gamma*population['infected'][t-1]
        R += gamma*population['infected'][t-1]
        S -= (lam*population['susceptible'][t-1]*population['infected'][t-1])/N
        
        population['susceptible'].append(S)
        population['infected'].append(I)
        population['removed'].append(R)
        population['time'].append(t)

def plot_results(population: dict):
    
    plt.figure(figsize=(12,8))
    plt.plot(population['time'], population['susceptible'], label='Susceptible')
    plt.plot(population['time'], population['infected'], label='Infected')
    plt.plot(population['time'], population['removed'], label='Removed')
    plt.xlabel('Time')
    plt.ylabel('N individuals')
    plt.legend()
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    
    args = get_input_parameters()
    print(f'Input parameters: {vars(args)}')
    
    # SET THE SEED
    #
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # folder_path = create_output_folder()
    
    population = initial_population(args.initial_population)
    
    simulate(population, max_time=args.sim_time, lam=args.repr_rate, gamma=args.recov_rate)
    
    plot_results(population)