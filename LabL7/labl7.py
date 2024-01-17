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
    parser.add_argument('--fatality_rate', type=float, default=.03,
                        help='Fatality rate')
    parser.add_argument('--hospitalized_percentage', type=float, default=.1,
                        help='Percentage of infected individuals that needs to be Hospitalized')
    parser.add_argument('--hosp_places', type=int, default=10_000,
                        help='Number of places available in the hospital system')
    parser.add_argument('--death_limit', type=int, default=100_000,
                        help='Maximum number of death in one year')
    parser.add_argument('--sim_time', type=int, default=365,
                        help='Period of time to simulate')
    parser.add_argument('--restr_rate', type=float, default=[.5,.4,.35,.3], nargs='+',
                        help='Float between 0 and 1 representing how strong the restriction are')

    # utility parameters
    parser.add_argument('--accuracy_threshold', type=float, default=.8,
                        help='Accuracy value for which we accept the result')
    parser.add_argument('--confidence_level', type=float, default=.8,
                        help='Value of confidence we want for the accuracy calculation')
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
        'infected':[int(n*.001)], # starts the simulation with a little number of infected
        'hospitalized':[0],
        'recovered':[0],
        'deaths':[0]
    }
    
    return population

#-----------------------------------------------------------------------------------------------------------#
# SIMULATION
#
#
def simulate(population: dict, max_time, lam, gamma, restr_rate, death_rate, hosp_places):
    S = population['susceptible'][0]
    I = population['infected'][0]
    R = population['recovered'][0]
    H = population['hospitalized'][0]
    D = population['deaths'][0]
    N = S+I+R
    
    #------------------------------------#
    # SIM LOOP
    #
    #
    for day in range(1,max_time):
        
        #---------------------------------#
        # CALCULATE NEW DAYS INFECTED INDIVIDUALS
        #
        new_removed = gamma*(population['infected'][day-1] + population['hospitalized'][day-1])
        new_infected = (lam*restr_rate*population['susceptible'][day-1]*population['infected'][day-1])/N
        S -= new_infected
        
        #---------------------------------#
        # SEPARETE INFECTED INTO COMMON AND HOSPITALIZED
        #
        d_i = new_infected - new_removed # increase in the infected population
        I += d_i * (1 - args.hospitalized_percentage) # add the percentage of common infected
        H += d_i * (args.hospitalized_percentage) # add the percentage of hospitalized infected
        if H > hosp_places: # if the are no place left
            # adding the left out to the count of the infected
            I += d_i * (args.hospitalized_percentage) + (H - hosp_places)
            H = hosp_places
        if H < 0: # there are no hospitalized left to remove
            I += d_i * (args.hospitalized_percentage)
            H = 0
        
        #---------------------------------#
        # SEPARETE REMOVED INTO RECOVERED AND DEATHS
        #
        R += new_removed * (1-death_rate) # percentage of removed that recover
        D += new_removed * death_rate # percentage of removed that die
        
        population['susceptible'].append(int(S))
        population['infected'].append(int(I))
        population['hospitalized'].append(int(H))
        population['recovered'].append(int(R))
        population['deaths'].append(int(D))
        population['time'].append(day)

#-----------------------------------------------------------------------------------------------------------#
# PLOT GRAPHS
#
#
def plot_results(population: dict = None, populations: list = None):
    
    if population:
        plt.figure(figsize=(12,8))
        #plt.plot(population['time'], population['susceptible'], label='Susceptible')
        plt.plot(population['time'], population['infected'], label='Infected', c='orange')
        #plt.plot(population['time'], population['recovered'], label='Recovered', c='g')
        plt.plot(population['time'], population['deaths'], label='Deaths', c='black')
        plt.plot(population['time'], population['hospitalized'], label='Hospitalized')
        plt.axhline(y=args.death_limit, color='r', linestyle='--', label='Desired number of deaths')
        plt.xlabel('Time')
        plt.ylabel('N individuals')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    if populations:
        plt.figure(figsize=(12,8))
        for p,population in populations:
            plt.plot(population['time'], population['deaths'], label=f'Deaths with p = {p}')
        plt.axhline(y=args.death_limit, color='r', linestyle='--', label='Desired max number of deaths')
        plt.xlabel('Time')
        plt.ylabel('N individuals')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(12,8))
        for p,population in populations:
            plt.plot(population['time'], population['hospitalized'], label=f'Hospitalized individuals with p = {p}')
        plt.axhline(y=args.hosp_places, color='r', linestyle='--', label='Max places in hospitals')
        plt.xlabel('Time')
        plt.ylabel('N individuals')
        plt.legend()
        plt.grid(True)
        plt.show()

#-----------------------------------------------------------------------------------------------------------#
# MAIN
#
#
if __name__ == '__main__':
    
    args = get_input_parameters()
    print(f'Input parameters: {vars(args)}')
    
    # SET THE SEED
    #
    random.seed(args.seed)
    
    # folder_path = create_output_folder()
    
    populations = []
    
    for restr_rate in args.restr_rate:
        population = initial_population(args.initial_population)
        
        simulate(population, 
                 max_time=args.sim_time, 
                 lam=args.repr_rate, 
                 gamma=args.recov_rate, 
                 death_rate=args.fatality_rate, 
                 restr_rate=restr_rate,
                 hosp_places=args.hosp_places)
        
        #plot_results(population=population)
        
        populations.append((restr_rate,population))
    plot_results(populations=populations)