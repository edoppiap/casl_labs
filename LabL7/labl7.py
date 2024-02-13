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
import os
from datetime import datetime
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import numpy as np

#-----------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
def get_input_parameters():
    
    parser = argparse.ArgumentParser(description='Input parameters for the simulation')
    
    # simulation parameters
    parser.add_argument('--initial_population', type=int, default=50_000_000,
                        help='Initial number of individual in the system')
    parser.add_argument('--repr_rate', type=float, default=4,
                        help='Reproducibility rate (R(0)) for the simulation') 
    parser.add_argument('--recov_rate', type=float, default=1/14,
                        help='Recovery rate')
    parser.add_argument('--quar_rate', type=float, default=1/3,
                        help='Number of days in which an individual as to be quarantined')
    parser.add_argument('--fatality_rate', type=float, default=.03,
                        help='Fatality rate')
    parser.add_argument('--hospitalized_percentage', type=float, default=.1,
                        help='Percentage of infected individuals that needs to be Hospitalized')
    parser.add_argument('--hosp_places', type=int, default=10_000,
                        help='Number of places available in the hospital system')
    parser.add_argument('--it_places', default=50_000, type=int,
                        help='Number of places for Intensive Treatment')
    parser.add_argument('--it_percentage', type=float, default=.06,
                        help='Percentage of infected individuals that needs to be Intensive Treated')
    parser.add_argument('--death_limit', type=int, default=100_000,
                        help='Maximum number of death in one year')
    parser.add_argument('--sim_time', type=int, default=365,
                        help='Period of time to simulate')
    parser.add_argument('--restr_rate', type=float, default=[1, .8, .75, .7, .5], nargs='+',
                        help='Float between 0 and 1 representing how strong the restriction are')

    # utility parameters
    parser.add_argument('--accuracy_threshold', type=float, default=.8,
                        help='Accuracy value for which we accept the result')
    parser.add_argument('--confidence_level', type=float, default=.8,
                        help='Value of confidence we want for the accuracy calculation')
    parser.add_argument('--verbose', action='store_true',
                        help='See the number of infected during the simulation')
    parser.add_argument('--seed', type=int, default=42, 
                        help='For reproducibility reasons')
    
    return parser.parse_args()

# CREATE OUTPUT FOLDER
#
def create_output_folder():
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    folder_path = os.path.join(script_directory, 'outputs',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    print(f'Output images will be saved in the folder: {folder_path}')
    
    return folder_path

#-----------------------------------------------------------------------------------------------------------#
# RESTRICTION STRATEGY
#
#
def calculate_restriction_strategy(taken,tot,gamma):
    code_int = [.3,.5] # 30% and 50%
        
    if taken < tot*code_int[0]:
        restr_rate = 1 # green code
    elif taken >= tot*code_int[0] and taken < tot*code_int[1]:
        restr_rate = .35  # orange code
    else:
        restr_rate = .1  # red code

    return restr_rate

#-----------------------------------------------------------------------------------------------------------#
# SIR MODEL POPULATION
#
#
def initial_population(n):
    initial_infected = int(1)
    
    # since the population is a function of time
    # store the time also in a list to track the numbers in time
    population = {
        'time': [0],
        'susceptible':[n-initial_infected], # initial susceptible = tot - initial infected
        'infected':[initial_infected], # initial infected = 1
        'quarantined':[0],
        'hospitalized':[0],
        'intensive':[0],
        'recovered':[0],
        'deaths':[0]
    }
    
    return population

#-----------------------------------------------------------------------------------------------------------#
# CALCULATE PROCESS WITH STOCHASTIC INTENSITY
#
#
def marked_poisson(intensity, gamma):
    points = np.random.poisson(intensity)
    
    p_i = intensity / gamma
    
    T_i = np.random.rand(points)
    selected_T_i = T_i[T_i < p_i] # selecting points with p_i probability
    
    return selected_T_i

#-----------------------------------------------------------------------------------------------------------#
# STOCHASTIC SIMULATION 
#
#
def simulate(population: dict, max_time, r_0, gamma, death_rate, hosp_places):
    #---------------------------------#
    # CALCULATE LAMBDA BASED ON R(t=0)
    #   
    lam = (r_0 * gamma) / population['susceptible'][0]
    
    quarantined_gamma = args.quar_rate # days in which an individual as to be quarantined
    
    loop = tqdm(range(1,max_time+1), desc='Simulating with Point process', disable=args.verbose)
    
    #------------------------------------#
    # SIM LOOP
    #
    #
    for day in loop:
        S_t = population['susceptible'][-1]
        I_t = population['infected'][-1]
        Q_t = population['quarantined'][-1]
        H_t = population['hospitalized'][-1]
        T_t = population['intensive'][-1]
        R_t = population['recovered'][-1]
        D_t = population['deaths'][-1]
        
        restr_rate = min(calculate_restriction_strategy(H_t, hosp_places, gamma), calculate_restriction_strategy(T_t, args.it_places, gamma))
        
        infected_intensity = (restr_rate * lam * S_t * I_t) / gamma # new infected depends only from I_t (Q_t and H_t do not infect others)
        quarantined_intensity = quarantined_gamma * I_t # Only the infected can became quaratined
        removed_intensity_I = gamma * (I_t + H_t) # new removed depends from both I_t and H_t
        removed_intensity_Q = gamma * Q_t
        
        if args.verbose:
            print(f'------------------------\n{day = }')
            print(f'{infected_intensity = } \n{removed_intensity_I =}')
        
        #---------------------------------#
        # CALCULATE NEW DAYS INFECTED/REMOVED INDIVIDUALS
        #
        new_infected = marked_poisson(infected_intensity, gamma)
        new_quarantined = marked_poisson(quarantined_intensity, quarantined_gamma)
        try:
            new_removed_I = marked_poisson(removed_intensity_I, gamma)
            new_removed_Q = marked_poisson(removed_intensity_Q, gamma)
        except ValueError:
            print(f'{removed_intensity_I = } \n{I_t = } \n {H_t = }')
            return 
        
        #------------------#
        # NEW SUSCEPTIBLE
        #
        new_S = max(0, S_t - len(new_infected))
        
        #------------------#
        # NEW INFECTED
        #
        new_I = I_t + len(new_infected) if new_S > 0 else I_t + S_t # first take into account that the susceptible could ends
        new_I = max(0, new_I -len(new_removed_I) -len(new_quarantined)) # calculate the new infected
        new_Q = max(0, Q_t + len(new_quarantined) - len(new_removed_Q))
        
        new_R = len(new_removed_I) + len(new_removed_Q)
        
        if args.verbose: print(f'{new_S = }\n{new_I = }')
        
        #---------------------------------#
        # SEPARETE INFECTED INTO COMMON AND HOSPITALIZED
        #
        new_I_effective = int(new_I * (1 - args.hospitalized_percentage - args.it_percentage))
        new_Q_effective = int(new_Q * (1 - args.hospitalized_percentage - args.it_percentage))
        new_H = int((new_I+new_Q) * args.hospitalized_percentage)
        new_T = int((new_I+new_Q) * args.it_percentage)
        
        #---------------------------------#
        # SEPARETE REMOVED INTO RECOVERED AND DEATHS
        #
        new_R_effective = int(R_t + new_R * (1 - death_rate))
        new_D = int(D_t + (len(new_removed_I)+ len(new_removed_Q)) * death_rate)
        
        #---------------------------------#
        # TAKE INTO ACCOUNT THE MAX NUM OF HOSPITAL AND IT PLACES
        #
        if new_H > hosp_places:
            new_D += new_H - hosp_places # the left out will die
            new_H = hosp_places
        if new_T > args.it_places:
            new_D += new_T - args.it_places
            new_T = args.it_places
        
        #---------------------------------#
        # STORE RESULTS
        #
        population['time'].append(day)
        population['susceptible'].append(new_S)
        population['infected'].append(new_I_effective)
        population['quarantined'].append(new_Q_effective)
        population['hospitalized'].append(new_H)
        population['deaths'].append(new_D)
        population['intensive'].append(new_T)
        population['recovered'].append(new_R_effective)
        
        if args.verbose: print(f'{lam = }')
        
        loop.set_postfix_str(f'Susceptible = {S_t}')

#-----------------------------------------------------------------------------------------------------------#
# MEAN FIELD SIMULATION
#
#
def simulate_mean_field(population: dict, max_time, r_0, gamma, death_rate, hosp_places):
    S = population['susceptible'][0]
    I = population['infected'][0]
    R = population['recovered'][0]
    H = population['hospitalized'][0]
    IT = population['intensive'][0]
    Q = population['quarantined'][0]
    D = population['deaths'][0]
    
    lam = (r_0 * gamma) / population['susceptible'][0]
    #lam = .3
    
    if args.verbose: print(f'{lam = }')
    
    #------------------------------------#
    # SIM LOOP
    #
    #
    for day in tqdm(range(1,max_time+1), desc='Simulating with mean_field', disable=args.verbose):
        S_t = population['susceptible'][-1]
        I_t = population['infected'][-1]
        H_t = population['hospitalized'][-1]
        T_t = population['intensive'][-1]
        Q_t = population['quarantined'][-1]
        
        restr_rate = min(calculate_restriction_strategy(H, hosp_places, gamma), calculate_restriction_strategy(IT,args.it_places,gamma))
        
        #---------------------------------#
        # CALCULATE NEW DAYS INFECTED INDIVIDUALS
        #
        new_removed_I = gamma*(I_t + H_t)
        new_removed_Q = gamma*(Q_t)
        new_infected = (lam*restr_rate*S_t*I_t)
        new_quarantined = args.quar_rate * (I_t)        
        
        S -= new_infected
        
        #---------------------------------#
        # SEPARETE INFECTED INTO COMMON AND HOSPITALIZED
        #
        d_i = new_infected - new_removed_I # increase in the infected population
        I += d_i * (1 - args.hospitalized_percentage - args.it_percentage) # add the percentage of common infected
            
        d_q = new_quarantined - new_removed_Q
        Q += d_q * (1 - args.hospitalized_percentage - args.it_percentage)
        
        H += (d_i+d_q) * args.hospitalized_percentage # add the percentage of hospitalized infected
        IT += (d_i+d_q) * args.it_percentage
        
        #---------------------------------#
        # SEPARETE REMOVED INTO RECOVERED AND DEATHS
        #
        R += (new_removed_I+new_removed_Q) * (1-death_rate) # percentage of removed that recover
        D += (new_removed_I+new_removed_Q) * death_rate # percentage of removed that die
        
        if H > hosp_places: # if the are no place left
            # adding the left out to the count of the infected
            D += (H - hosp_places)
            H = hosp_places
        if IT > args.it_places:
            D += IT - args.it_places
            IT = args.it_places
        
        population['susceptible'].append(int(S))
        population['infected'].append(int(I))
        population['hospitalized'].append(int(H))
        population['quarantined'].append(int(Q))
        population['recovered'].append(int(R))
        population['intensive'].append(int(IT))
        population['deaths'].append(int(D))
        population['time'].append(day)

#-----------------------------------------------------------------------------------------------------------#
# PLOT GRAPHS
#
#
def plot_results(output_folder, restr_rate = None, population:dict=None, mean_pop: dict = None, populations: list = None):
    
    if population or mean_pop:
        # plt.figure(figsize=(12,8))
        # #plt.plot(population['time'], population['susceptible'], label='Susceptible')
        # if mean_pop:
        #     #plt.plot(mean_pop['time'], mean_pop['susceptible'], label='Susceptible (mean field)', c='b', linestyle='--')
        #     plt.plot(mean_pop['time'], mean_pop['infected'], label='Infected (mean field)', c='orange', linestyle='--')
        #     #plt.plot(mean_pop['time'], mean_pop['recovered'], label='Recovered', c='g', linestyle='--')
        #     plt.plot(mean_pop['time'], mean_pop['deaths'], label='Deaths (mean field)', c='black', linestyle='--')
        #     plt.plot(mean_pop['time'], mean_pop['hospitalized'], label='Hospitalized (mean field)', c='royalblue', linestyle='--')
        # if population:
        #     #plt.plot(population['time'], population['susceptible'], label='Susceptible', c='b')
        #     plt.plot(population['time'], population['infected'], label='Infected', c='orange')
        #     #plt.plot(population['time'], population['quarantined'], label='Quarantined', c='y')
        #     #plt.plot(population['time'], population['recovered'], label='Recovered', c='g')
        #     plt.plot(population['time'], population['deaths'], label='Deaths', c='black')
        #     plt.plot(population['time'], population['hospitalized'], c='royalblue', label='Hospitalized')
        # plt.axhline(y=args.death_limit, color='r', linestyle='dashdot', label='Desired number of deaths')
        # plt.axhline(y=args.hosp_places, color='b', linestyle='dashdot', label='Max places in hospitals')
        # plt.xlabel('Time')
        # plt.ylabel('N individuals')
        # title_str = f'Epidemic process with rho = {restr_rate}' if restr_rate else 'Epidemic process'
        # plt.title(title_str)
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        
        plt.figure(figsize=(12,8))
        plt.plot(population['time'], population['infected'], label='Infected', c='orange')
        plt.plot(population['time'], population['quarantined'], label='Quarantined', c='y')
        #plt.plot(population['time'], population['recovered'], label='Recovered', c='g')
        plt.plot(population['time'], population['deaths'], label='Deaths', c='black')
        plt.plot(population['time'], population['hospitalized'], c='royalblue', label='Hospitalized')
        plt.plot(population['time'], population['intensive'], c='red', label='Intensive')
        plt.axhline(y=args.death_limit, color='r', linestyle='dashdot', label='Desired number of deaths')
        plt.axhline(y=args.hosp_places, color='b', linestyle='dashdot', label='Max places in hospitals')
        plt.axhline(y=args.it_places, color='g', linestyle='dashdot', label='Max places in intensive treatment')
        plt.xlabel('Time')
        plt.ylabel('N individuals')
        plt.title(f'Stochastic epidemic process simulation')
        plt.legend()
        plt.grid(True)
        file_name = os.path.join(output_folder, f'stochastic.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        # plt.show()
        
        plt.figure(figsize=(12,8))
        plt.plot(mean_pop['time'], mean_pop['infected'], label='Infected', c='orange')
        plt.plot(population['time'], population['quarantined'], label='Quarantined', c='y')
        #plt.plot(mean_pop['time'], mean_pop['recovered'], label='Recovered', c='g', linestyle='--')
        plt.plot(mean_pop['time'], mean_pop['deaths'], label='Deaths', c='black')
        plt.plot(mean_pop['time'], mean_pop['hospitalized'], label='Hospitalized', c='royalblue')
        plt.plot(mean_pop['time'], mean_pop['intensive'], label='Intensive', c='red')
        plt.axhline(y=args.death_limit, color='r', linestyle='dashdot', label='Desired number of deaths')
        plt.axhline(y=args.hosp_places, color='b', linestyle='dashdot', label='Max places in hospitals')
        plt.axhline(y=args.it_places, color='g', linestyle='dashdot', label='Max places in intensive treatment')
        plt.xlabel('Time')
        plt.ylabel('N individuals')
        plt.title(f'Mean field epidemic process simulation')
        plt.legend()
        plt.grid(True)
        file_name = os.path.join(output_folder, f'mean_field.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        # plt.show()

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
    np.random.seed(args.seed)
    
    output_folder = create_output_folder()
    
    population = initial_population(args.initial_population) # population for the stochastic simulation
    
    mean_pop = copy.deepcopy(population) # population for the mean field simulation
    
    simulate(population=population,
                max_time=args.sim_time, 
                r_0=args.repr_rate, 
                gamma=args.recov_rate, 
                death_rate=args.fatality_rate,
                hosp_places=args.hosp_places)
    
    simulate_mean_field(population=mean_pop, 
             max_time=args.sim_time, 
             r_0=args.repr_rate, 
             gamma=args.recov_rate, 
             death_rate=args.fatality_rate,
             hosp_places=args.hosp_places)
    
    plot_results(output_folder, mean_pop=mean_pop, population=population)