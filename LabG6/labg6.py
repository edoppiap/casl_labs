"""
    Consider a simulator for natural selection with the following simplified simulation model:

    All the individuals belong to the same species
    The initial population is equal to P
    The reproduction rate for each individual is lambda
    The lifetime LF(k) of individual k whose parent is d(k) is distributed according to the following distribution:
    LF(k)=
    uniform(LF(d(k),LF(d(k)*(1+alpha)) with probability prob_improve   
    uniform(0,LF(d(k)) with probability 1-prob_improve

    where prob_improve  is the probability of improvement for a generation and alpha is the improvement factor (>=0)

    Answer to the following questions:

    Describe some interesting questions to address based on the above simulator.
    List the corresponding output metrics.
    Develop the simulator
    Define some interesting scenario in terms of input parameters.
    Show and comment some numerical results addressing the above questions.
    Upload only the py code and the report (max 2 pages).
"""

import random
from queue import PriorityQueue
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
parser = argparse.ArgumentParser(description='Input parameters for the simulation')

# Population parameters
parser.add_argument('--prob_improve', '--p_i', type=float, default=.01,
                    help='Probability of improvement of the lifetime')
parser.add_argument('--init_population', '--p', type=int, default=50,
                    help='Number of individuals for the 1st generation')
parser.add_argument('--improve_factor', '--alpha', type=float, default=.2,
                    help='Improve factor that an individual can develop at birth')
parser.add_argument('--init_lifetime', type=int, default=50, 
                    help='Lifetime of the 1st generation')
parser.add_argument('--repr_rate', '--lambda', type=float, default=3.5,
                    help='Rate at which an individual reproduces')

# Simulation parameters
parser.add_argument('--sim_time', type=int, default=1000,
                    help='Time to run the simulation')
parser.add_argument('--accuracy_threshold', type=float, default=.8,
                    help='Accuracy value for which we accept the result')
parser.add_argument('--confidence_level', type=float, default=.8,
                    help='Value of confidence we want for the accuracy calculation')
parser.add_argument('--verbose', action='store_true',
                    help='To see the progress of the simulation')
parser.add_argument('--seed', type=int, default=42, 
                    help='For reproducibility reasons')

class Individual():
    def __init__(self, birth_time, param, parent_lf):
        lam, alpha, p_i = param 
        self.birth_time = birth_time
        self.lifetime = random.uniform(parent_lf, parent_lf*(1+alpha))\
            if random.random() < p_i else random.uniform(0, parent_lf)
        self.lam = lam
        
    def __str__(self) -> str:
        return f'Birth_time: {self.birth_time:.2f} - Lifetime: {self.lifetime:.2f}'
    
    def get_reproduce_time(self, current_time):
        return current_time + random.expovariate(self.lam)
            
    def get_death_time(self, current_time):
        return current_time + self.lifetime
    
class Event:
    def __init__(self, event_time, event_type, individual: Individual):
        self.time = event_time
        self.type = event_type
        self.ind = individual
        
    def __lt__(self, other):
        return self.time < other.time

def gen_init_population(init_p, lam, alpha, init_lifetime):
    return [Individual(0, # born all at the same time
                       lam=lam, 
                       alpha=alpha, 
                       p_i=0, # the 1st gen can't improve
                       parent_lf=init_lifetime) for _ in range(init_p)]

# we can schedule a new birth based on len(population) after the new_death
def death(current_time, param, population):
    pass

# we have to schedule a new death associated to the new_born
# we can schedule a new birth based on len(population) after the new_born
def birth(current_time, FES: PriorityQueue,  param, parent_lf, population):
    new_born = Individual(birth_time=current_time,
                          param=param,
                          parent_lf=parent_lf)
    population.append(new_born)
    
    # schedule the death associated with the new_born
    FES.put(Event(new_born.get_death_time(), 'death', new_born))
    
    # schedule a new birth with the new len(population)
    birth_time = random.expovariate(lam*len(population))
    rand_individual = population[random.randint(0, len(population)-1)]
    FES.put(Event(birth_time, 'birth', rand_individual))

#-----------------------------------------------------------------------------------------------------------#
# MAIN METHOD
#
#
if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Input parameters: {vars(args)}')
    
    # CREATE OUTPUT FOLDER
    #
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'outputs',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    print(f'Output images will be saved in the folder: {folder_path}')
    
    FES = PriorityQueue()
    time = 0
    lam = args.repr_rate
    param = lam, args.improve_factor, p_i=args.p_i
    
    population = gen_init_population(init_p=args.init_population,
                                          lam=lam,
                                          alpha=args.improve_factor,
                                          init_lifetime=args.init_lifetime)
    
    for individual in population:
        FES.put(Event(individual.get_death_time(), 'death', individual))
    
    # the birth process follows a Poisson distr with lam = sum(lambdas)
    # because each one individual reproduce followint a Poisson distr with lam 
    birth_time = random.expovariate(lam*len(population))
        
    # first event to start the simulation
    rand_individual = population[random.randint(0, len(population)-1)]
    first_repr = Event(birth_time, 'birth', rand_individual)
    FES.put(first_repr)
    
    #----------------------------------------------------------------#
    # EVENT LOOP
    #
    while time < args.sim_time:
        if FES.empty():
            pass
        
        event = FES.get()
        time = event.time
        
        if event.type == 'birth':
            parent = event.individual
            birth(current_time=time,
                  FES=FES,
                  param=param,
                  parent_lf=parent.lifetime,
                  population=population)
        elif event.type == 'death':
            death()