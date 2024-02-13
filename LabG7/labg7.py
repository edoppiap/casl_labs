"""
    Consider a simulator for natural selection with the following simplified simulation model:

    All the individuals belong to S different species
    Let s be the index of each species, s=1...S
    The initial population of s is equal to P(s)
    The reproduction rate for each individual is lambda
    The theoretical lifetime LF(k) of individual k whose parent is d(k) is distributed according to the following distribution:
    LF(k)=
    uniform(LF(d(k),LF(d(k)*(1+alpha)) with probability prob_improve   
    uniform(0,LF(d(k)) with probability 1-prob_improve

    where prob_improve  is the probability of improvement for a generation and alpha is the improvement factor (>=0)

    The individuals move randomly in a given region and when individuals of different species meet, they may fight and may not survive. In such a case, the actual lifetime of a individual may be lower than its theoretical lifetime. 
    A died individual cannot reproduce.
    Answer to the following questions:

    Describe some interesting questions to address based on the above simulator.
    List the corresponding input metrics.
    List the corresponding output metrics.
    Describe in details the mobility model with finite speed
    Describe in details the fight model and the survivabilty model
    Develop the simulator
    Define all the events you used and their attribute
    Define some interesting scenario in terms of input parameters.
    Show and comment some numerical results addressing the above questions.
    Upload only the py code and the report (max 3 pages).
"""

""" 
    MODELLO LOTKA-VOLTERRA
    alpha = max prey growth rate
    beta = prey mortality rate
    
    gamma = predator mortality rate
    delta = predation rate
    
    x = prey population
    y = predator population
    
    instantaneus growth rates prey = dx / dt = alpha * x - beta * x * y
    instantaneus growth rates predator = dy / dt = delta * x * y - gamma * y
    
    --multiprocessing --not_debug --accuracy_threshold 0.8 --confidence_leve 0.8 --move_rate .01 --prob_improve 1 --improve_factor 0
    
    --verbose --prob_improve 1 --improve_factor 0 --grid_dimentions 10 --sim_time 1000 --percentage .5 --fight_rate 0.025 --repr_rate 0.05 --increase_factor 1.01
    
    --prob_improve 1 --improve_factor 0 --grid_dimentions 5 --sim_time 3000 --fight_rate 0.2 --repr_rate 0.025 --increase_factor 1.1 --decrease_factor 0.9 --initial_transient 30 --max_population 5000 --init_population 500 --percentage .3 --puberty_time 15

    --prob_improve 1 --improve_factor 0 --grid_dimentions 5 --sim_time 3000 --fight_rate 0.25 --repr_rate 0.035 --increase_factor 1.1 --decrease_factor 0.9 --initial_transient 30 --max_population 5000 --init_population 500 --percentage .25 --puberty_time 15
    
    --verbose --prob_improve 1 --improve_factor 0 --grid_dimentions 5 --sim_time 3000 --fight_rate 0.25 --repr_rate 0.035 --increase_factor 1.1 --decrease_factor 0.9 --initial_transient 30 --max_population 5000 --init_population 500 --percentage .25 --puberty_time 15
"""

#--------------------------------------------------------------------------------------------------------------------------------------------#
# IMPORTS
#
#
import pandas as pd
import numpy as np
import random
from queue import PriorityQueue
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import copy
import numpy as np
from scipy.stats import t

#--------------------------------------------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
parser = argparse.ArgumentParser(description='Input parameters for the simulation')

# Population parameters
parser.add_argument('--prob_improve', '--p_i', type=float, default=[.8, .5, .2, 1], nargs='+',
                    help='Probability of improvement of the lifetime')
parser.add_argument('--init_population', '--p', type=int, default=[500], nargs='+',
                    help='Number of individuals for the 1st generation')
parser.add_argument('--improve_factor', '--alpha', type=float, default=[.5, .5, .5, 0], nargs='+',
                    help='Improve factor that an individual can develop at birth')
parser.add_argument('--repr_rate', '--lambda', type=float, default=[.05], nargs='+',
                    help='Rate at which an individual reproduces')
parser.add_argument('--max_population', type=int, default=800,
                    help='This semplified version need a limit otherwise will infinite grow')
parser.add_argument('--grid_dimentions', type=int, default=5,
                    help='Side of the square of the grid dimension')
parser.add_argument('--win_prob', type=float, default=.2,
                    help='Probability that a single predator has to have a successful hunt')
parser.add_argument('--percentages', default=.4, type=float,
                    help='Percentage of predators out of total')
parser.add_argument('--initial_transient', default=30, type=int,
                    help='Initial period in which the individuals do not fight but only reproduce')
parser.add_argument('--move_rate', default=[250/365], type=float, nargs='+',
                    help='Rate at which the individual change position')
parser.add_argument('--decrease_factor', type=float, default=.89,
                    help='Decrease factor for reducing the repr_rate')
parser.add_argument('--increase_factor', type=float, default=1.009,
                    help='Decrease factor for reducing the repr_rate')
parser.add_argument('--pop_threshold', type=int, default=250,
                    help='Number of individual that change the repr_rate in a population')

# Simulation parameters
parser.add_argument('--sim_time', type=int, default= 25 * 365, # 12 years
                    help='Time to run the simulation')
parser.add_argument('--calculate_accuracy', action='store_true', 
                    help='This will run a single simulation per combination without calculating the accuracy')
parser.add_argument('--accuracy_threshold', type=float, default=.9,
                    help='Accuracy value for which we accept the result')
parser.add_argument('--confidence_level', type=float, default=.9,
                    help='Value of confidence we want for the accuracy calculation')
parser.add_argument('--verbose', action='store_true',
                    help='To see the progress of the simulation')
parser.add_argument('--seed', type=int, default=100, 
                    help='For reproducibility reasons')
parser.add_argument('--not_debug', action='store_true',
                    help='This will save the images into a directory')
parser.add_argument('--multiprocessing', action='store_true',
                    help='This will run the simulation in parallel processes')

#--------------------------------------------------------------------------------------------------------------------------------------------#
# SPECIES INPUT PARAMETERS
#
#
SPECIES = {
    'fox': {
        'name':'fox',
        'init_lifetime':356 * 5, # 15 years
        'improv_factor':.2,
        'prob_improve':.9,
        'av_repr_rate': 4 / 365, # 4 times a year on average
        'av_num_child_per_birth': 6,
        'type':'predator',
        'puberty_time': .15, # 6 months
        'possible_day_with_no_food': 21, # 3 weeks
        'fight_rate': 1/7,
        'starve_rate': 1/31,
        'in_heat_period': 7, # 1 week
        'pregnancy_duration': 40, 
        'days_between_hunts': 3,
        'attack_on_group': True
    },
    'rabbit': {
        'name':'rabbit',
        'init_lifetime':356 * 4, # 10 years
        'improv_factor':.2,
        'prob_improve':.9,
        'av_repr_rate': 5 / 365, # 5 time a year on average 
        'av_num_child_per_birth': 3,
        'type':'prey',
        'puberty_time': .15, # 6 months
        'possible_day_with_no_food': None,
        'fight_rate':None,
        'starve_rate': None,
        'in_heat_period': 7, # 1 week
        'pregnancy_duration':  25, 
        'days_between_hunts': None,
        'attack_on_group': None
    }
}

#--------------------------------------------------------------------------------------------------------------------------------------------#
# MEASURE CLASS
#
#
class Measure:
    def __init__(self):
        self.num_birth = 0
        self.num_death = 0
        self.average_pop = 0
        self.time_last_event = 0
        # self.num_fight = 0
        self.end_time = 0
        # self.starve_death = 0
        # self.mate_event = 0
        self.species_exctinct_time = {}
        self.births_time = {'time':[], 'births':[]}
        self.time_size_pop = {}
        self.birth_per_species = {}
        self.death_per_species = {}
        self.birth_per_gen_per_species = {}
        self.in_heat_event = {}
        self.num_win = []
        self.wolf_death_time = None
        self.repr_rate = {}
        self.stringa = ''
        
    def increment_gen_birth(self, gen, species):
        if species['name'] not in self.birth_per_gen_per_species:
            self.birth_per_gen_per_species[species['name']] = {}
        self.birth_per_gen_per_species[species['name']][gen]['n birth'] = self.birth_per_gen_per_species[species['name']].setdefault(gen,{'n birth':0, 'tot lf':0, 'tot l':0})['n birth'] + 1
        
    def increment_gen_lf(self, gen, lf, species):
        if species['name'] not in self.birth_per_gen_per_species:
            self.birth_per_gen_per_species[species['name']] = {}
        self.birth_per_gen_per_species[species['name']][gen]['tot lf'] = self.birth_per_gen_per_species[species['name']].setdefault(gen,{'n birth':0, 'tot lf':0, 'tot l':0})['tot lf'] + lf
        
    def increment_gen_l(self, gen, l, species):
        if species['name'] not in self.birth_per_gen_per_species:
            self.birth_per_gen_per_species[species['name']] = {}
        self.birth_per_gen_per_species[species['name']][gen]['tot l'] = self.birth_per_gen_per_species[species['name']].setdefault(gen,{'n birth':0, 'tot lf':0, 'tot l':0})['tot l'] + l

#--------------------------------------------------------------------------------------------------------------------------------------------#
# WORLD CLASS (LATTICE WITH SOME FUNCTIONS)
#
#
class World(dict):
    def __init__(self, dim,population):
        super().__init__(self.generate_world(dim,population))
        self.dim = dim
        
    def __getitem__(self, key):
        return super().__getitem__(key)
        
    def generate_world(self,dim,population):
        world = {(x,y): {'individuals':[], 'neighbors':[]} for x in range(dim) for y in range(dim)}
        
        for ind in [ind for pop in population.values() for ind in pop ]:
            world[ind.current_position]['individuals'].append(ind)
        
        for x in range(dim):
            for y in range(dim):
                if x+1 < dim:
                    world[(x,y)]['neighbors'].append((x+1,y))
                    world[(x+1,y)]['neighbors'].append((x,y))
                if y+1 < dim:
                    world[(x,y)]['neighbors'].append((x,y+1))
                    world[(x,y+1)]['neighbors'].append((x,y))
        return world

#--------------------------------------------------------------------------------------------------------------------------------------------#
# POPULATION CLASS (DICT WITH SOME FUNCTIONS)
#
#      
class Population(dict):
    def __init__(self, init_p, world_dim, species, percentages, FES, init_transient):
        super().__init__(self.gen_init_situation(init_p, world_dim, species, percentages, FES, init_transient))
        
    #---------------------------------------------------------------------#
    # GENERATION OF INITIAL POPULATION
    #
    #
    def gen_init_situation(self, init_p, world_dim, species, percentages, FES: PriorityQueue, init_transient):
        
        init_ps = [int(init_p * percent) for percent in percentages] # 20% wolf - 80% sheep
        
        for specie,n_p in zip(species, init_ps):
            for _ in range(n_p):
                new_born = Individual(birth_time=0,
                                    father_lf=species[specie]['init_lifetime'],
                                    gen=0,
                                    species=species[specie],
                                    world_dim=world_dim,
                                    mother = True)
                birth_time = random.uniform(0, init_transient)
                FES.put(Event(birth_time, 'birth', new_born))
        
        return {specie: [] for specie in species}
    
    def __len__(self) -> int:
        return sum(len(individuals) for individuals in self.values())

#--------------------------------------------------------------------------------------------------------------------------------------------#
# INDIVIDUAL CLASS
#
#
class Individual():
    def __init__(self, father_lf, gen, species: dict, mother = None, world_dim=None, birth_time = None):
        self.sex = 'Y' if random.random() < .5 else 'X' # Y -> male - X -> Female
        self.gen = gen
        self.birth_time = birth_time
        if self.gen != 0 and random.random() < species['prob_improve']:
            self.lifetime = random.uniform(father_lf, father_lf*(1+species['improv_factor']))
        else:
            self.lifetime = random.uniform(0, father_lf) # first individual can't improve
        
        self.current_position = None
        if world_dim:
            self.current_position = (random.randint(0,world_dim-1),random.randint(0,world_dim-1))
        
        self.species = species
        self.puberty_time = species['puberty_time'] * self.lifetime
        if species['type'] == 'predator':
            self.last_hunt_time = self.birth_time
        
        if self.sex == 'X':
            self.in_heat = False
            self.pregnant = False
        else:
            self.sterile = False
        self.mother = mother
        
    def __str__(self) -> str:
        return f'Birth_time: {self.birth_time:.2f} - Lifetime: {self.lifetime:.2f}'

    def get_age(self, current_time):
        return current_time - self.birth_time

#--------------------------------------------------------------------------------------------------------------------------------------------#
# EVENT CLASS
#
#
class Event:
    def __init__(self, event_time, event_type, individual: Individual = None):
        self.time = event_time
        self.type = event_type
        self.individual = individual
        
    def __lt__(self, other):
        return self.time < other.time
        

#--------------------------------------------------------------------------------------------------------------------------------------------#
# DEATH EVENT
#
#
# we can schedule a new birth based on len(population) after the new_death
def death(current_time, population: Population, individual: Individual, data: Measure, world: World):
    if individual in population[individual.species['name']]:
        data.average_pop += len(population)*(current_time - data.time_last_event)
        data.time_last_event = current_time
        data.num_death+=1
        data.death_per_species[individual.species['name']] = data.death_per_species.setdefault(individual.species['name'], 0) + 1
        actual_lifetime = current_time - individual.birth_time
        data.increment_gen_l(individual.gen, l=actual_lifetime, species=individual.species)
        population[individual.species['name']].remove(individual)
        world[individual.current_position]['individuals'].remove(individual)

#--------------------------------------------------------------------------------------------------------------------------------------------#
# MATE EVENT
#
#
# in an mate event a male looks for a female that is in-heat, if he finds her thet will reproduce
def mate(current_time, current_ind: Individual, FES: PriorityQueue, data: Measure, world: World):
    pos = world[current_ind.current_position] # position in which the male is located
    if len(pos['individuals']) > 2: # proceeds if there are at least other individuals in the same position
        # look for other individual of the same species
        other_same_species = [ind for ind in pos['individuals'] if ind.species['name'] == current_ind.species['name'] and ind != current_ind] 
        
        # if there are any, return because no reproduction is possible
        if len(other_same_species) == 0: 
            return
        
        # seeks female in heat 
        females_in_heat = [ind for ind in other_same_species if ind.sex == 'X' and ind.in_heat]
        
        # for female in females_in_heat: # this will mean that a male will reproduce with all the female in the same position currently in heat
        
        if len(females_in_heat) > 0:
            # data.mate_event += 1
            female = random.choice(females_in_heat) # the male choose a single female to reproduce with
            av_lifetime = (female.lifetime + current_ind.lifetime) / 2 # calculate the average lifetime to pass at the new_born
            female.pregnant = True # store that the female can't enter in new heat period while is pregnant
            female.in_heat = False
            n_child = 0
            while n_child == 0:
                n_child = np.random.poisson(female.species['av_num_child_per_birth']) # generate how many new_born this birth event will have
            for _ in range(n_child): 
                new_born = Individual(father_lf=av_lifetime,
                            gen=female.gen+1,
                            species=female.species,
                            mother = female)
                FES.put(Event(current_time + female.species['pregnancy_duration'], 'birth', new_born))

#--------------------------------------------------------------------------------------------------------------------------------------------#
# FIGHT EVENT
#
#
# in a fight event the predator will seek for preys, if there are more predator in the same position, they will form a group of max 3 individual
# to have higher chance of killing the prey        
def fight(current_time, population: Population, individual: Individual, FES: PriorityQueue, data: Measure, world: World, init_win_prob):
    if individual not in population[individual.species['name']]: # check if the individual is dead
        return
    
    if current_time - individual.last_hunt_time >= individual.species['days_between_hunts']: # if the individual has already eaten is not interested in hunt again
        
        try:
            # winning probability depends on the number of predators individual in the whole population (this will prevent to have overkill predators)
            win_prob = min(1, init_win_prob / (len(population[individual.species['name']]) / (len(population['rabbit'])/2.3))) 
        except ZeroDivisionError:
            return
        
        pos = world[individual.current_position]
        if len(pos['individuals']) > 2:
            other_predators = [ind for ind in pos['individuals'] if ind.species['type'] == 'predator' # select all predator
                            and ind != individual # but not the current individual
                            and current_time - ind.last_hunt_time >= ind.species['days_between_hunts']] # not all predator are interested in eating a new prey
            # select all preys
            hide_prob = min(1, .2 * (len(population['rabbit']) / len(population['fox'])/2)) 
            preys = [ind for ind in pos['individuals'] if ind.species['type'] == 'prey' and
                     random.random() < hide_prob] # % of the preys is able to successfully hide from the predator
            
            if len(preys) > 0: # otherwise there is no change to have a fight TODO: models if two individuals of the same species can fight eachother
            
                # each predator wants to hunt a prey
                # (it can be also model that more than one predator hunt the same prey,
                #  the winning prob should be higher in that case)
                if False: # a predator attack a single prey 
                    prey = random.choice(preys)
                    if random.random() < 1: # TODO: the winning prob should be 1) an input or 2) a parameters given by the inds chars
                        # predator wins
                        death_event = Event(current_time, 'death', prey)
                        FES.put(death_event) # kill the prey
                        
                        # update predator variables
                        individual.last_hunt_time = current_time# schedule the next starve event
                        starve_time = 0
                        while starve_time < individual.species['possible_day_with_no_food']:
                            starve_time = random.expovariate(individual.species['starve_rate'])
                        FES.put(Event(current_time + starve_time, 'starve', individual))
                        
                        # schedule the next fight event
                        hunt_time = 0
                        while hunt_time < individual.species['days_between_hunts']:
                            hunt_time = random.expovariate(individual.species['fight_rate'])
                        FES.put(Event(current_time + hunt_time, 'fight', individual))
                        data.num_win.append((1,0))
                        return
                    else:
                        data.num_win.append((0,1))
                else: # more than one predator can attack a single prey
                    # create the group 
                    # if there is a single predator the group will be formed only by him
                    n_predator_each_pray = 3
                    # n_groups_of_predators = math.ceil(len(predators) / n_predator_each_pray)                    
                    # predators_tuples = [(predators[i*3:(i+1)*3] + [None] * (3 - len(predators[i*3:(i+1)*3]))) for i in range(n_groups_of_predators)]
                    
                    group_of_predators = [individual]
                    for _ in range(n_predator_each_pray):
                        if len(other_predators) == 0: break
                        other_predator = random.choice(other_predators)
                        group_of_predators.append(other_predator)
                        other_predators.remove(other_predator)                        
                    
                    # for each group select a random prey and start a fight
                    prey_to_attack = min(preys, key=lambda x: x.get_age(current_time)) # the youngest
                    # prey_to_attack = random.choice(preys) # the group choice a prey to hunts
                    
                    # that the total winnin probability depends on the number of predators that takes part at the fight 
                    win = win_prob * len(group_of_predators) 
                    if random.random() < win: # predators win
                        data.num_win.append((1,0))
                        FES.put(Event(current_time, 'death', prey_to_attack)) # add the death event for the prey into the FES
                        for predator in group_of_predators:
                            predator.last_hunt_time = current_time # saving that the predators has eaten
                            
                            # here i'm not scheduling new events associated ad each predator
                            # because this will add too many events during the simulation
                        
                    else: # prey win
                        data.num_win.append((0,1))
    
    #---------------------------------------------------------------------------------------------------------------------------------------#
    # schedule the next fight and starve event
    #
    #
    # schedule only this individual new events to prevent the accumulate of events     
            
    # schedule a new fight event because the predator hasn't eaten
    hunt_time = 0
    while hunt_time < individual.species['days_between_hunts']:
        hunt_time = random.expovariate(individual.species['fight_rate']) # the predator will hunt almost immediately because he's hungry
    FES.put(Event(current_time + hunt_time, 'fight', individual))
    
    # schedule the next starve event
    starve_time = 0
    while starve_time < individual.species['possible_day_with_no_food']: # TODO: this should be an input parameter
        starve_time = random.expovariate(individual.species['starve_rate'])
    FES.put(Event(current_time + starve_time, 'starve', individual))

#--------------------------------------------------------------------------------------------------------------------------------------------#
# STARVE EVENT
#
#      
def starve(current_time, population: Population, individual: Individual, FES: PriorityQueue, data: Measure, world: World):
    
    days_without_food = current_time - individual.last_hunt_time
    
    # check if the individual has eaten in the past days
    if individual in population[individual.species['name']] \
        and individual.species['type'] == 'predator' \
        and days_without_food > individual.species['possible_day_with_no_food']: # check if the individual hasn't eaten yet
            
            # to prevent predator to go exctinct to quicly when there are lack of prey
            starve_prob = min(1, max(0.2, (len(population['fox']) / len(population['rabbit']))))
            
            if random.random() < starve_prob:
                FES.put(Event(current_time, 'death', individual))
            else:
                starve_time = 0
                while starve_time < individual.species['possible_day_with_no_food']: 
                    starve_time = random.expovariate(individual.species['starve_rate'])
                FES.put(Event(current_time + starve_time, 'starve', individual))
        
#--------------------------------------------------------------------------------------------------------------------------------------------#
# MOVE EVENT
#
#      
def move(current_time, population: Population, individual: Individual, move_rate, FES: PriorityQueue, data: Measure, world: World):
    if individual in population[individual.species['name']]:
        pos = world[individual.current_position]
        new_position = random.choice(pos['neighbors'])
        pos['individuals'].remove(individual)
        world[new_position]['individuals'].append(individual)
        individual.current_position = new_position
            
        next_move_time = current_time + random.expovariate(move_rate)
        FES.put(Event(next_move_time, 'move', individual))
        
        #----------------------------------------------------------------------------------------------------------#
        # the move event trigger possible mating
        #
        if individual.sex == 'Y': # if it's a male
            in_heat_females = [ind for ind in world[individual.current_position]['individuals'] 
                 if ind.species['name'] == individual.species['name'] and
                 ind.sex == 'X' and
                 (ind.gen == 0 or current_time - ind.birth_time > ind.puberty_time) and
                 ind.in_heat]
            if len(in_heat_females) > 0: # if there are any female in heat
                FES.put(Event(current_time, 'mate', individual))
        elif individual.in_heat: # if it's a female in heat
            males = [ind for ind in world[individual.current_position]['individuals'] 
                 if ind.species['name'] == individual.species['name'] and
                 ind.sex == 'Y' and
                 (ind.gen == 0 or current_time - ind.birth_time > ind.puberty_time) and
                 not ind.sterile]
            if len(males) > 0: # if there are any reproductive male
                FES.put(Event(current_time, 'mate', random.choice(males)))            
            

#--------------------------------------------------------------------------------------------------------------------------------------------#
# BIRTH EVENT
#
#
# we have to schedule a new death associated to the new_born
# we can schedule a new birth based on len(population) after the new_born
def birth(current_time, new_born: Individual, FES: PriorityQueue, population: Population, move_rate, data: Measure, world: World, pop_limit):
    data.average_pop += len(population)*(current_time - data.time_last_event)
    data.time_last_event = current_time
    
    # check if the parent is still alive because it can be killed 
    specie = new_born.species['name']
    pop = population[specie]
    
    # mother is True -> is a first_birth
    # mother is an individual -> is a common birth
    if (isinstance(new_born.mother, bool) and new_born.mother is True) or (isinstance(new_born.mother, Individual) and new_born.mother in pop):
        if current_time > 0:
            data.birth_per_species[specie] = data.birth_per_species.setdefault(specie, 0) + 1
            data.num_birth += 1
        
        if isinstance(new_born.mother, Individual):
            new_born.current_position = new_born.mother.current_position
            new_born.mother.pregnant = False
            heat_time = current_time + random.expovariate(new_born.species['av_repr_rate'])
            FES.put(Event(heat_time, 'start_heat', new_born.mother))
            
        if new_born.birth_time is None:
            new_born.birth_time = current_time
            new_born.last_hunt_time = current_time
        world[new_born.current_position]['individuals'].append(new_born)
        population[specie].append(new_born)
        # data.increment_gen_birth(parent.gen)
        # data.increment_gen_lf(new_born.gen, new_born.lifetime)
        data.increment_gen_birth(new_born.gen, species=new_born.species)
        data.increment_gen_lf(new_born.gen, new_born.lifetime, species=new_born.species)

        # schedule the death associated with the new_born
        death_time = current_time + new_born.lifetime
        FES.put(Event(death_time, 'death', new_born))
        
        #----------------------------------------------------------------------------------------------------------#
        # Sterility probability
        # this will works as upper-bond for a population from growth exponentially
        #
        ster_prob = min(.95, (len(population[specie]) / pop_limit))
        
        # the higher the number of individual, the more likely is that the new_born is sterile
        if random.random() > ster_prob: # if ster_prob = 40% -> 60% pass this condition (60% are reproductive)
            # the new_born is not sterile
            # schedule the first heat event relative if it's a female
            if new_born.sex == 'X':
                if new_born.gen != 0:
                    heat_time = current_time + new_born.puberty_time + random.expovariate(new_born.species['av_repr_rate'])
                else:
                    heat_time = current_time + random.expovariate(new_born.species['av_repr_rate'])
                FES.put(Event(heat_time, 'start_heat', new_born))
            # if it's a male it's already not sterile
        else:
            # the new_born is sterile
            if new_born.sex == 'Y':
                new_born.sterile = True
            # if it's a female not schedule the first heat event is enough to make it sterile
            
        if new_born.species['type'] == 'predator':
            # schedule the first starve event
            starve_time = 0
            while starve_time < new_born.species['possible_day_with_no_food']:
                starve_time = random.expovariate(new_born.species['starve_rate'])
            FES.put(Event(current_time + starve_time, 'starve', new_born))
            
            # schedule the first fight event
            hunt_time = 0
            while hunt_time < new_born.species['days_between_hunts']:
                hunt_time = random.expovariate(new_born.species['fight_rate'])
            FES.put(Event(current_time + hunt_time, 'fight', new_born))
        
        # schedule the first move event
        next_move_time = current_time + random.expovariate(move_rate)
        FES.put(Event(next_move_time, 'move', new_born))

#--------------------------------------------------------------------------------------------------------------------------------------------#
# START IN HEAT EVENT
#
#
def start_in_heat(current_time, individual: Individual, population: Population, FES: PriorityQueue, world: World, data:Measure):
    data.in_heat_event[individual.species['name']] = data.in_heat_event.setdefault(individual.species['name'], 0) +1
    if individual in population[individual.species['name']] and individual.sex == 'X' and not individual.pregnant: # check if it's a female to be sure
        individual.in_heat = True
        FES.put(Event(current_time + individual.species['in_heat_period'], 'stop_heat', individual)) # 
        
        males = [ind for ind in world[individual.current_position]['individuals'] 
                 if ind.species['name'] == individual.species['name'] and
                 ind.sex == 'Y' and
                 (ind.gen == 0 or current_time - ind.birth_time > ind.puberty_time) and
                 not ind.sterile]
        if len(males) > 1:
            FES.put(Event(current_time, 'mate', random.choice(males)))

#--------------------------------------------------------------------------------------------------------------------------------------------#
# STOP IN HEAT EVENT
#
#  
def stop_in_heat(current_time, individual: Individual, population: Population, FES: PriorityQueue):
    if individual in population[individual.species['name']] and individual.sex == 'X' and individual.in_heat: # check if it's a female to be sure
        individual.in_heat = False
        
        heat_time = current_time + random.expovariate(individual.species['av_repr_rate'])
        FES.put(Event(heat_time, 'start_heat', individual))

#--------------------------------------------------------------------------------------------------------------------------------------------#
# SIMULATION
#
#
def simulate(init_p, percent, move_rate, world_dim, species, data: Measure, args):
    FES = PriorityQueue()
    t = 0
    
    population = Population(init_p=init_p, world_dim=world_dim, species=species, percentages=percent, FES=FES, init_transient=args.initial_transient)
    world = World(world_dim, population)
    
    for s in species:
        data.species_exctinct_time[s] = None
    
    #----------------------------------------------------------------#
    # EVENT LOOP
    #
    while not FES.empty():
        
        event = FES.get()
        
        t = event.time        
        
        if args.verbose:
            if t > 0:
                
                print(f'{len(population["fox"]) = } - {len(population["rabbit"]) = } - time = {t:.2f} (days)' + ' '*5, end='\r')
            else:
                print(f'Initializing the population...' + ' '*20, end='\r')
        
        if event.type == 'birth':
            birth(current_time=t,
                  new_born=event.individual,
                  move_rate=move_rate,
                  FES=FES,
                  population=population,
                  data=data,
                  world=world,
                  pop_limit=args.max_population)
        elif event.type == 'death':
            death(current_time=t,
                  population=population,
                  individual=event.individual,
                  data=data,
                  world=world)
        elif event.type == 'start_heat':
            start_in_heat(current_time=t,
                        individual=event.individual,
                        population=population,
                        world=world,
                        FES=FES, 
                        data=data)
        elif event.type == 'stop_heat':
            stop_in_heat(current_time=t,
                        individual=event.individual,
                        population=population,
                        FES=FES)
        elif event.type == 'move':
            move(current_time=t,
                 individual=event.individual,
                 population=population,
                 move_rate=move_rate,
                 FES=FES,
                 data=data,
                 world=world)
        elif event.type == 'mate':
            mate(current_time=t,
                 current_ind=event.individual,
                 FES=FES,
                 data=data,
                 world=world)
        elif event.type == 'fight':
            fight(current_time=t,
                  individual=event.individual,
                  FES=FES,
                  data=data,
                  population=population,
                  world=world,
                  init_win_prob=args.win_prob)
        elif event.type == 'starve':
            starve(current_time=t,
                   individual=event.individual,
                   population=population,
                   FES=FES,
                   data=data,
                   world=world)
        
        # STORE THE LEN OF THE POPULATION PER SPECIES
        if t > 0 :
            # previous_day = int(t)
            data.time_size_pop.setdefault('time',[]).append(t)
            for specie in population.keys():
                data.time_size_pop.setdefault(specie,[]).append(len(population[specie]))
                
        if t > args.initial_transient:
            for name in population:
                if len(population[name]) == 0:
                    data.species_exctinct_time[name] = t
        
        if len(population) == 0  or t > args.sim_time or len(population) > args.max_population * 2:
            break       
        
        # if len(population['fox']) == 0 and data.wolf_death_time is None:
        #     data.wolf_death_time = t
        #     break
    
    
        # if len(population['fox']) == 0 and data.wolf_death_time is None:
        #     data.wolf_death_time = t
        #     break
    
    if args.verbose:
        print(f'{len(population) = }')
        print(f'{len(population["fox"]) = }\n{len(population["rabbit"]) = }')
        for name in population:
            print(f'{name} birth events: {data.birth_per_species.setdefault(name, 0)}')
        print(f'Num wolf win: {sum(predator_win for predator_win, _ in data.num_win)}')
        print(f'Num sheep win: {sum(sheep_win for _,sheep_win in data.num_win)}')
        print(f'End {t = :.2f}')
        
    data.end_time = t

#--------------------------------------------------------------------------------------------------------------------------------------------#
# PLOT RESULTS
#
#
def plot_results(data: Measure, param, current_time: str = None, folder_path = None, end_time = None, idx=None):
    init_p, prob_improve_prey, impr_factor_prey, prob_improve_predator, impr_factor_predator, \
    percent, move_rate, world_dim, args, _ = param
            
    time_size_pop = data.time_size_pop
    
    if prob_improve_predator == 1 and prob_improve_prey == 1:
        output_str = f'NO NATURAL SELECTION SIMULATION'
    elif prob_improve_prey > prob_improve_predator:
        output_str = f'NATURAL SELECTION SIMULATION (prey evolve better than predator)'
    elif prob_improve_prey < prob_improve_predator:
        output_str = f'NATURAL SELECTION SIMULATION (prey evolve worse than predator)'
    else:
        output_str = f'NATURAL SELECTION SIMULATION (prey and predator evolve with equal probability)'
        

    output_str += '\n--------------------------------------------\n\n'
    
    # output_str += f'Input parameters: \n__{init_p = },\n__{prob_improve_prey = },\n__{impr_factor_prey = },\n__{prob_improve_predator = },\n' \
    #             + f'__{impr_factor_predator = }\n__repr_rate_prey = {repr_rate_prey*365} in_heat/year,\n__repr_rate_predator = {repr_rate_predator * 365} in_heat/year,\n'\
    #             + f'__{starve_rate = }\n__{fight_rate = }\n'\
    #             + f'__{num_child_predator = },\n__{num_child_prey = }\n__{heat_period_predator = },\n__{heat_period_prey = },\n__{days_between_hunts = },\n' \
    #             + f'__{puberty_time_predator = },\n__{percent = },\n__{puberty_time_prey = },\n__move_rate = {move_rate * 365} move/year\n__{world_dim = }\n' \
    #             + f'__{args.decrease_factor = },\n__{args.increase_factor = }\n__{args.pop_threshold = }\n' \
    #             + '--------------------------------------------\n\n'
    output_str += f'Output measures:\n'
    output_str += f'__Wolf predation rate = {sum(i for i,_ in data.num_win) / (data.end_time - args.initial_transient)} hunts/day\n'
    for species in data.birth_per_species:
        output_str += f'__{species.capitalize()} growth rate = {data.birth_per_species[species] / data.end_time} birth/day\n' \
                    + f'__{species.capitalize()} total birth events: {data.birth_per_species[species]}\n' 
        if species in data.death_per_species:
            output_str += f'__{species.capitalize()} have a new_death with a rate of {data.death_per_species[species] / data.end_time} deaths/day\n' \
                    + f'__{species.capitalize()} total deaths events: {data.death_per_species[species]}\n' \
                    + '--------------------------------------------\n'
    for name in data.in_heat_event:
        output_str += f'{name.capitalize()} in_heat_event = {data.in_heat_event[name]}\n'
    # for name in data.repr_rate:
    #     output_str += f'{name.capitalize()} repr_rate:'
    #     for repr_rate in data.repr_rate[name]:
    #         output_str += f', {repr_rate}'
    #     output_str += '\n'
        
    # output_str += data.stringa
    
    if folder_path:
        file_name = os.path.join(folder_path, f'{idx}_{current_time}_logs.txt')
        with open(file_name, 'w') as f:
            f.write(output_str)
    else:
        print(output_str)

    if time_size_pop is not None:
        plt.figure(figsize=(12,8))
        plt.plot([t/365 for t in time_size_pop['time']],time_size_pop["fox"],label='fox')
        plt.plot([t/365 for t in time_size_pop['time']],time_size_pop["rabbit"],label='rabbit')
        if args.initial_transient != 0:
            plt.axvline(x=args.initial_transient/365, ls='--', color='g', label='End initial transient')
        plt.xlabel('Time (years)')
        plt.ylabel('Size of population')
        plt.title(f'Population size over time')
        plt.grid()
        plt.legend()
        if folder_path:
            file_name = os.path.join(folder_path, f'{idx}_{current_time}_pop_time_species.')
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    birth_per_gen_per_species = data.birth_per_gen_per_species
    
    if birth_per_gen_per_species:
        plt.figure(figsize=(12,8))
        for species in birth_per_gen_per_species:
            plt.bar(list(birth_per_gen_per_species[species].keys()), [gen['n birth'] for gen in birth_per_gen_per_species[species].values()], 
                    label=species.capitalize(),
                    alpha=.5)
        plt.xlabel('Number of generation')
        plt.ylabel('Number of birth')
        plt.title(f'Number of birth events per generation')
        plt.legend()
        plt.grid(True)
        if folder_path:
            file_name = os.path.join(folder_path, f'{idx}_{current_time}_birth_gen_time_species.')
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        plt.figure(figsize=(12,8))
        for species in birth_per_gen_per_species:
            try:
                plt.bar(list(birth_per_gen_per_species[species].keys()), [(gen['tot lf'] / gen['n birth']) / 365 for gen in birth_per_gen_per_species[species].values()], 
                        label=species.capitalize(),
                        alpha = .5)
            except ZeroDivisionError:
                print(f'Found ZeroDivisionError in {species} species')
        plt.xlabel('Number of generation')
        plt.ylabel('Average life expectancy (years)')
        plt.title(f'Life expectancy (without considering the death for fight outcomes)\n per generation')
        plt.legend()
        plt.grid(True)
        if folder_path:
            file_name = os.path.join(folder_path, f'{idx}_{current_time}_life_expectancy_gen_time_species.')
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
def plot_accuracy_results(acc_results, current_time: str = None, folder_path = None):
    
    plt.figure(figsize=(12,8))
    for prob_imp in [p for p in acc_results['prob_improve_prey'].unique()]:
        selected_df = acc_results[acc_results['prob_improve_prey'] == prob_imp]
        plt.plot(selected_df['prob_improve_predator'], selected_df['mean_lf_predator']/365, marker='o', label=f'Improve probability prey = {prob_imp}')
        plt.errorbar((selected_df['prob_improve_predator']), selected_df['mean_lf_predator']/365, 
                    yerr=[selected_df['mean_lf_predator']/365 - selected_df['interval_lf_predator_low']/365,
                        selected_df['interval_lf_predator_up']/365 - selected_df['mean_lf_predator']/365],
                    fmt='o', capsize=5, c='black', zorder=0)
    plt.xlabel('Predator probability of improvement')
    plt.ylabel('Average life expectancy (years)')
    plt.title('Average life expectancy (years) vs predator probability of improvement')
    plt.grid(True)
    plt.legend()
    if folder_path:
        file_name = os.path.join(folder_path, f'{current_time}_av_lf_predator.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    plt.figure(figsize=(12,8))
    for prob_imp in [p for p in acc_results['prob_improve_prey'].unique()]:
        selected_df = acc_results[acc_results['prob_improve_prey'] == prob_imp]
        plt.plot(selected_df['prob_improve_predator'], selected_df['mean_l_predator'], marker='o', label=f'Improve probability prey = {prob_imp}')
        plt.errorbar((selected_df['prob_improve_predator']), selected_df['mean_l_predator'], 
                    yerr=[selected_df['mean_l_predator'] - selected_df['interval_l_predator_low'],
                        selected_df['interval_l_predator_up'] - selected_df['mean_l_predator']],
                    fmt='o', capsize=5, c='black', zorder=0)
    plt.xlabel('Predator probability of improvement')
    plt.ylabel('Average actual lifetime (days)')
    plt.title('Average actual lifetime (days) vs predator probability of improvement')
    plt.grid(True)
    plt.legend()
    if folder_path:
        file_name = os.path.join(folder_path, f'{current_time}_av_l_predator.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    improvement_probabilities_prey = acc_results['prob_improve_prey']
    life_expectancy_prey = acc_results['mean_lf_prey']/365  # Life expectancy of prey for each improvement probability

    improvement_probabilities_predator = acc_results['prob_improve_predator']
    life_expectancy_predator = acc_results['mean_lf_predator']/365 # Life expectancy of predator for each improvement probability

    bar_width = 0.35
    plt.figure(figsize=(18,8))
    x = np.arange(len(improvement_probabilities_prey))
    plt.bar(x - bar_width/2, life_expectancy_prey, width=bar_width, label='Rabbit')
    plt.bar(x + bar_width/2, life_expectancy_predator, width=bar_width, label='Fox')
    plt.xlabel('Improvement Probability')
    plt.ylabel('Life Expectancy (years)')
    plt.title('Effect of Improvement Probability on Life Expectancy')
    x_labels = [f'{prey} / {predator}' for prey, predator in zip(improvement_probabilities_prey, improvement_probabilities_predator)]
    plt.xticks(x, x_labels)
    plt.legend()
    plt.grid(True)
    if folder_path:
        file_name = os.path.join(folder_path, f'{current_time}_overal_lf.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
    improvement_probabilities_prey = acc_results['prob_improve_prey']
    life_expectancy_prey = acc_results['mean_l_prey']/365  # Life expectancy of prey for each improvement probability

    improvement_probabilities_predator = acc_results['prob_improve_predator']
    life_expectancy_predator = acc_results['mean_l_predator']/365 # Life expectancy of predator for each improvement probability

    bar_width = 0.35
    plt.figure(figsize=(18,8))
    x = np.arange(len(improvement_probabilities_prey))
    plt.bar(x - bar_width/2, life_expectancy_prey, width=bar_width, label='Rabbit')
    plt.bar(x + bar_width/2, life_expectancy_predator, width=bar_width, label='Fox')
    plt.xlabel('Improvement Probability')
    plt.ylabel('Actual lifetime (years)')
    plt.title('Effect of Improvement Probability on Actual lifetime')
    x_labels = [f'{prey} / {predator}' for prey, predator in zip(improvement_probabilities_prey, improvement_probabilities_predator)]
    plt.xticks(x, x_labels)
    plt.legend()
    plt.grid(True)
    if folder_path:
        file_name = os.path.join(folder_path, f'{current_time}_overal_l.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    plt.figure(figsize=(12,8))
    plt.plot(acc_results['prob_improve_prey'], acc_results['mean_exctint_time_prey'], marker='o', label='Average exctint time')
    # plt.errorbar(selected_df['bias prob'], selected_df['consensus time mean'], 
    #             yerr=[selected_df['consensus time mean'] - selected_df['time interval low'],
    #                 selected_df['time interval up'] - selected_df['consensus time mean']],
    #             fmt='o', capsize=5, c='black', zorder=1)
    # init_p, prob_improve_prey, impr_factor_prey, prob_improve_predator, impr_factor_predator, repr_rate_prey, \
    #     repr_rate_predator, num_child_predator, num_child_prey, heat_period_predator, heat_period_prey, \
    #     days_between_hunts, puberty_time_predator, puberty_time_prey, percent, move_rate, world_dim, args, _, _ = param
    plt.xlabel('Probability of improvement')
    plt.ylabel('Mean time of exctintion')
    plt.title('Mean time of extinction vs probability of improvement (prey)')
    plt.grid(True)
    plt.legend()
    if folder_path:
        file_name = os.path.join(folder_path, f'{current_time}_extinction_time_prob.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    
#--------------------------------------------------------------------------------------------------------------------------------------------#
# CREATE OUTPUT FOLDER
#
#   
def create_folder_path():
    # CREATE OUTPUT FOLDER
    #
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'outputs',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    print(f'Output images will be saved in the folder: {folder_path}')
    return folder_path

def simulate_wrapper(param):
    init_p, prob_improve_prey, impr_factor_prey, prob_improve_predator, impr_factor_predator, \
        percent, move_rate, world_dim, args, _, seed = param
    print(f'Simualte with {prob_improve_prey = }, {prob_improve_predator = }, {seed = }')
    
    species = copy.deepcopy(SPECIES)
    random.seed(seed)
    np.random.seed(seed)
    
    species['fox']['improv_factor'] = impr_factor_predator
    species['fox']['prob_improve'] = prob_improve_predator
    
    species['rabbit']['improv_factor'] = impr_factor_prey
    species['rabbit']['prob_improve'] = prob_improve_prey
    
    data = Measure()
    simulate(init_p=init_p, 
        world_dim=world_dim, 
        data=data,
        species=species,
        percent=percent,
        move_rate=move_rate,
        args=args)
    # plot_results(data, folder_path=folder_path, param=param, end_time=data.end_time)
    return data

# -------------------------------------------------------------------------------------------------------#
# CONDIFENCE INTERVAL METHOD
#
#
def calculate_confidence_interval(data: list, conf):
    data = [data for data in data if data is not None]
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / (len(data)**(1/2)) # this is the standard error
    
    interval = t.interval(confidence = conf, # confidence level
                          df = len(data)-1, # degree of freedom
                          loc = mean, # center of the distribution
                          scale = se # spread of the distribution 
                          # (we use the standard error as we use the extimate of the mean)
                          )
    
    MOE = interval[1] - interval[0] # this is the margin of error
    re = (MOE / (2 * abs(mean))) # this is the relative error
    acc = 1 - re # this is the accuracy
    return mean,interval,acc

def intervals_wrapper(param:tuple, tot_results: list, conf, acc_threshold):
    # mean_time,interval_time,acc_time = calculate_confidence_interval([data.end_time for data in tot_results],conf)
    acc_exctint_time_prey, acc_exctint_time_predator = 0,0
    mean_exctint_time_prey,interval_exctint_time_prey = None,None
    prey_extinction_times = [data.species_exctinct_time['rabbit'] for data in tot_results if data.species_exctinct_time['rabbit']]
    if len(prey_extinction_times) > 1:
        mean_exctint_time_prey,interval_exctint_time_prey, acc_exctint_time_prey = calculate_confidence_interval(prey_extinction_times,conf)
    
    predator_extinction_times = [data.species_exctinct_time['fox'] for data in tot_results if data.species_exctinct_time['fox']]
    mean_exctint_time_predator,interval_exctint_time_predator = None,None
    if len(predator_extinction_times) > 1:
        mean_exctint_time_predator,interval_exctint_time_predator, acc_exctint_time_predator = calculate_confidence_interval(predator_extinction_times,conf)
    
    av_lf_prey = [value['tot lf'] / value['n birth'] for data in tot_results for key,value in data.birth_per_gen_per_species['rabbit'].items() if key > 0]
    mean_lf_prey,interval_lf_prey,acc_lf_prey = calculate_confidence_interval(av_lf_prey,conf)
    av_lf_predator = [value['tot lf'] / value['n birth'] for data in tot_results for key,value in data.birth_per_gen_per_species['fox'].items() if key > 0]
    mean_lf_predator,interval_lf_predator,acc_lf_predator = calculate_confidence_interval(av_lf_predator,conf)
    
    av_l_prey = [value['tot l'] / value['n birth'] for data in tot_results for key,value in data.birth_per_gen_per_species['rabbit'].items() if key > 0]
    mean_l_prey,interval_l_prey,acc_l_prey = calculate_confidence_interval(av_l_prey,conf)
    
    av_l_predator = [value['tot l'] / value['n birth'] for data in tot_results for key,value in data.birth_per_gen_per_species['fox'].items() if key > 0]
    mean_l_predator,interval_l_predator,acc_l_predator = calculate_confidence_interval(av_l_predator,conf)
    
    print(f'{acc_lf_prey = } - {acc_lf_predator = }')
    print(f'{acc_l_prey = } - {acc_l_predator = }')
    print(f'{acc_exctint_time_prey = } - {acc_exctint_time_predator = } - {len(tot_results) = }')
    
    results = None
    if acc_lf_prey > acc_threshold and acc_lf_predator > acc_threshold and \
        acc_l_prey > acc_threshold and acc_l_predator > acc_threshold and \
        (acc_exctint_time_prey > acc_threshold or acc_exctint_time_predator > acc_threshold or len(tot_results) > 100):
        results = {
            'prob_improve_prey':param[1],
            'impr_factor_prey':param[2],
            'prob_improve_predator':param[3],
            'impr_factor_predator':param[4],
            'mean_exctint_time_prey':mean_exctint_time_prey,
            'interval_exctint_time_prey_low':interval_exctint_time_prey[0] if interval_exctint_time_prey else None,
            'interval_exctint_time_prey_up':interval_exctint_time_prey[1] if interval_exctint_time_prey else None,
            'acc_exctint_time_prey':acc_exctint_time_prey,
            'exctinct_prob_prey': sum(1 for data in tot_results if data.species_exctinct_time['rabbit'] is not None) / len(tot_results), # if it's none then it not exctinct
            'mean_exctint_time_predator':mean_exctint_time_predator,
            'interval_exctint_time_predator_low':interval_exctint_time_predator[0] if interval_exctint_time_predator else None,
            'interval_exctint_time_predator_up':interval_exctint_time_predator[1] if interval_exctint_time_predator else None,
            'acc_exctint_time_predator':acc_exctint_time_predator,
            'exctinct_prob_prdator': sum(1 for data in tot_results if data.species_exctinct_time['fox'] is not None) / len(tot_results),
            'mean_lf_prey': mean_lf_prey,
            'interval_lf_prey_low':interval_lf_prey[0],
            'interval_lf_prey_up':interval_lf_prey[1],
            'acc_lf_prey':acc_lf_prey,
            'mean_l_prey': mean_l_prey,
            'interval_l_prey_low':interval_l_prey[0],
            'interval_l_prey_up':interval_l_prey[1],
            'acc_lf_predator':acc_lf_predator,
            'mean_lf_predator':mean_lf_predator,
            'interval_lf_predator_low':interval_lf_predator[0],
            'interval_lf_predator_up':interval_lf_predator[1],
            'acc_lf_predator':acc_lf_predator,
            'acc_l_predator':acc_l_predator,
            'mean_l_predator':mean_l_predator,
            'interval_l_predator_low':interval_l_predator[0],
            'interval_l_predator_up':interval_l_predator[1],
            'acc_l_predator':acc_l_predator
        }
    return results

#--------------------------------------------------------------------------------------------------------------------------------------------#
# MAIN METHOD
#
#
if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Input parameters: {vars(args)}')
    
    if args.not_debug:
        folder_path = create_folder_path()
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S-%f")
    else:
        folder_path = None
        current_time = None
    
    random.seed(args.seed)
    
    start_time = datetime.now()
    
    params = [(init_p, prob_improve_prey, impr_factor_prey, prob_improve_predator, impr_factor_predator, 
               (args.percentages,1-args.percentages), move_rate, args.grid_dimentions, args, folder_path) 
                    for init_p in args.init_population
                    for prob_improve_prey,impr_factor_prey in zip(args.prob_improve,args.improve_factor)
                    for prob_improve_predator,impr_factor_predator in zip(args.prob_improve,args.improve_factor)
                    for move_rate in args.move_rate if (prob_improve_predator == 1 and prob_improve_prey == 1) or (prob_improve_predator != 1 and prob_improve_prey != 1)]
    
    acc_results = []
    if args.multiprocessing:
        num_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=num_processes)
        if args.calculate_accuracy:
            print(f'Number of combination to simulate = {len(params)} with {num_processes} processes in parallel\n'
                + f'To reach {args.accuracy_threshold * 100}% accuracy with confidence level = {args.confidence_level * 100:.0f}%')   
        else:
            print(f'Run {len(params)} simulation with {num_processes} process in parallelel without calculating accuracy\n')     
        
        for i,param in enumerate(params):
            acc_result = None
            tot_results = []
            while acc_result is None:
                seeds = [random.randint(0, 100_000) for _ in range(num_processes)]
                with multiprocessing.Pool(processes=num_processes) as p:
                    param_seed = [tuple(param + (seed,)) for seed in seeds]        
                    simulation_results = p.map(simulate_wrapper, param_seed)
                tot_results.extend(simulation_results)
                if not args.calculate_accuracy: break
                acc_result = intervals_wrapper(param, tot_results, args.confidence_level, args.accuracy_threshold)
            if args.calculate_accuracy:
                acc_results.append(pd.DataFrame([acc_result]))
                # plot_results(random.choice(tot_results), param, current_time, folder_path, idx=i)
            else:
                for j,data in enumerate(tot_results):
                    plot_results(data, param, current_time, folder_path, idx=i+j)
        
        pool.close()
        pool.join()
    else:
        if args.verbose:
            loop = params
        else:
            loop = tqdm(params, desc='Simulating in sequence...')
        
        for i,param in enumerate(loop):
            if args.calculate_accuracy:
                tot_results = []
                acc_result = None
                while acc_result is None:
                    simulation_results = simulate_wrapper(param + (random.randint(0, 100_000),))
                    
                    if i % 6 == 0:
                        tot_results.append(simulation_results)
                        acc_result = intervals_wrapper(param=param, tot_results=tot_results, conf=args.confidence_level, acc_threshold=args.accuracy_threshold)
                acc_results.append(pd.DataFrame([acc_result]))
                plot_results(random.choice(tot_results), param, current_time, folder_path, idx=i)
            else:
                simulation_results = simulate_wrapper(param + (random.randint(0, 100_000),))
                plot_results(simulation_results, param, current_time, folder_path, idx=i)
    
    if args.calculate_accuracy:
        acc_results_df = pd.concat(acc_results, ignore_index=True)
        file_name = os.path.join(folder_path, 'results.csv')
        acc_results_df.to_csv(file_name)
        plot_accuracy_results(acc_results_df, current_time=current_time, folder_path=folder_path)
    
    print(f'The simulation took {datetime.now() - start_time}')