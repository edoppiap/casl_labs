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

#--------------------------------------------------------------------------------------------------------------------------------------------#
# IMPORTS
#
#
import pandas as pd
import math
import random
from queue import PriorityQueue
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

#--------------------------------------------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
parser = argparse.ArgumentParser(description='Input parameters for the simulation')

# Population parameters
parser.add_argument('--prob_improve', '--p_i', type=float, default=[.8], nargs='+',
                    help='Probability of improvement of the lifetime')
parser.add_argument('--init_population', '--p', type=int, default=[20_000], nargs='+',
                    help='Number of individuals for the 1st generation')
parser.add_argument('--improve_factor', '--alpha', type=float, default=[.5], nargs='+',
                    help='Improve factor that an individual can develop at birth')
parser.add_argument('--init_lifetime', type=int, default=[356 * 3], nargs='+', # 3 years
                    help='Lifetime of the 1st generation')
parser.add_argument('--repr_rate', '--lambda', type=float, default=[1.5],
                    help='Rate at which an individual reproduces')
parser.add_argument('--max_population', type=int, default=50_000,
                    help='This semplified version need a limit otherwise will infinite grow')
parser.add_argument('--grid_dimentions', type=int, default=35, 
                    help='Side of the square of the grid dimension')

# Simulation parameters
parser.add_argument('--sim_time', type=int, default=356 * 25, # 25 years
                    help='Time to run the simulation')
parser.add_argument('--accuracy_threshold', type=float, default=.8,
                    help='Accuracy value for which we accept the result')
parser.add_argument('--confidence_level', type=float, default=.8,
                    help='Value of confidence we want for the accuracy calculation')
parser.add_argument('--verbose', action='store_true',
                    help='To see the progress of the simulation')
parser.add_argument('--seed', type=int, default=42, 
                    help='For reproducibility reasons')

#--------------------------------------------------------------------------------------------------------------------------------------------#
# SPECIES CLASS
#
#
SPECIES = {
    'wolf': {
        'name':'wolf',
        'init_lifetime':356 * 15, # 15 years
        'av_improv_factor':.2,
        'av_prob_improve':.6,
        'av_repr_rate': 1 / 365, # once a year on average
        'type':'predator',
        'max_day_with_no_food': 30,
        'attack_on_group': True
    },
    'sheep': {
        'name':'sheep',
        'init_lifetime':356 * 10, # 10 years
        'av_improv_factor':.1,
        'av_prob_improve':.3,
        'av_repr_rate': 1 / 365, # once a year on average 
        'type':'prey',
        'max_day_with_no_food': None,
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
        self.time_size_pop = {}
        self.birth_per_species = {}
        self.birth_per_gen_per_species = {}
        self.num_win = []
        
    def increment_gen_birth(self, gen, species):
        if species['name'] not in self.birth_per_gen_per_species:
            self.birth_per_gen_per_species[species['name']] = {}
        self.birth_per_gen_per_species[species['name']][gen]['n birth'] = self.birth_per_gen_per_species[species['name']].setdefault(gen,{'n birth':0, 'tot lf':0})['n birth'] + 1
        
    def increment_gen_lf(self, gen, lf, species):
        if species['name'] not in self.birth_per_gen_per_species:
            self.birth_per_gen_per_species[species['name']] = {}
        self.birth_per_gen_per_species[species['name']][gen]['tot lf'] = self.birth_per_gen_per_species[species['name']].setdefault(gen,{'n birth':0, 'tot lf':0})['tot lf'] + lf

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
    
    #----------------------------------------------------------------#
    # MOVE RANDOMLY ALL THE INDIVIDUALS
    #
    # an individual can move only in a single dimension
    def move_randomly(self):
        for pos in self.values():
            for individual in pos['individuals']:
                new_position = random.choice(pos['neighbors'])
                pos['individuals'].remove(individual)
                self[new_position]['individuals'].append(individual)
                individual.current_position = new_position
                
    def update_predator_food(self, current_time, FES: PriorityQueue):
        for pos in self.values():
            if len(pos['individuals']) > 1 and 'predator' in [ind.species['type'] for ind in pos['individuals']]:
                for individual in [ind for ind in pos['individuals'] if ind.species['type'] == 'predator']:
                    individual.day_with_no_food += 2
                    if individual.day_with_no_food > individual.species['max_day_with_no_food']:
                        death_event = Event(current_time, 'death', individual)
                        FES.put(death_event)
    
    def simulate_fights(self, FES, current_time, data):
        # iterate over all positions
        for pos in self.values():
            #
            # for each position check if there are more than one individuals
            if len(pos['individuals']) > 1:
                predators = [ind for ind in pos['individuals'] if ind.species['type'] == 'predator']
                preys = [ind for ind in pos['individuals'] if ind.species['type'] == 'prey']
                if len(predators) == 0 or len(preys) == 0:
                    break # there is no change to have a fight TODO: models if two individuals of the same species can fight eachother
                # each predator wants to hunt a prey
                # (it can be also model that more than one predator hunt the same prey,
                #  the winning prob should be higher in that case)
                if False:
                    for predator in predators:
                        if len(preys) == 0: break # there's no more prey
                        prey = random.choice(preys)
                        preys.remove(prey)
                        if random.random() < 1: # TODO: the winning prob should be 1) an input or 2) a parameters given by the inds chars
                            # predator wins
                            death_event = Event(current_time, 'death', prey)
                            predator.day_with_no_food = 0
                            FES.put(death_event)
                            data.num_win.append((1,0))
                        else:
                            data.num_win.append((0,1))
                else:
                    # this function will attack a prey with a group strategy
                    # groups of max n_predator_each_pray will fight the same prey
                    # the more predator are in a group, the higher the chance of win
                    n_predator_each_pray = 3
                    n_groups_of_predators = math.ceil(len(predators) / n_predator_each_pray)                    
                    predators_tuples = [(predators[i*3:(i+1)*3] + [None] * (3 - len(predators[i*3:(i+1)*3]))) for i in range(n_groups_of_predators)]
                    
                    # for each group select a random prey and start a fight
                    for group_of_predators in predators_tuples:
                        if len(preys) == 0: break # there are no more preys left
                        prey_to_attack = random.choice(preys)
                        
                        # this will mean
                        # one predator vs one prey = 40% chance of win
                        # two predator vs one prey = 80% chance of win
                        # three predator vs one prey > 100% chance of win
                        win_prob = sum(.3 for pred in group_of_predators if pred is not None)
                        if random.random() < win_prob:
                            data.num_win.append((1,0))
                            preys.remove(prey_to_attack) # remove the prey here inside means that if the prey wins could potentially fight with other groups in the next iteration
                            FES.put(Event(current_time, 'death', prey_to_attack)) # add the death event into the FES
                            for pred in group_of_predators:
                                if pred is not None:
                                    pred.day_with_no_food = 0 # saving that the predators has eaten
                        else:
                            data.num_win.append((0,1))

#--------------------------------------------------------------------------------------------------------------------------------------------#
# POPULATION CLASS (DICT WITH SOME FUNCTIONS)
#
#      
class Population(dict):
    def __init__(self, init_p, FES):
        super().__init__(self.gen_init_situation(init_p, FES))
        
    #---------------------------------------------------------------------#
    # GENERATION OF INITIAL POPULATION
    #
    #
    def gen_init_situation(self,init_p, FES: PriorityQueue):
        
        init_ps = [int(init_p * .2), int(init_p * .8)] # 20% wolf - 80% sheep
        
        for specie,n_p in zip(SPECIES, init_ps):
            for _ in range(n_p): FES.put(Event(0, 'birth', specie))
        
        return {specie: [] for specie in SPECIES}
    
    def __len__(self) -> int:
        return sum(len(individuals) for individuals in self.values())

#--------------------------------------------------------------------------------------------------------------------------------------------#
# INDIVIDUAL CLASS
#
#
class Individual():
    def __init__(self, birth_time, parent_lf, gen, species: dict, initial_position=None, world_dim=None):
        self.sex = 'Y' if random.random() < .5 else 'X'
        self.gen = gen
        self.birth_time = birth_time
        self.lifetime = random.uniform(parent_lf, parent_lf*(1+species['av_improv_factor']))\
            if random.random() < species['av_prob_improve'] else random.uniform(0, parent_lf)
        assert world_dim is not None or initial_position is not None,'At least one between initial_position and world_dim must be not None'
        self.current_position = (random.randint(0,world_dim-1),random.randint(0,world_dim-1)) if initial_position == None else initial_position
        
        self.species = species
        if species['type'] == 'predator':
            self.day_with_no_food = 0
        
        self.repr_rate = species['av_repr_rate']
        self.prob_improve = species['av_prob_improve']
        self.improv_factor = species['av_improv_factor']
        # TODO: make repr_rate, prob_improve and improv_factor random for each individual
        
    def __str__(self) -> str:
        return f'Birth_time: {self.birth_time:.2f} - Lifetime: {self.lifetime:.2f}'

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
def death(current_time, population: dict, individual, data: Measure, world: World):
    if individual in population[individual.species['name']]:
        data.average_pop += len(population)*(current_time - data.time_last_event)
        data.time_last_event = current_time
        data.num_death+=1
        population[individual.species['name']].remove(individual)
        world[individual.current_position]['individuals'].remove(individual)

#--------------------------------------------------------------------------------------------------------------------------------------------#
# BIRTH EVENT
#
#
# we have to schedule a new death associated to the new_born
# we can schedule a new birth based on len(population) after the new_born
def birth(current_time, new_born, parent, FES: PriorityQueue, population, data: Measure, world: World):
    data.average_pop += len(population)*(current_time - data.time_last_event)
    data.time_last_event = current_time
    
    # check if the parent is still alive because it can be killed 
    specie = new_born.species['name']
    pop = population[specie]
    if parent == 'first' or (parent is not None and parent in pop):
        data.birth_per_species[specie] = data.birth_per_species.setdefault(specie, 0) + 1
        data.num_birth += 1
        
        #rand_individual = population[random.randint(0, len(population)-1)]
        world[new_born.current_position]['individuals'].append(new_born)
        population[specie].append(new_born)
        # data.increment_gen_birth(parent.gen)
        # data.increment_gen_lf(new_born.gen, new_born.lifetime)
        data.increment_gen_birth(new_born.gen, species=new_born.species)
        data.increment_gen_lf(new_born.gen, new_born.lifetime, species=new_born.species)

        death_time = current_time + new_born.lifetime
        # schedule the death associated with the new_born
        FES.put(Event(death_time, 'death', new_born))
        
        # schedule all the birth event relative to this individual
        birth_time = current_time
        while True:
            # schedule a new birth event with the new len(population)
            birth_time += random.expovariate(new_born.repr_rate)
            if birth_time > death_time:
                break
            FES.put(Event(birth_time, 'birth', new_born))

#--------------------------------------------------------------------------------------------------------------------------------------------#
# SIMULATION
#
#
def simulate(init_p, data: Measure):
    FES = PriorityQueue()
    t = 0
    
    population = Population(init_p=init_p, FES=FES)
    world = World(args.grid_dimentions, population)
    
    # for individual in [ind for pop in population.values() for ind in pop]:
    #     # data.increment_gen_lf(individual.gen, individual.lifetime)
    #     FES.put(Event(individual.lifetime, 'death', individual))
    
    # the birth process follows a Poisson distr with lam = sum(lambdas)
    # because each one individual reproduce followint a Poisson distr with lam 
    # birth_time = random.expovariate(lam*len(population))
        
    # first event to start the simulation
    # for species in population.keys():
    #     first_parent = random.choice(population[species])
    #     FES.put(Event(birth_time, 'birth', individual=first_parent))
        
    previous_day = 0
    
    start_time = datetime.now()
    
    #----------------------------------------------------------------#
    # EVENT LOOP
    #
    while not FES.empty():
        
        event = FES.get()
        
        t = event.time
        
        print(f'{len(population) = } - {t = :.2f}     ', end='\r')
        
        if event.type == 'birth':
            if event.individual in SPECIES: # it means it's the first birth events
                new_born = Individual(birth_time=t,
                                parent_lf=SPECIES[event.individual]['init_lifetime'],
                                gen=0,
                                species=SPECIES[event.individual],
                                world_dim=args.grid_dimentions)
                parent = 'first'
            else: # it means it a normal birth event
                parent = event.individual
                new_born = Individual(birth_time=t,
                                      parent_lf=parent.lifetime,
                                      gen=parent.gen+1,
                                      initial_position=parent.current_position,
                                      species=parent.species)
            birth(current_time=t,
                  new_born=new_born,
                  parent=parent,
                  FES=FES,
                  population=population,
                  data=data,
                  world=world)
        elif event.type == 'death':
            individual = event.individual
            death(current_time=t,
                  population=population,
                  individual=individual,
                  data=data,
                  world=world)
            
        #----------------------------------------------------------------#
        # move randomly every day
        #
        current_day = int(t)
        if current_day - previous_day >= 1:
            previous_day = current_day
            world.update_predator_food(t, FES)
            world.move_randomly()
            world.simulate_fights(FES, t, data)
        
        # STORE THE LEN OF THE POPULATION PER SPECIES
        if t > 0:
            data.time_size_pop.setdefault('time',[]).append(t)
            for specie in population.keys():
                data.time_size_pop.setdefault(specie,[]).append(len(population[specie]))
        
        if len(population) == 0 or len(population) > args.max_population or t > args.sim_time:
            break
    
    print(f'The simulation took {datetime.now() - start_time}')
    print(f'{len(population) = }')
    print(f'{len(population['wolf']) = }\n{len(population['sheep']) = }')
    for species in population:
        print(f'{species} birth events: {data.birth_per_species[species]}')
    print(f'Num wolf win: {sum(predator_win for predator_win, _ in data.num_win)}')
    print(f'Num sheep win: {sum(sheep_win for _,sheep_win in data.num_win)}')
    print(f'End {t = :.2f}')
    return t

#--------------------------------------------------------------------------------------------------------------------------------------------#
# PLOT RESULTS
#
#
def plot_results(data: Measure, folder_path = None):
            
    time_size_pop = data.time_size_pop

    if time_size_pop is not None:
        plt.figure(figsize=(12,8))
        plt.plot([t/365 for t in time_size_pop['time']],time_size_pop['wolf'],label='Wolf')
        plt.plot([t/365 for t in time_size_pop['time']],time_size_pop['sheep'],label='Sheep')
        plt.xlabel('Time (years)')
        plt.ylabel('Size of population')
        plt.grid()
        plt.legend()
        if folder_path:
            file_name = os.path.join(folder_path, 'pop_time_species.')
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
        plt.title('Number of birth events per generation')
        plt.legend()
        plt.grid(True)
        if folder_path:
            file_name = os.path.join(folder_path, 'birth_gen_time_species.')
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
        plt.ylabel('Average life expectancy (without considering the death for fight outcomes)')
        plt.title('Life expectancy per generation (years)')
        plt.legend()
        plt.grid(True)
        if folder_path:
            file_name = os.path.join(folder_path, 'birth_gen_time_species.')
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

#--------------------------------------------------------------------------------------------------------------------------------------------#
# MAIN METHOD
#
#
if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Input parameters: {vars(args)}')
    
    #folder_path = create_folder_path()
    
    random.seed(args.seed)
    
    results = []
    
    params = [(init_p, init_lifetime, alpha, lam, p_i) for init_p in args.init_population 
              for init_lifetime in args.init_lifetime 
              for alpha in args.improve_factor
              for lam in args.repr_rate
              for p_i in args.prob_improve]
    
    print(f'params combination: {len(params)}')
    
    for init_p, init_lifetime, alpha, lam, p_i in params:
        
        data = Measure()
        print(f'Simulate with init_p={init_p} - init_lifetime={init_lifetime} - alpha={alpha} - lambda={lam} - p_i={p_i}')
        end_time = simulate(init_p=init_p,
                data=data)
        
        result = {
            'init_p': init_p,
            'init_lifetime':init_lifetime,
            'alpha': alpha,
            'lambda':lam,
            'p_i':p_i,
            'num_birth': data.num_birth,
            'num_death': data.num_death,
            'average_pop': data.average_pop/end_time,
            # 'gen_num': len(data.birth_per_gen),
            'end_time': end_time
        }
        results.append(pd.DataFrame([result]))
        plot_results(data)
        
        # if len(data.birth_per_gen) > 7:
        #     plot_gen_birth(data.time_size_pop,data.birth_per_gen, folder_path, init_p, init_lifetime, alpha, lam, p_i)
    
    result_df = pd.concat(results, ignore_index=True)
    
    #file_name = os.path.join(folder_path, 'results.csv')
    #result_df.to_csv(file_name)