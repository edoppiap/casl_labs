"""
    Consider a simulator for natural selection with the following simplified simulation model:

    All the individuals belong to S different species
    Let s be the index of each species, s=1...S
    The initial population of s is equal to P(s)
    The reproduction rate for each individual is lambda
    The lifetime LF(k) of individual k whose parent is d(k) is distributed according to the following distribution:
    LF(k)=
    uniform(LF(d(k),LF(d(k)*(1+alpha)) with probability prob_improve   
    uniform(0,LF(d(k)) with probability 1-prob_improve

    where prob_improve  is the probability of improvement for a generation and alpha is the improvement factor (>=0)

    The individuals move randomly in a given region and when individuals of different species meet, they may fight and may not survive.
    Answer to the following questions:

    Describe some interesting questions to address based on the above simulator.
    List the corresponding input metrics.
    List the corresponding output metrics.
    Describe in details the mobility model
    Describe in details the fight model and the survivabilty model
    Develop the simulator
    Define some interesting scenario in terms of input parameters.
    Show and comment some numerical results addressing the above questions.
    Upload only the py code and the report (max 3 pages).
"""

#--------------------------------------------------------------------------------------------------------------------------------------------#
# IMPORTS
#
#
import pandas as pd
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
parser.add_argument('--init_population', '--p', type=int, default=[2000], nargs='+',
                    help='Number of individuals for the 1st generation')
parser.add_argument('--improve_factor', '--alpha', type=float, default=[.5], nargs='+',
                    help='Improve factor that an individual can develop at birth')
parser.add_argument('--init_lifetime', type=int, default=[356 * 3], nargs='+', # 3 years
                    help='Lifetime of the 1st generation')
parser.add_argument('--repr_rate', '--lambda', type=float, default=[1.5],
                    help='Rate at which an individual reproduces')
parser.add_argument('--max_population', type=int, default=5_000,
                    help='This semplified version need a limit otherwise will infinite grow')
parser.add_argument('--grid_dimentions', type=int, default=100, 
                    help='Side of the square of the grid dimension')

# Simulation parameters
parser.add_argument('--sim_time', type=int, default=1_000,
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
species = {
    'wolf': {
        'name':'wolf',
        'init_lifetime':356 * 5, # 5 years
        'av_improv_factor':.2,
        'av_prob_improve':.6,
        'av_repr_rate':1.5,
        'type':'predator'
    },
    'sheep': {
        'name':'sheep',
        'init_lifetime':356 * 7, # 7 years
        'av_improv_factor':.1,
        'av_prob_improve':.3,
        'av_repr_rate':2.5, 
        'type':'prey'
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
        self.time_size_pop = {
            'time':[0],
            'wolf':[400],
            'sheep':[1600]
        }
        self.birth_per_gen = {}
        
    def increment_gen_birth(self, gen):
        self.birth_per_gen[gen]['n birth'] = self.birth_per_gen.setdefault(gen,{'n birth':0, 'tot lf':0})['n birth']+1
        
    def increment_gen_lf(self, gen, lf):
        self.birth_per_gen[gen]['tot lf'] = self.birth_per_gen.setdefault(gen,{'n birth':0, 'tot lf':0})['tot lf']+lf

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
    
    def simulate_fights(self, FES, current_time):
        # iterate over all positions
        for pos in self.values():
            #
            # for each position check if there are more than one individuals
            if len(pos['individuals']) > 1:
                predators = [ind for ind in pos['individuals'] if ind.species['type'] == 'predator']
                preys = [ind for ind in pos['individuals'] if ind.species['type'] == 'prey']
                if len(predators) == 0 or len(preys) == 0:
                    break
                # each predator wants to hunt a prey
                # (it can be also model that more than one predator hunt the same prey,
                #  the winning prob should be higher in that case)
                for predator in predators:
                    if len(preys) == 0: break # there's no more prey
                    prey = random.choice(preys)
                    preys.remove(prey)
                    if random.random() < .5:
                        # predator wins
                        death_event = Event(current_time, 'death', prey)
                        FES.put(death_event)
                    # else:
                    #     # prey wins
                    #     print(f'The sheep is gone away')
                    
class Population(dict):
    def __init__(self, init_p):
        super().__init__(self.gen_init_situation(init_p))
        
    #--------------------------------------------------------------------------------------------------------------------------------------------#
    # GENERATION OF INITIAL POPULATION
    #
    #
    def gen_init_situation(self,init_p):
        species_1 = species['wolf']
        species_2 = species['sheep']
        
        population = {
            'wolf': [Individual(0, parent_lf=species_1['init_lifetime'], gen=0, species=species_1, world_dim=100) for _ in range(int(init_p * .2))],
            'sheep': [Individual(0, parent_lf=species_2['init_lifetime'], gen=0, species=species_2, world_dim=100) for _ in range(int(init_p * .8))]
        }
        
        return population
    
    def __len__(self) -> int:
        n_individuals = 0
        for individuals in self.values():
            n_individuals += len(individuals)
        return n_individuals

#--------------------------------------------------------------------------------------------------------------------------------------------#
# INDIVIDUAL CLASS
#
#
class Individual():
    def __init__(self, birth_time, parent_lf, gen, species: dict, initial_position=None, world_dim=None):
        # self.lam = lam -> in this simplified version all individual share the same lambda
        # so there is no need to store a lambda variable inside this class (maybe add in next version)
        self.gen = gen
        self.birth_time = birth_time
        self.lifetime = random.uniform(parent_lf, parent_lf*(1+species['av_improv_factor']))\
            if random.random() < species['av_prob_improve'] else random.uniform(0, parent_lf)
        assert world_dim is not None or initial_position is not None,'At least one between initial_position and world_dim must be not None'
        self.current_position = (random.randint(0,world_dim-1),random.randint(0,world_dim-1)) if initial_position == None else initial_position
        self.species = species
    
    # at every move the individual can move only in a single dimention
    def move_randomly(self, world: dict):
        old_position = self.current_position
        if self not in world[old_position]['individuals']:
            print(f'{self} non è in {self.current_position} come dovrebbe essere')
            trovato = False
            for position in world.keys():
                if self in world[position]['individuals']:
                    print(f'{self} è in {position}')
                    trovato = True
            if not trovato: print(f'{self} non è più in world')
        self.current_position = random.choice(world[self.current_position]['neighbors'])
        world[old_position]['individuals'].remove(self)
        world[self.current_position]['individuals'].append(self)
        
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
    else:
        print(f'Already dead', end='\r')
    # population.remove(individual)

#--------------------------------------------------------------------------------------------------------------------------------------------#
# BIRTH EVENT
#
#
# we have to schedule a new death associated to the new_born
# we can schedule a new birth based on len(population) after the new_born
def birth(current_time, parent, FES: PriorityQueue, population, data: Measure, world: World):
    data.average_pop += len(population)*(current_time - data.time_last_event)
    data.time_last_event = current_time
    
    # check if the parent is still alive because it can be killed 
    name = parent.species['name']
    pop = population[name]
    if parent is not None and parent in pop:
        data.num_birth += 1
        
        #rand_individual = population[random.randint(0, len(population)-1)]
        new_born = Individual(birth_time=current_time,
                            parent_lf=parent.lifetime,
                            gen=parent.gen+1,
                            species=parent.species,
                            initial_position=parent.current_position)
        world[new_born.current_position]['individuals'].append(new_born)
        population[name].append(new_born)
        data.increment_gen_birth(parent.gen)
        data.increment_gen_lf(new_born.gen, new_born.lifetime)

        death_time = current_time + new_born.lifetime
        # schedule the death associated with the new_born
        FES.put(Event(death_time, 'death', new_born))
        
        # schedule all the birth event relative to this individual
        birth_time = current_time
        while birth_time < death_time:
            # schedule a new birth event with the new len(population)
            birth_time += random.expovariate(new_born.species['av_repr_rate'])
            FES.put(Event(birth_time, 'birth', new_born))

#--------------------------------------------------------------------------------------------------------------------------------------------#
# SIMULATION
#
#
def simulate(init_p, data: Measure):
    FES = PriorityQueue()
    time = 0
    
    population = Population(init_p=init_p)
    world = World(args.grid_dimentions, population)
    
    for individual in [ind for pop in population.values() for ind in pop]:
        data.increment_gen_lf(individual.gen, individual.lifetime)
        FES.put(Event(individual.lifetime, 'death', individual))
    
    # the birth process follows a Poisson distr with lam = sum(lambdas)
    # because each one individual reproduce followint a Poisson distr with lam 
    birth_time = random.expovariate(lam*len(population))
        
    # first event to start the simulation
    for species in population.keys():
        first_parent = random.choice(population[species])
        print(f'Primo nato è un {first_parent.species}')
        first_repr = Event(birth_time, 'birth', individual=first_parent)
        FES.put(first_repr)
    
    #----------------------------------------------------------------#
    # EVENT LOOP
    #
    while not FES.empty():
        if len(population) == 0 or len(population) > args.max_population or time > args.sim_time:
            break
        
        event = FES.get()
        
        time = event.time
        
        print(f'{len(population) = }', end='\r')
        
        if event.type == 'birth':
            birth(current_time=time,
                  parent=event.individual,
                  FES=FES,
                  population=population,
                  data=data,
                  world=world)
        elif event.type == 'death':
            individual = event.individual
            death(current_time=time,
                  population=population,
                  individual=individual,
                  data=data,
                  world=world)
            
        #----------------------------------------------------------------#
        # move randomly every two days
        #
        if int(time) in range(0,args.sim_time,2):
            for ind in [ind for pop in population.values() for ind in pop ]:
                ind.move_randomly(world)
            world.simulate_fights(FES, time)
        
        # if len(population['wolf']) % 5 == 0:
        #data.time_size_pop.append((time,len(population)))
        data.time_size_pop['time'].append(time)
        data.time_size_pop['wolf'].append(len(population['wolf']))
        data.time_size_pop['sheep'].append(len(population['sheep']))
    
    print(f'End {time = :.2f}')
    return time

#--------------------------------------------------------------------------------------------------------------------------------------------#
# PLOT RESULTS
#
#
def plot_num_ind(time_size_pop: dict):
    plt.figure(figsize=(12,8))
    plt.plot(time_size_pop['time'],time_size_pop['wolf'],label='Wolf')
    plt.plot(time_size_pop['time'],time_size_pop['sheep'],label='Sheep')
    plt.xlabel('Time (days)')
    plt.ylabel('Size of population')
    plt.grid()
    plt.legend()
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------#
# PLOT RESULTS
#
#
def plot_gen_birth(time_size_pop: list, gen_stats: dict, init_p, init_lifetime, alpha, lam, p_i, folder_path=None):
    if folder_path is not None:
        folder_str = f'in_p_{init_p}_in_lt_{init_lifetime}_a_{alpha}_l_{lam}_p_i_{p_i}'
        graph_path = os.path.join(folder_path, folder_str)
        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
        
    t = [t for t,_ in time_size_pop]
    n_pop = [pop for _,pop in time_size_pop]
        
    plt.figure(figsize=(12,8))
    plt.plot(t, n_pop, marker='o')
    plt.xlabel('Time (units of time)')
    plt.ylabel('Size of population')
    plt.title(f'Size of the population over time (init_n={init_p}, init_lifetime={init_lifetime}, alpha={alpha}, lam={lam}, p_i={p_i})')
    plt.grid(True)
    # file_name = os.path.join(graph_path, 'pop_time')
    # plt.savefig(file_name, dpi=300, bbox_inches='tight')
    # plt.close()
    plt.show()
    
    plt.figure(figsize=(12,8))
    plt.bar(list(gen_stats.keys()), [gen['n birth'] for gen in gen_stats.values()])
    plt.ylabel(f'Total number of children')
    plt.xlabel(f'Generation')
    plt.title('Total number of children for generation '\
        +f'(init_n={init_p}, init_lifetime={init_lifetime}, alpha={alpha}, lam={lam}, p_i={p_i})')
    plt.grid(True, axis='y')
    # file_name = os.path.join(graph_path, 'tot')
    # plt.savefig(file_name, dpi=300, bbox_inches='tight')
    # plt.close()
    plt.show()
    
    av_lf = []    
    av_num_child = []
    for i in range(len(gen_stats)):
        if i==0:
            av_lf.append(gen_stats[i]['tot lf'] / init_p)
        elif i+1 in gen_stats:
            av_num_child.append(gen_stats[i+1]['n birth'] / gen_stats[i]['n birth'])
            av_lf.append(gen_stats[i]['tot lf'] / gen_stats[i]['n birth'])
        else:
            av_num_child.append(0)
            
    plt.figure(figsize=(12,8))
    plt.bar(range(len(av_lf)), av_lf)
    plt.ylabel(f'Average lifetime')
    plt.xlabel(f'Generation')
    plt.title(f'Average lifetime for generation'\
        +f'(init_n={init_p}, init_lifetime={init_lifetime}, alpha={alpha}, lam={lam}, p_i={p_i})')
    plt.grid(True, axis='y')
    # file_name = os.path.join(graph_path, 'lf')
    # plt.savefig(file_name, dpi=300, bbox_inches='tight')
    # plt.close()
    plt.show()
    
    plt.figure(figsize=(12,8))
    plt.bar(range(len(av_num_child)), av_num_child)
    plt.ylabel(f'Average number of children')
    plt.xlabel(f'Generation')
    plt.title('Average number of children for generation'\
        +f'(init_n={init_p}, init_lifetime={init_lifetime}, alpha={alpha}, lam={lam}, p_i={p_i})')
    plt.grid(True, axis='y')
    # file_name = os.path.join(graph_path, 'av')
    # plt.savefig(file_name, dpi=300, bbox_inches='tight')
    # plt.close()
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
            'gen_num': len(data.birth_per_gen),
            'end_time': end_time
        }
        results.append(pd.DataFrame([result]))
        plot_num_ind(data.time_size_pop)
        
        # if len(data.birth_per_gen) > 7:
        #     plot_gen_birth(data.time_size_pop,data.birth_per_gen, folder_path, init_p, init_lifetime, alpha, lam, p_i)
    
    result_df = pd.concat(results, ignore_index=True)
    
    #file_name = os.path.join(folder_path, 'results.csv')
    #result_df.to_csv(file_name)