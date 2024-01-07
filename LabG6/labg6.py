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

import pandas as pd
from tqdm import tqdm
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
parser.add_argument('--prob_improve', '--p_i', type=float, default=[.5,.1], nargs='+',
                    help='Probability of improvement of the lifetime')
parser.add_argument('--init_population', '--p', type=int, default=[5,25], nargs='+',
                    help='Number of individuals for the 1st generation')
parser.add_argument('--improve_factor', '--alpha', type=float, default=[.1], nargs='+',
                    help='Improve factor that an individual can develop at birth')
parser.add_argument('--init_lifetime', type=int, default=[2], nargs='+',
                    help='Lifetime of the 1st generation')
parser.add_argument('--repr_rate', '--lambda', type=float, default=[.1, .01, .001],
                    help='Rate at which an individual reproduces')
parser.add_argument('--max_population', type=int, default=100_000,
                    help='This semplified version need a limit otherwise will infinite grow')

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

class Measure:
    def __init__(self):
        self.num_birth = 0
        self.num_death = 0
        self.average_pop = 0
        self.time_last_event = 0
        self.birth_per_gen = {}
        
    def increment_gen_birth(self, gen):
        self.birth_per_gen[gen]['n birth'] = self.birth_per_gen.setdefault(gen,{'n birth':0, 'tot lf':0})['n birth']+1
        
    def increment_gen_lf(self, gen, lf):
        self.birth_per_gen[gen]['tot lf'] = self.birth_per_gen.setdefault(gen,{'n birth':0, 'tot lf':0})['tot lf']+lf

class Individual():
    def __init__(self, birth_time, alpha, p_i, parent_lf, gen):
        # self.lam = lam -> in this simplified version all individual share the same lambda
        # so there is no need to store a lambda variable inside this class (maybe add in next version)
        self.gen = gen
        self.birth_time = birth_time
        self.lifetime = random.uniform(parent_lf, parent_lf*(1+alpha))\
            if random.random() < p_i else random.uniform(0, parent_lf)
        
    def __str__(self) -> str:
        return f'Birth_time: {self.birth_time:.2f} - Lifetime: {self.lifetime:.2f}'
    
class Event:
    def __init__(self, event_time, event_type, individual: Individual = None):
        self.time = event_time
        self.type = event_type
        self.individual = individual
        
    def __lt__(self, other):
        return self.time < other.time

def gen_init_population(init_p, alpha, init_lifetime):
    return [Individual(0, # born all at the same time
                       alpha=alpha, 
                       p_i=0, # the 1st gen can't improve
                       parent_lf=init_lifetime,
                       gen=0) for _ in range(init_p)]

# we can schedule a new birth based on len(population) after the new_death
def death(current_time, FES: PriorityQueue, lam, population, individual, data: Measure):
    data.average_pop += len(population)*(current_time - data.time_last_event)
    data.time_last_event = current_time
    data.num_death+=1
    population.remove(individual)
    
    if len(population) > 0:
        # schedule a new birth with the new len(population)
        sum_lam = lam*len(population)
        birth_time = current_time + random.expovariate(sum_lam)
        FES.put(Event(birth_time, 'birth'))

# we have to schedule a new death associated to the new_born
# we can schedule a new birth based on len(population) after the new_born
def birth(current_time, FES: PriorityQueue, lam,  alpha, p_i, population, data: Measure):
    data.average_pop += len(population)*(current_time - data.time_last_event)
    data.time_last_event = current_time
    
    # a new individual can born only if there are some in the population
    if len(population) > 0:
        data.num_birth += 1
        
        rand_individual = population[random.randint(0, len(population)-1)]
        new_born = Individual(birth_time=current_time,
                            alpha=alpha,
                            p_i=p_i,
                            parent_lf=rand_individual.lifetime,
                            gen=rand_individual.gen+1 )
        population.append(new_born)
        data.increment_gen_birth(rand_individual.gen)
        data.increment_gen_lf(new_born.gen, new_born.lifetime)
    
        # schedule the death associated with the new_born
        FES.put(Event(current_time + new_born.lifetime, 'death', new_born))
    
    # schedule a new birth event with the new len(population)
    sum_lam = lam*len(population)
    birth_time = current_time + random.expovariate(sum_lam)
    FES.put(Event(birth_time, 'birth'))
    
def simulate(init_p, init_lifetime, alpha, lam, p_i, data: Measure):
    FES = PriorityQueue()
    time = 0
    
    population = gen_init_population(init_p=init_p,
                                          alpha=alpha,
                                          init_lifetime=init_lifetime)
    
    for individual in population:
        data.increment_gen_lf(individual.gen, individual.lifetime)
        FES.put(Event(individual.lifetime, 'death', individual))
    
    # the birth process follows a Poisson distr with lam = sum(lambdas)
    # because each one individual reproduce followint a Poisson distr with lam 
    birth_time = random.expovariate(lam*len(population))
        
    # first event to start the simulation
    first_repr = Event(birth_time, 'birth')
    FES.put(first_repr)
    
    # pbar = tqdm(total=args.sim_time,
    #             desc=f'Simulating natural selection',
    #             postfix=len(population),
    #             bar_format='{l_bar}{bar:30}{n:.0f}s/{total}s [{elapsed}<{remaining}, {rate_fmt}, n indivisuals: {postfix}]')
    
    #----------------------------------------------------------------#
    # EVENT LOOP
    #
    while not FES.empty():
        if len(population) == 0 or len(population) > args.max_population:
            break
        
        event = FES.get()
        
        # pbar.postfix = len(population)
        # if event.time < args.sim_time: # to prevent a warning to appear
        #     pbar.update(event.time - time)
        # else:
        #     pbar.update(args.sim_time - time)
        time = event.time
        
        if event.type == 'birth':
            birth(current_time=time,
                  FES=FES,
                  lam=lam,
                  alpha=alpha,
                  p_i=p_i,
                  population=population,
                  data=data)
        elif event.type == 'death':
            individual = event.individual
            death(current_time=time,
                  FES=FES,
                  lam=lam,
                  population=population,
                  individual=individual,
                  data=data)
        print(f'N individuals: {len(population)} - time: {time:.4f}', end='\r')
        
    return time
    # pbar.close()
    
def plot_gen_birth(gen_stats: dict, folder_path, file_str, init_p):
    
    plt.figure(figsize=(12,8))
    plt.bar(list(gen_stats.keys()), [gen['n birth'] for gen in gen_stats.values()])
    plt.ylabel(f'Total number of children')
    plt.xlabel(f'Generation')
    plt.title('Total number of children for generation')
    plt.grid(True, axis='y')
    file_name = os.path.join(folder_path, 'tot_'+file_str)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    
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
    plt.title('Average lifetime for generation')
    plt.grid(True, axis='y')
    file_name = os.path.join(folder_path, 'lf_'+file_str)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12,8))
    plt.bar(range(len(av_num_child)), av_num_child)
    plt.ylabel(f'Average number of children')
    plt.xlabel(f'Generation')
    plt.title('Average number of children for generation')
    plt.grid(True, axis='y')
    file_name = os.path.join(folder_path, 'av_'+file_str)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

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
                init_lifetime=init_lifetime,
                alpha=alpha,
                lam=lam,
                p_i=p_i,
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
        
        if len(data.birth_per_gen) > 7:
            file_str = f'in_p_{init_p}_in_lt_{init_lifetime}_a_{alpha}_l_{lam}_p_i_{p_i}.'
            plot_gen_birth(data.birth_per_gen, folder_path, file_str, init_p)
    
    result_df = pd.concat(results, ignore_index=True)
    
    file_name = os.path.join(folder_path, 'results.csv')
    result_df.to_csv(file_name)