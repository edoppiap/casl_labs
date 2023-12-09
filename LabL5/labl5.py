"""
    Write a piece of code to simulate dynamical processes on graphs.
    
    The simulator should be able to:
    i)  generate in an efficient way a either G(n,p) graphs or regular  grids  (with order of 100k nodes);
    ii)  handle in  an efficient way the FES (resort on properties of Poisson processes). 

    Deliver the code along with a brief report, in which you clearly describe:
    i)   the data structure you use;
    ii)  which are the events and how the FES is handled.
    ii)  the algorithm according to which you generate samples of  G(n,p) graphs.

    Furthermore for n=100k, p= 10^{-4}  compare the empirical distribution of the degree with analytical predictions.  Build a q-q plot and execute a \chi^2 test.
    
    You find a brief discussion on G(n,p) model  and its properties here: 
    https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model

    You find tables with \chi^2 quantiles here:
    https://en.wikipedia.org/wiki/Chi-squared_distribution
"""

from queue import PriorityQueue
import random
import numpy as np
from tqdm import tqdm

P_THRESHOLD = .01

SIM_TIME = 200_000

def generate_graph_ER(n, p, choices: list):
    g = {node: {'state': random.choice(choices), 'neighbors': []} for node in range(n)}
    
    if p < P_THRESHOLD:
        m = int((p*n*(n-1)) // 2) # expected number of edge
        
        for _ in tqdm(range(m), desc='Generating with ER with p little'): # O(m) -> O(n)
            while True:
                i,j = tuple(random.randint(0, len(g)-1) for _ in range(2)) # pick two random nodes
                if j in g[i]['neighbors']:
                    continue # don't add the edge if already exists
                else:
                    g[i]['neighbors'].append(j)
                    g[j]['neighbors'].append(i)
                    break
    else:    
        for i in tqdm(range(n), desc='Generating with ER'): # O(n)
            for j in range(i + 1, n): # O(log(n))
                if random.random() < p:
                    g[i]['neighbors'].append(j)
                    g[j]['neighbors'].append(i)
    
    return g

def simulate(g):
    FES = PriorityQueue()
    
    random_index = random.randint(0, len(g)-1)
    time = 0
    
    FES.put((time, random_index))
    
    pbar = tqdm(total=SIM_TIME,
                    desc=f'Simulating sim_time = {SIM_TIME}',
                    bar_format='{l_bar}{bar:30}{n:.0f}s/{total}s [{elapsed}<{remaining}, {rate_fmt}]')
    
    while time < SIM_TIME:
        if FES.empty():
            break
        
        new_time, current_i = FES.get()
        
        if new_time <= SIM_TIME: # to prevent a warning to appear
            pbar.update(new_time - time)
        else:
            pbar.update(SIM_TIME - time)
                
        time = new_time
        neighbors = g[current_i]['neighbors']
        if neighbors:
            random_neighbor = random.choice(g[current_i]['neighbors'])
        
            g[current_i]['state'] = g[random_neighbor]['state']
        
        FES.put((time + random.expovariate(1), random.randint(0, len(g)-1)))

# -------------------------------------------------------------------------------------------------------#
# VOTER MODEL
#
#
choices = [-1,1]
g = generate_graph_ER(100_000, 10**(-4), choices)

simulate(g)