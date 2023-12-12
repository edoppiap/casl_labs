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
import matplotlib.pyplot as plt
from scipy.special import factorial
import scipy.stats as stats

P_THRESHOLD = .01

SIM_TIME = 1_000_000

random.seed(42)
np.random.seed(42)

def degree_of(node):
    return len(node['neighbors'])

def chi_squared_test(g, p):
    n = len(g)
    lam = (n-1)*p
    degrees = np.array([degree_of(g[node]) for node in g])
    
    x = np.unique(degrees)
    observed, _ = np.histogram(degrees, bins=x, density=True)
    x = x[:-1]
    expected = stats.poisson.pmf(x, lam)
    
    plt.figure(figsize=(12,8))
    plt.title('Observed and expected frequencies')
    plt.hist(degrees, bins=np.unique(degrees), alpha=.5, label='Observed', density=True)
    #plt.hist(expected, bins=x, alpha=.5, label='Expected')
    plt.plot(x,expected, label='Expected (Poisson)')
    plt.legend()
    plt.show()
    
    #expected = stats.binom.pmf(x, n, p)
    
    observed /= np.sum(observed)
    expected /= np.sum(expected)
    
    chi_2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    print(f'p_value = {p_value}')

def generate_graph_ER(n, p, choices: list):
    g = {node: {'state': np.random.choice(choices, p=[.9, .1]), 'neighbors': []} for node in range(n)}
    
    bar_format='{l_bar}{bar:30}{n:.0f}/{total} nodes [{elapsed}<{remaining}, {rate_fmt}]'
    
    if p < P_THRESHOLD:
        m = int((p*n*(n-1)) // 2) # expected number of edges
        
        for _ in tqdm(range(m), desc='Generating with ER with p little', # O(m) -> O(n)
                    bar_format=bar_format): 
            while True:
                i,j = tuple(random.randint(0, len(g)-1) for _ in range(2)) # pick two random nodes
                if i==j or j in g[i]['neighbors']:
                    continue # don't add the edge if already exists or if it's the same node
                else:
                    # add an edge 
                    g[i]['neighbors'].append(j)
                    g[j]['neighbors'].append(i)
                    break
    else:    
        for i in tqdm(range(n), desc='Generating with ER', bar_format=bar_format): # O(n)
            for j in range(i + 1, n): # O(log(n))
                if random.random() < p:
                    g[i]['neighbors'].append(j)
                    g[j]['neighbors'].append(i)
    
    return g

def simulate(g):
    FES = PriorityQueue()
    
    time_batch = 100_000
    next_batch = time_batch
    
    random_index = random.randint(0, len(g)-1)
    time = 0
    
    FES.put((time, random_index))
    
    pbar = tqdm(total=SIM_TIME,
                    desc=f'Simulating sim_time = {SIM_TIME}',
                    bar_format='{l_bar}{bar:30}{n:.0f}s/{total}s [{elapsed}<{remaining}, {rate_fmt}]')
    
    data = []
    
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
            random_neighbor = random.choice(neighbors)
        
            g[current_i]['state'] = g[random_neighbor]['state']
            
        if time > next_batch: # for resource convinience store the results only after a while
            next_batch += time_batch
            data.append((time, sum(g[node]['state'] == 1 for node in g), sum(g[node]['state'] == -1 for node in g)))
        
        inter_time = random.expovariate(1)
        random_next_node = random.randint(0, len(g)-1)  
        FES.put((time + inter_time, random_next_node))
        
    times = [time for time,_,_ in data]
    plus = [plus for _,plus,_ in data]
    minus = [minus for _,_,minus in data]
    plt.figure(figsize=(12,8))
    plt.plot(times, plus, label='+1')
    plt.plot(times,minus, label='-1')
    plt.xlabel('Time (time unit)')
    plt.ylabel('State variable occurrences')
    plt.title(f'State variable occurrences over time')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # -------------------------------------------------------------------------------------------------------#
    # VOTER MODEL
    #
    #
    n = 100_000
    p = 10**(-4)
    choices = [-1,1]
    g = generate_graph_ER(n, p, choices)
    chi_squared_test(g, p)
    
    #simulate(g)
    
    n = 10_000
    p = .001
    #g = generate_graph_ER(n, p, choices)
    #chi_squared_test(g, p)
    
    #g = generate_graph_ER(1_000, .5, choices)    
    