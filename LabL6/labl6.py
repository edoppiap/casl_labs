"""
    The goal of Lab L6 is to study properties of  the dynamic processes 
    over graphs.

    Consider a voter model over a G(n,p) with n =10^5 and p_g=10^{-4}.
    According to the initial condition,   each node  has a probability p_1  of being in state +1 with p_1\in {0.55, 0.6, 065, 0.7,  0.8 0.9}.
    Evaluate the probability of reaching  a +1-consensus  (if the graph is not connected consider only the giant component). Evaluate, as well, the time needed to reach consensus.
    
    Then consider a voter model over finite portion of   Z^2 and Z^3.
    Fix p_1=0.6 and, for several values of n \in[10^3, 10^5], estimate, as before,
    the probability of reaching  a +1-consensus  and the time needed to reach consensus.

    Deliver the code along with a brief report, in which you present and comment your results. Are the obtained results in line with  theoretical predictions?

    For the plagiarism checks, please upload your code here: 
    https://www.dropbox.com/request/aoHu1HOIlLVeYlgFbc6U
"""

from queue import PriorityQueue, Queue
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as stats
from multiprocessing import Pool
import os
import time
import argparse
from datetime import datetime

input_parameters = {
    'num_nodes' : [10**(3)],
    'gen_probs' : [10**(-3)],
    'bias_probs' : [.5, .55, .6, .65, .7, .8, .9]
}


#-----------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
#
parser = argparse.ArgumentParser(description='Input parameters for the simulation')

parser.add_argument('--n_sim', type=int, default=6,
                    help='Number of simulation to run for the same type of graph')
parser.add_argument('--factor_of_10', type=int, default=8,
                    help='Difference in factor of 10 between n and p for the generation method')
parser.add_argument('--sim_time', type=int, default=1_000_000, 
                    help='The max time for the simulation')
parser.add_argument('--parallelization', action='store_true',
                    help='To runs the simulations in parallel or not (the reproducibility is compromised)')

SEEDS = [643522, 308619,  90445, 473637, 564870, 910011]

def degree_of(node):
    return len(node['neighbors'])

def qq_plot(g,p):
    n = len(g)
    lam = (n-1)*p
    degrees = np.array([degree_of(g[node]) for node in g])
    
    probplot_data = stats.probplot(degrees, dist=stats.poisson(lam), plot=plt)
    
    plt.title('Q-Q Plot for Poisson Distribution')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid(True)
    plt.show()

def chi_squared_test(g, p):
    n = len(g)
    lam = (n-1)*p
    degrees = np.array([degree_of(g[node]) for node in g])
    #lam = degrees.mean()
    
    x = np.unique(degrees)
    observed, _ = np.histogram(degrees, bins=x, density=True)
    df = len(x) -1
    x = x[:-1]
    expected = stats.poisson.pmf(x, lam)
    
    #expected = stats.binom.pmf(x, n, p)
    
    chi2_stat = np.sum((observed - expected)**2 / expected)
    p_value = 1 - stats.chi2.pdf(chi2_stat, df)
    #chi_2, p_value = stats.chisquare(f_obs=observed, f_exp=expected, ddof=df)
    print(f'chi_stat = {chi2_stat:.4f} - p_value = {p_value}')
    
    plt.figure(figsize=(12,8))
    plt.title('Observed and expected frequencies')
    plt.hist(degrees, bins=np.unique(degrees), alpha=.5, label='Observed', density=True)
    #plt.hist(expected, bins=x, alpha=.5, label='Expected', density=True)
    plt.plot(x, expected, color='r', label='Expected pdf (Poisson)')
    plt.legend()
    plt.show()
    
def BFS(g, start):
    queue = Queue()
    visited = set()
    
    for node in g[start]['neighbors']:
        queue.put(node)
            
    visited.add(start)
        
    while not queue.empty():
        node = queue.get()
        visited.add(node)
        for neighbor in g[node]['neighbors']:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.put(neighbor)
    return visited

def connected_components(g):
    components = []
    visited = set()
    for node in g:
        if node not in visited:
            component = BFS(g, node)
            visited.update(component)
            components.append(component)
    return components

def get_giant_component(g):
    return max(connected_components(g), key=lambda x: len(x))

def generate_z2(n, choices: list, prob):
    dim = int(np.sqrt(n))
    g = {(x,y): {'state': np.random.choice(choices, p=[prob, 1-prob]), 'neighbors': []}\
        for x in range(dim) for y in range(dim)}
    
    for x in range(dim):
        for y in range(dim):
            if x+1 < dim:
                g[(x,y)]['neighbors'].append((x+1,y))
                g[(x+1,y)]['neighbors'].append((x,y))
            if y+1 < dim:
                g[(x,y)]['neighbors'].append((x,y+1))
                g[(x,y+1)]['neighbors'].append((x,y))
    return g

def generate_z3(n, choices:list, prob):
    dim = int(n ** (1/3))
    g ={(x,y,z): {'state': np.random.choice(choices, p=[prob, 1-prob]), 'neighbors': []}\
        for x in range(dim) for y in range (dim) for z in range(dim)}
    
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                if x+1 < dim:
                    g[(x,y,z)]['neighbors'].append((x+1,y,z))
                    g[(x+1,y,z)]['neighbors'].append((x,y,z))
                if y+1 < dim:
                    g[(x,y,z)]['neighbors'].append((x,y+1,z))
                    g[(x,y+1,z)]['neighbors'].append((x,y,z))
                if z+1 < dim:
                    g[(x,y,z)]['neighbors'].append((x,y,z+1))
                    g[(x,y,z+1)]['neighbors'].append((x,y,z))
    return g

def generate_graph_ER(n, p, choices: list, prob):
    g = {node: {'state': np.random.choice(choices, p=[prob, 1-prob]), 'neighbors': []} for node in range(n)}    
    
    # this calculate the differences between num_nodes and p to see if they differ at least of FACTOR_OF_10 factors of ten
    if abs(int(np.log10(p)) - int(np.log10(n))) >= args.factor_of_10: # it means p is sufficiently small
        m = int((p*n*(n-1)) // 2) # expected number of edges
        
        #bar_format='{l_bar}{bar:30}{n:.0f}/{total} edges [{elapsed}<{remaining}, {rate_fmt}]'
        
        for _ in range(m): 
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
        bar_format='{l_bar}{bar:30}{n:.0f}/{total} nodes [{elapsed}<{remaining}, {rate_fmt}]'
        
        for i in range(n): # O(n)
            for j in range(i + 1, n): # O(log(n))
                if random.random() < p:
                    g[i]['neighbors'].append(j)
                    g[j]['neighbors'].append(i)
    
    return g

def plot_graph(all_data, n, p, bias, folder_path):
    
    fig, ax = plt.subplots(2,3)
    consensus = 0
    for i,datas in enumerate(all_data):
        len_giant_component, data = datas
        coordinates = (int(i>2), i % 3)
        consensus_time = None
        for t, plus, _ in data:
            if plus == len_giant_component:
                consensus_time = t
                break
        times = [time for time,_,_ in data]
        plus = [plus for _,plus,_ in data]
        minus = [minus for _,_,minus in data]
        
        ax[coordinates].plot(times, plus, label='+1')
        ax[coordinates].plot(times,minus, label='-1')
        if consensus_time:
            consensus += 1
            ax[coordinates].axvline(x=consensus_time, color='red', linestyle='--', label='Consensus time')
        ax[coordinates].set_xlabel('Time (time unit)')
        ax[coordinates].set_ylabel('State variable occurrences')
        #ax[coordinates].legend()
        ax[coordinates].grid(True)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f'State variable occurrences over time, cons_prob = {consensus_time/6 * 100:.2f}% (n_nodes={n}, neighbors_prob={p}, biased_prob={bias})')
    
    plt.tight_layout()
    file_name = os.path.join(folder_path, f'n_{n}_p_{p}_bias_{bias}.')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()

def simulate(g, max_component):
    FES = PriorityQueue()
    for node in list(g.keys()):
        if node not in max_component:
            g.pop(node)
    lam = len(g) # wake-up process for each node follows poisson(lam = 1) distr , so the whole graph follows poisson(lam = n_nodes)
    #lam = 1
    keys = list(g.keys())
    
    random_index = random.choice(keys)
    time = 0
    
    FES.put((time, random_index))
    
    if not args.parallelization:
        pbar = tqdm(total=args.sim_time,
                        desc=f'Simulating sim_time = {args.sim_time}',
                        bar_format='{l_bar}{bar:30}{n:.0f}s/{total}s [{elapsed}<{remaining}, {rate_fmt}]')
    
    data = []
    plus = sum(g[node]['state'] == 1 for node in max_component)
    minus = sum(g[node]['state'] == -1 for node in max_component)
    
    while time < args.sim_time and not FES.empty():
    
        new_time, current_i = FES.get()
    
        if not args.parallelization:
            if new_time <= args.sim_time: # to prevent a warning to appear
                pbar.update(new_time - time)
            else:
                pbar.update(args.sim_time - time)
                
        time = new_time
        current = g[current_i]
        neighbors = current['neighbors']
        if neighbors:
            random_neighbor = random.choice(neighbors)
        
            previous_state = current['state']
            current['state'] = g[random_neighbor]['state']
            
            if current['state'] != previous_state:
                if current['state'] == 1:
                    plus+=1
                    minus-=1
                else:
                    minus+=1
                    plus-=1
        
        data.append((time, plus, minus))
        
        inter_time = random.expovariate(lam)
        # again, significantly more efficient than
        random_next_node = random.choice(keys)
        #random_next_node = random.randint(0, len(keys)-1)
        
        FES.put((time + inter_time, random_next_node))
                
    return len(max_component), data

def run_simulation(params, seed):
    choices = [-1,1]
    n, p, prob = params
    
    # this ensure that each process will be different from the other
    if args.parallelization:
        seed = (os.getpid() * int(time.time())) % random.randint(0, 1_000_000)
    
    random.seed(seed)
    np.random.seed(seed)
    
    g = generate_graph_ER(n, p, choices, prob)            
    #qq_plot(g,p)
    #chi_squared_test(g, p)
    
    giant_component = get_giant_component(g)
    
    if len(giant_component) == len(g):
        print(f'This graph is connected')
    else:
        print(f'The giant component of this graph has {len(giant_component)} nodes')
    
    len_giant_component, data = simulate(g, giant_component)
    #plot_graph(data, n, p, prob)
    
    return len_giant_component, data


if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Input parameters: {vars(args)}')
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'outputs',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    print(f'Output images will be saved in the folder: {folder_path}')
        
    n_sim = args.n_sim
    parameters = [(n, p, prob) for n,p in zip(input_parameters['num_nodes'],input_parameters['gen_probs']) 
                  for prob in input_parameters['bias_probs']]
    
    # -------------------------------------------------------------------------------------------------------#
    # VOTER MODEL
    #
    #
    all_data = []
    start_time = time.time()
    for param in parameters:
        n,p,bias = param
        if args.parallelization:
            if bias == .5:
                print(f'Generating and simulating in parallel {n_sim} graphs G(n={n},p={p}) and no bias')
            else:
                print(f'Generating and simulating in parallel {n_sim} graphs G(n={n},p={p}) and bias={bias}')
        else:
            if bias == .5:
                print(f'Generating and simulating sequentially {n_sim} graphs G(n={n},p={p}) and no bias')
            else:
                print(f'Generating and simulating sequentially {n_sim} graphs G(n={n},p={p}) and bias={bias}')
            
        if args.parallelization:
            with Pool(n_sim) as pool:
                datas = pool.map(run_simulation, [param]*n_sim)
                #all_data.append(datas)
        else:
            datas = []
            for i in range(n_sim):
                datas.append(run_simulation(param, SEEDS[i]))
        plot_graph(datas, n, p, bias, folder_path)
    
    print(f'The simulation took: {time.time() - start_time:.2f}s ({(time.time() - start_time) / 60:.2f}min)')
            
    print('\n---------------------------------------------------------------------------------------------------------\n')