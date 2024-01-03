"""
    The goal of Lab L6 is to study properties of  the dynamic processes 
    over graphs.

    Consider a voter model over a G(n,p) with  n chosen in the range [10^3, 10^4]  ( in such a way that simulation last a reasonable time i.e. 5/10 min at most) and p_g= 10/n.
    According to the initial condition,   each node  has a probability p_1  of being in state +1 with p_1\in {0.51, 0.55, 0.6, 0.7}.
    Evaluate the probability of reaching  a +1-consensus  (if the graph is not connected consider only the giant component). Evaluate, as well, the time needed to reach consensus.
    
    Then consider a voter model over finite portion of   Z^2 and Z^3.
    Fix p_1=0.51 and, for 2/3 values of n \in[10^2, 10^4], estimate, as before,
    the probability of reaching  a +1-consensus  and the time needed to reach consensus.

    Deliver the code along with a brief report, in which you present and comment your results. Are the obtained results in line with  theoretical predictions?

    For the plagiarism checks, please upload your code here: 
    https://www.dropbox.com/request/aoHu1HOIlLVeYlgFbc6U
"""

"""
    Duration p_values simulations E-R graph with seed = 42: 
    n = 1000 ~ 2min
    n = 3000 ~ 20min
    n = 4000 ~ 1h
    n = 6000 ~ 1h 50min
"""

from queue import PriorityQueue, Queue
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import time
import argparse
from datetime import datetime
from scipy.stats import t
import pandas as pd

#-----------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
parser = argparse.ArgumentParser(description='Input parameters for the simulation')

parser.add_argument('--n_nodes_er', type=int, default=[1000, 4000], nargs='+',
                    help='Number of nodes to simulate for the ER graph (it can be more than one value)')
parser.add_argument('--n_nodes_z', type=int, default=[100, 300, 500, 700, 900], nargs='+',
                    help='Number of nodes to simulate for the Z^2 and Z^3 graph (it should be more than one value)')
parser.add_argument('--bias_prob', type=float, default=[.51, .55, .6, .7], nargs='+',
                    help='Probabilities p_1 of being in state +1 for each node (it should be more than one)')
parser.add_argument('--types_of_graph', type=str, default=['ER','Z2','Z3'], nargs='+',
                    choices=['ER', 'Z2', 'Z3'], help='Type of graph to simulate')
parser.add_argument('--n_sim', type=int, default=6,
                    help='Minimum number of simulation to run for the same type of graph')
parser.add_argument('--factor_of_10', type=int, default=8,
                    help='Difference in factor of 10 between n and p for the generation method')
parser.add_argument('--accuracy_threshold', type=float, default=.8,
                    help='Accuracy value for which we accept the result')
parser.add_argument('--confidence_level', type=float, default=.8,
                    help='Value of confidence we want for the accuracy calculation')
parser.add_argument('--verbose', action='store_true',
                    help='To see the consensus sum of the nodes')
parser.add_argument('--seed', type=int, default=42, 
                    help='For reproducibility reasons')

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
    
    x = np.unique(degrees)
    observed, _ = np.histogram(degrees, bins=x, density=True)
    df = len(x) -1
    x = x[:-1]
    expected = stats.poisson.pmf(x, lam)
    
    chi2_stat = np.sum((observed - expected)**2 / expected)
    p_value = 1 - stats.chi2.pdf(chi2_stat, df)
    print(f'chi_stat = {chi2_stat:.4f} - p_value = {p_value}')
    
    plt.figure(figsize=(12,8))
    plt.title('Observed and expected frequencies')
    plt.hist(degrees, bins=np.unique(degrees), alpha=.5, label='Observed', density=True)
    #plt.hist(expected, bins=x, alpha=.5, label='Expected', density=True)
    plt.plot(x, expected, color='r', label='Expected pdf (Poisson)')
    plt.legend()
    plt.show()

#-----------------------------------------------------------------------------------------------------------#
#  BREADTH-FIRST SEARCH ALGORITHM
#
#
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

#-----------------------------------------------------------------------------------------------------------#
#  GIANT COMPONENT SEARCH
#
#
def get_giant_component(g):
    max_component = None
    visited = set()
    for node in g:
        if node not in visited:
            component = BFS(g, node)
            visited.update(component)
            if max_component is None or len(component) > len(max_component):
                max_component = component
    return max_component

#-----------------------------------------------------------------------------------------------------------#
#  Z^2 GRAPH GENERATION
#
#
def generate_z2(n, choices: list, prob):
    dim = int(np.ceil(n ** (1/2))) # this ensure that at least n nodes are generated
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

#-----------------------------------------------------------------------------------------------------------#
#  Z^3 GRAPH GENERATION
#
#
def generate_z3(n, choices:list, prob):
    dim = int(np.ceil(n ** (1/3))) # this ensure that at least n nodes are generated
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

#-----------------------------------------------------------------------------------------------------------#
#  ERDOS-RENYI GRAPH GENERATION
#
#
def generate_graph_ER(n, p, choices: list, prob):
    g = {node: {'state': np.random.choice(choices, p=[prob, 1-prob]), 'neighbors': []} for node in range(n)}    
    
    # this calculate the differences between num_nodes and p to see if they differ at least of FACTOR_OF_10 factors of ten
    if abs(int(np.log10(p)) - int(np.log10(n))) >= args.factor_of_10: # it means p is sufficiently small
        m = int((p*n*(n-1)) // 2) # expected number of edges
        
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

#-----------------------------------------------------------------------------------------------------------#
#  PLOT RESULTS
#
#
def plot_graph(results_df, folder_path):
    file_name = os.path.join(folder_path, 'results.csv')
    results_df.to_csv(file_name)
    
    types_of_graph = results_df['type of graph'].unique()
    
    if 'ER' in types_of_graph:
        er_result_df = results_df[results_df['type of graph'] == 'ER']

        plt.figure(figsize=(12,8))
        for n_nodes in er_result_df["n nodes"].unique():
            selected_df = er_result_df[er_result_df['n nodes'] == n_nodes]
            plt.plot(selected_df['bias prob'], 
                    selected_df['consensus time mean'], label = f'N nodes = {n_nodes}')
            plt.errorbar(selected_df['bias prob'], selected_df['consensus time mean'], 
                        yerr=[selected_df['consensus time mean'] - selected_df['time interval low'],
                            selected_df['time interval up'] - selected_df['consensus time mean']],
                        fmt='o', capsize=5, c='black', zorder=1)
        plt.xticks(er_result_df['bias prob'].unique())    
        plt.xlabel('Biases')
        plt.ylabel('Times')
        plt.title(f'Biases vs average time to reach consensus in ER graph')
        plt.grid(True)
        plt.legend()
        file_name = os.path.join(folder_path, f'er_times.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        
        plt.figure(figsize=(12,8))
        for n_nodes in er_result_df["n nodes"].unique():
            selected_df = er_result_df[er_result_df['n nodes'] == n_nodes]
            plt.plot(selected_df['bias prob'], selected_df['consensus prob mean'], label=f'N nodes = {n_nodes}')
            plt.errorbar(selected_df['bias prob'], selected_df['consensus prob mean'], 
                    yerr=[selected_df['consensus prob mean'] - selected_df['consensus interval low'],
                        selected_df['consensus interval up'] - selected_df['consensus prob mean']],
                    fmt='o', capsize=5, c='black', zorder=1)
        plt.xlabel('Bias probability')
        plt.xticks(er_result_df['bias prob'].unique())
        plt.ylabel('+1 probability')
        plt.title(f'Bias prob vs +1 probability in ER graph')
        plt.grid(True)
        plt.legend()
        file_name = os.path.join(folder_path, f'er_consensus.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        
    if 'Z2' in types_of_graph or 'Z3' in types_of_graph:
        z_result_df = results_df[results_df['type of graph'] != 'ER']
        bias_prob = z_result_df["bias prob"].unique()[0]
        
        plt.figure(figsize=(12,8))
        for type_of_graph in z_result_df['type of graph'].unique():
            selected_df = z_result_df[z_result_df['type of graph'] == type_of_graph]
            plt.plot(selected_df['n nodes'], selected_df['consensus time mean'], label=f'Graph {type_of_graph}')
            plt.errorbar(selected_df['n nodes'], selected_df['consensus time mean'], 
                    yerr=[selected_df['consensus time mean'] - selected_df['time interval low'],
                        selected_df['time interval up'] - selected_df['consensus time mean']],
                    fmt='o', capsize=5, c='black', zorder=1)
        plt.xlabel('Number of nodes')
        plt.xticks(z_result_df['n nodes'].unique())
        plt.ylabel('Time to reach consensus (units of time)')
        plt.title(f'Number of node vs average time to reach consensus with bias prob = {bias_prob}')
        plt.grid(True)
        plt.legend()
        file_name = os.path.join(folder_path, f'z_times_bias_{bias_prob}.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        
        plt.figure(figsize=(12,8))
        for type_of_graph in z_result_df['type of graph'].unique():
            selected_df = z_result_df[z_result_df['type of graph'] == type_of_graph]
            plt.plot(selected_df['n nodes'], selected_df['consensus prob mean'], label=f'Graph {type_of_graph}')
            plt.errorbar(selected_df['n nodes'], selected_df['consensus prob mean'], 
                    yerr=[selected_df['consensus prob mean'] - selected_df['consensus interval low'],
                        selected_df['consensus interval up'] - selected_df['consensus prob mean']],
                    fmt='o', capsize=5, c='black', zorder=1)
        plt.xlabel('Number of nodes')
        plt.xticks(z_result_df['n nodes'].unique())
        plt.ylabel('+1 probability')
        plt.title(f'Number of node vs +1 probability with bias prob = {bias_prob}')
        plt.grid(True)
        plt.legend()
        file_name = os.path.join(folder_path, f'z_consensus_bias_{bias_prob}.')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')    

#-----------------------------------------------------------------------------------------------------------#
# SIMULATION ON A SINGLE GRAPH
#
#
def simulate(g, max_component):
    FES = PriorityQueue()
    for node in list(g.keys()):
        if node not in max_component:
            g.pop(node)
    # wake-up process for each node follows poisson(lam = 1) distr ,
    lam = len(g) # so the whole graph follows poisson(lam = n_nodes)
    keys = list(g.keys())
    
    current_i = random.choice(keys)
    sim_time = 0
    
    plus = sum(g[node]['state'] == 1 for node in max_component)
    minus = sum(g[node]['state'] == -1 for node in max_component)
    
    while True:
        
        if args.verbose:
            print(f'Plus: {plus}, minus: {minus}'.ljust(21), end='\r')
                
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
        
        if plus == len(max_component) or minus == len(max_component):
            return sim_time, plus == len(max_component)
        
        sim_time += random.expovariate(lam)
        current_i = random.choice(keys)

# -------------------------------------------------------------------------------------------------------#
# CONDIFENCE INTERVAL METHOD
#
#
def calculate_confidence_interval(data, conf):
    
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

#-----------------------------------------------------------------------------------------------------------#
# FUNCTION THAT PREPARE THE SIMULATION
#
#
def run_simulation(params, type_of_graph):
    choices = [1,-1]
    n, p, prob = params
    n_sim = args.n_sim
    
    print_str = f'Generating and simulating at least {n_sim} graphs {type_of_graph} (n={n}'
    
    if type_of_graph == 'ER': print_str += f',p={p}'
    print_str += f') and bias={bias}'
    print(print_str)
    print('-----------------------')
    
    start_time = time.time()
    
    time_datas, cons_datas = [],[] # store result data for this iteration
    time_acc, cons_acc, i, plus = 0,0,0,0
    # continue to run simulations until a accettable accuracy is reached
    while time_acc < args.accuracy_threshold or cons_acc < args.accuracy_threshold or i < n_sim:
        if type_of_graph == 'ER':
            g = generate_graph_ER(n, p, choices, prob)
        elif type_of_graph == 'Z2':
            g = generate_z2(n, choices, prob)
        else:
            g = generate_z3(n, choices, prob)
            
        giant_component = get_giant_component(g)
        
        if args.verbose:
            if len(giant_component) == len(g):
                print(f'This graph is connected')
            else:
                print(f'The giant component of this graph has {len(giant_component)} nodes')
                              
        if args.verbose: print(f'Graph {i+1}:')
        result = simulate(g, giant_component)
        
        i+=1
        
        if args.verbose:
            t, _ = result
            print(f'This graph reached consensus in {t:.2f} units of time')
            print(f'The simulation took: {time.time() - start_time:.2f}s ({(time.time() - start_time) / 60:.2f}min)')
        
        if result is not None:
            consensus_time, consensus = result
            if consensus:
                plus += 1
            if i > 0: cons_datas.append((plus/i))
            time_datas.append(consensus_time)
        
        # calculate confidence level each n_sim iteration
        if len(time_datas) > 1 and i % n_sim == 0:
            time_mean, time_interval, time_acc = calculate_confidence_interval(time_datas, conf=args.confidence_level)
            if len(cons_datas) > 1:
                cons_mean, cons_interval, cons_acc = calculate_confidence_interval(cons_datas, conf=args.confidence_level)
            if args.verbose: print(f'Accuracy: {time_acc}')
        
        if args.verbose: print('-----------------------')
        
    result = {
        'type of graph': type_of_graph,
        'n nodes': n,
        'edge prob': p,
        'bias prob': bias,
        'consensus time mean': time_mean,
        'time accuracy': time_acc,
        'time interval low': time_interval[0],
        'time interval up': time_interval[1],
        'consensus prob mean': cons_mean,
        'consensus prob accuracy': cons_acc,
        'consensus interval low': cons_interval[0],
        'consensus interval up': cons_interval[1],
        'plus': plus,
        'n runs': i
    }    
    print(f'Number of simulation: {i}')
        
    return result

#-----------------------------------------------------------------------------------------------------------#
# MAIN METHOD
#
#
if __name__ == '__main__':
    
    args = parser.parse_args()
    print(f'Input parameters: {vars(args)}')
    
    # SET THE SEED
    #
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # CREATE OUTPUT FOLDER
    #
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'outputs',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    print(f'Output images will be saved in the folder: {folder_path}')
        
    # FINAL OUTPUT
    #
    final_results = []
    
    # TYPES OF GRAPH TO SIMULATE IN THIS RUN
    #
    types = list(args.types_of_graph)
    
    # -------------------------------------------------------------------------------------------------------#
    # VOTER MODEL ER RANDOM GRAPH
    #
    #
    if 'ER' in types:
        for n in args.n_nodes_er:
            p = 10/n
            for bias in args.bias_prob: # runs simulation for each bias probability input parameters
                param = n,p,bias
                
                result = run_simulation(param, type_of_graph='ER')
                
                # store the results into final_results lst for the final dataframe
                final_results.append(pd.DataFrame([result]))         
        
        types.remove('ER') # remove ER graph from the input list to simulate only the Zs graphs after
        
    # -------------------------------------------------------------------------------------------------------#
    # VOTER MODEL Z^2 and Z^3 GRAPH
    #
    #
    for type_of_graph in types:
        for n in args.n_nodes_z:
            bias = .51
            param = (n,None,bias)
            
            result = run_simulation(param, type_of_graph=type_of_graph)
            final_results.append(pd.DataFrame([result]))
    
    # -------------------------------------------------------------------------------------------------------#
    # STORE, SAVE AND PLOT THE FINAL DATA
    #
    # store the data into a dataframe to save it into a csv file
    final_df = pd.concat(final_results, ignore_index=True)
    
    # save and plot graph for the whole simulation process to produce some analisys
    plot_graph(final_df, folder_path)
            
    print('\n---------------------------------------------------------------------------------------------------------\n')