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
from scipy.sparse import coo_matrix
import sys
import time

def generate_with_sparse(n,p):
    adj_matrix = coo_matrix((n,n), dtype=np.int8).toarray()
    
    for i in tqdm(range(n), desc='Generating with sparse'):
        for j in range(i+1, n):
            if random.random() < p:
                adj_matrix[i,j] = np.int8(1)
    return adj_matrix

def generate_graph_ER(n, p):
    g = {node: {'state':0, 'neighbors': []} for node in range(n)}
    
    for i in tqdm(range(n), desc='Generating with ER'): # O(n)
        for j in range(i + 1, n): # O(n)
            if random.random() < p:
                g[i]['neighbors'].append(j)
                g[j]['neighbors'].append(i)
    
    return g

def genera_grafo(n, p):
    # Crea una lista vuota per contenere le righe della matrice
    grafo = []
    
    # Itera su ogni riga
    for i in tqdm(range(n), desc='Generando una riga per volta'):
        # Crea una riga vuota
        riga = np.zeros(n, dtype=np.uint8)
        
        # Itera su ogni colonna nella metà superiore della riga
        for j in range(i+1, n):
            # Genera un numero casuale e controlla se è minore di p
            if np.random.rand() < p:
                # Se lo è, aggiungi un bordo nella riga
                riga[j] = 1
                
        # Aggiungi la riga al grafo
        grafo.append(riga)
        
    return np.array(grafo)
            
def generate_sparse(n,p):
    size = (n,n)
    mat = {}
    
    for i in tqdm(range(n), desc='Generating sparce matrix'):
        for j in range(i+1, n):
            if random.random() < p:
                mat[i,j]=int(1)
    
    return mat, size

matrix = generate_with_sparse(10_000, .5)

g = generate_graph_ER(10_000, .5)
matrix, size = generate_sparse(10_000,.5)

"""for i in range(size[0]):
    row = []
    for j in range(size[1]):
        row.append(str(matrix.get((i,j), 0))),
    print(' '.join(row))

start_time = time.time()
g = genera_grafo(10_000,.5)
print(f'Tempo impiegato con il metodo di bing: {time.time() - start_time:.2f}s')"""
            
"""start_time = time.time()

g = genera_grafo(100_000, .5)

print(f'Tempo impiegato generando solo metà matrice: {time.time() - start_time:.2f}s')

start_time = time.time()
            
g = generate_graph_numpy(100_000, .5)

print(f'Tempo impiegato generando riga per riga: {time.time() - start_time:.2f}s')"""

#print(f'Size: {sys.getsizeof(g)/ (1024**3):.2f}GB')

#print('numpy generated')

#g = generate_graph_ER(10_000, .5) # O(n^2)

# g = generate_graph_fast(100_000, .5)
#for _,dict in g.items():   

#print(np.mean([len(dict['neighbors']) for _,dict in g.items()]))
#print(np.max([len(dict['neighbors']) for _,dict in g.items()]))
#print(len([dict['neighbors'] for _,dict in g.items()]))