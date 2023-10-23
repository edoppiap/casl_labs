import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt

def calculate_confidence_interval(X,C):
    mean = X.mean()
    std = X.std(ddof=1)
    se = std / (len(X)**(1/2)) # this is the standard error
    
    interval = t.interval(confidence = C, # confidence level
                          df = len(X)-1, # degree of freedom
                          loc = mean, # center of the distribution
                          scale = se # spread of the distribution (we use the standard error as we use the extimate of the mean and not the real mean)
                          )
    
    MOE = interval[1] - interval[0]
    re = (MOE / (2 * abs(mean))) # this is the relative error
    
    return interval,re

def simulate(n_sample=100_000):
    Ns = np.arange(100,n_sample,1000)
    Cs = np.arange(.85,.99,.02)

    results = np.empty((len(Ns) * len(Cs), 5), dtype=np.float64)
    
    i = 0
    for N in Ns:
        for C in Cs:
            X = np.random.uniform(0, 10, N)
            interval, re = calculate_confidence_interval(X, C=C)
            results[i] = [N, C, 1 - re, interval[0], interval[1]]
            i += 1
    
    return results

def plot_graphs(results, n=None, c=None):
    N = results[:,0]
    C = results[:,1]
    accs = results[:,2]
    starts = results[:,3]
    ends = results[:,4]
    
    unique_c = np.unique(C)
    
    if n == None:
        n = np.random.choice(N,1)[0]
    if c == None:
        c = np.random.choice(unique_c)
    
    plt.figure(figsize=(10, 6))
    n = np.random.choice(N,1)[0]
    mask = N == n
    plt.plot(C[mask], accs[mask], linestyle='-', marker='o')
    plt.title(f'Confidence level vs. Accuracy with number of samples = {n}')
    plt.xlabel('Confidence level')
    plt.xticks(np.unique(C))
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.plot(C[mask], starts[mask], label=f'Lower bound')
    plt.plot(C[mask], ends[mask], label=f'Upper bound')
    plt.title(f'Confidence level vs Interval for N = {n}')
    plt.xlabel('Confidence level')
    plt.ylabel('Interval')
    plt.legend(loc='upper right')
    plt.grid(True)
    #plt.savefig("LabG2/output.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10,6))
    mask = C == c
    plt.plot(N[mask], accs[mask], linestyle='-')
    plt.title(f'Number of samples vs. Accuracy for C = {c:.2}')
    plt.xlabel('Number of samples')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.plot(N[mask], starts[mask], label=f'Lower bound')
    plt.plot(N[mask], ends[mask], label=f'Upper bound')
    plt.title(f'Numbers of samples vs Interval for C = {c:.2}')
    plt.xlabel('Number of samples')
    plt.ylabel('Interval')
    plt.legend(loc='upper right')
    plt.grid(True)
    #plt.savefig("LabG2/output.png", dpi=300, bbox_inches='tight')
    plt.show()


n_sample = 100_000
plot_graphs(simulate(n_sample))