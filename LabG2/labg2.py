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
    acc = 1 - re # this is the accuracy
    return interval,acc

def simulate(n_sample=100_000):
    Ns = np.arange(100, # start value
                   n_sample, # end value (not included)
                   1000) # jump between values
    Cs = np.arange(.75,1,.03)

    # this matrix will store the results
    # [(n_samples con_level accuracy lower_bound upper_bound)]
    results_mat = np.empty((len(Ns) * len(Cs), 5), dtype=np.float64)
    
    i = 0
    for N in Ns:
        for C in Cs:
            X = np.random.uniform(0, 10, N)
            interval, acc = calculate_confidence_interval(X, C=C)
            results_mat[i] = [N, C, acc, interval[0], interval[1]]
            i += 1
    
    return results_mat
    
def plot_graphs(results_mat, n=None, c=None):
    N = results_mat[:, 0]
    C = results_mat[:, 1]
    accs = results_mat[:, 2]
    lowers = results_mat[:, 3]
    uppers = results_mat[:, 4]
    
    unique_c = np.unique(C)
    
    if n is None:
        n = np.random.choice(N, 1)[0]
    if c is None:
        c = np.random.choice(unique_c)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Confidence level vs. Accuracy
    mask = np.isclose(N,n)
    axes[0, 0].plot(C[mask], accs[mask], linestyle='-', marker='o')
    axes[0, 0].set_title(f'Confidence level vs. Accuracy with number of samples = {n}')
    axes[0, 0].set_xlabel('Confidence level')
    axes[0, 0].set_xticks(np.unique(C))
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True)
    
    # Plot 2: Confidence level vs. Interval for N = n
    axes[0, 1].plot(C[mask], uppers[mask], label='Upper bound', linestyle='-', marker='o')
    axes[0, 1].plot(C[mask], lowers[mask], label='Lower bound', linestyle='-', marker='o')
    axes[0, 1].set_title(f'Confidence level vs Intervals for N = {n}')
    axes[0, 1].set_xlabel('Confidence level')
    axes[0, 1].set_xticks(np.unique(C))
    axes[0, 1].set_ylabel('Intervals')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True)
    
    # Plot 3: Number of samples vs. Accuracy
    mask = np.isclose(C,c)
    axes[1, 0].plot(N[mask], accs[mask], linestyle='-')
    axes[1, 0].set_title(f'Number of samples vs. Accuracy for C = {c:.2}')
    axes[1, 0].set_xlabel('Number of samples')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].grid(True)
    
    # Plot 4: Number of samples vs. Interval for C = c
    axes[1, 1].plot(N[mask], uppers[mask], label='Upper bound')
    axes[1, 1].plot(N[mask], lowers[mask], label='Lower bound')
    axes[1, 1].set_title(f'Number of samples vs Intervals for C = {c:.2}')
    axes[1, 1].set_xlabel('Number of samples')
    axes[1, 1].set_ylabel('Intervals')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("LabG2/output.png", dpi=300, bbox_inches='tight')
    plt.show()


n_sample = 100_000
results_mat = simulate(n_sample)
plot_graphs(results_mat, n=99_100, c=.99)