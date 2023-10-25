import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_confidence_interval(X,C):
    """
    This will perform an estimate of the average of the population and return the confidence interval with its accuracy.
    It use the scipy.stats.t.interval function and it returns a coupla with containing (interval,accuracy)

    :param X: sample for which estimate the average and the confidence interval
    :param C: the choosen confidence intervla
    :return: ((lower_bound,upper_bound),float64)
    """
    
    mean = X.mean()
    std = X.std(ddof=1)
    se = std / (len(X)**(1/2)) # this is the standard error
    
    interval = t.interval(confidence = C, # confidence level
                          df = len(X)-1, # degree of freedom
                          loc = mean, # center of the distribution
                          scale = se # spread of the distribution (we use the standard error as we use the extimate of the mean and not the real mean)
                          )
    
    MOE = interval[1] - interval[0] # this is the margin of error
    re = (MOE / (2 * abs(mean))) # this is the relative error
    acc = 1 - re # this is the accuracy
    return interval,acc

def simulate(n_sample=100_000):
    """
    This will create the uniformly distributed samples and start all the estimates, 
    it will store all the results inside a numpy matrix and then return it.

    :param n_sample: number that represent the dimention of the uniformly distributed samples
    :return: numpy.array (shape = (len(Ns) * len(Cs), 5))
    """
    Ns = np.arange(100, # start value
                   n_sample, # end value (not included)
                   1000) # jump between values
    Cs = np.arange(.75,1,.03)

    # this matrix will store the results
    # [(n_samples conf_level accuracy lower_bound upper_bound)]
    results_mat = np.empty((len(Ns) * len(Cs), 5), dtype=np.float64)
    
    i = 0
    for N in Ns:
        X = np.random.uniform(0, 10, N)
        for C in Cs:
            interval, acc = calculate_confidence_interval(X, C=C)
            results_mat[i] = [N, C, acc, interval[0], interval[1]]
            i += 1
    
    return results_mat
    
def plot_graphs(results_mat, n=None, c=None):
    """
    This will create the graph requested for the lab

    :param results_mat: numpy.array (shape = (len(Ns) * len(Cs), 5)) 
    :param n: choosen N to visualize (optional)
    :param c: choosen C to visualize
    """
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
    axes[0, 0].set_title(f'(a) Confidence level vs. Accuracy for N = {n}')
    axes[0, 0].set_xlabel('Confidence level')
    axes[0, 0].set_xticks(np.unique(C))
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].grid(True, linestyle='--')
    
    # Plot 2: Confidence level vs. Interval for N = n
    axes[0, 1].plot(C[mask], uppers[mask], label='Upper bound', linestyle='-', marker='o')
    axes[0, 1].plot(C[mask], lowers[mask], label='Lower bound', linestyle='-', marker='o')
    for i in range(len(C[mask])):
        x = C[mask][i]
        y1 = lowers[mask][i]
        y2 = uppers[mask][i]
        arrow = patches.FancyArrowPatch((x, y1), (x, y2), arrowstyle='<->', alpha=.5, mutation_scale=15, color='black', zorder=10)
        axes[0, 1].add_patch(arrow)
    axes[0, 1].set_title(f'(b) Confidence level vs Intervals for N = {n}')
    axes[0, 1].set_xlabel('Confidence level')
    axes[0, 1].set_xticks(np.unique(C))
    axes[0, 1].set_ylabel('Intervals')
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle='--')
    
    # Plot 3: Number of samples vs. Accuracy
    mask = np.isclose(C,c)
    axes[1, 0].plot(N[mask], accs[mask], linestyle='-')
    axes[1, 0].set_title(f'(c) Number of samples vs. Accuracy for C = {c:.2}')
    axes[1, 0].set_xlabel('Number of samples')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].grid(True, linestyle='--')
    
    # Plot 4: Number of samples vs. Interval for C = c
    axes[1, 1].plot(N[mask], uppers[mask], label='Upper bound')
    axes[1, 1].plot(N[mask], lowers[mask], label='Lower bound')
    axes[1, 1].set_title(f'(d) Number of samples vs Intervals for C = {c:.2}')
    axes[1, 1].set_xlabel('Number of samples')
    axes[1, 1].set_ylabel('Intervals')
    axes[1, 1].legend(loc='upper right')
    axes[1, 1].grid(True, linestyle='--')
    
    plt.tight_layout()
    plt.savefig("LabG2/output.png", dpi=300, bbox_inches='tight')
    plt.show()


n_sample = 100_000
results_mat = simulate(n_sample)
plot_graphs(results_mat, n=99_100, c=.99)