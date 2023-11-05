import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ksone
import scipy.stats as stats
from scipy.special import erf
import seaborn as sns

"""
From the pdf (the probability density function) we can calculate the integral obtaining the cdf (cumulative
distribution function). One of the proprerty of the cdf is that the y-values are uniformly distribuuted
between 0 and 1. We have this proprerty for a cdf of a random variable since it starts from 0 and approaches 1 
as the x-value increase (so it is growing).

This property is used to calcuate a random variable that follow any desider distribution. The approch is as follow:
- Find the cdf of the desider distribution
- Find the inverse 
- Use a uniformly distributed variable as the y of that inverse to find the x
- Return it
"""

class Generator:

    #-----------------------------------------------------------------------------------------------------------------------#
    # UNIFORM RVs
    #
    # Generate a uniformly distributed random variable between [0,1)
    def uniform(self, size=1_000):
        return np.random.uniform(size=size)
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # EXPONENTIAL RVs
    #
    def exponential(self, lam=1, size=1_000):
        y = self.uniform(size)
        return - np.log(1 - y) / lam # this is the inverse of the cumulative for the exponential distribution
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # GAMMA VARIATE RVs
    #
    # to generate a gamma variate random variable we use the property of this distribution that the sum of k exponential random
    # variable with lambda = 1 / rate (the rate of the desired gamma variate distribution) is a gamma variate random variable
    # we repeat this process size time to optain the desired number of gamma variate random variables
    def gamma_variate(self, shape, rate, size=1_000):
        x = self.exponential(lam =  1/rate, size = size*shape) # array with shape(size*shape,)
        # in this way if we reshape the array to have shape=(size,shape) we can sum the columns to obtain
        # size gamma variate random variables
        return x.reshape(size,shape).sum(axis=1)

    #-----------------------------------------------------------------------------------------------------------------------#
    # RAYLEIGH RVs
    #
    # we can dircectly calculate the inverse of the cdf
    def rayleigh(self, sigma, size=1_000):
        y = self.uniform(size=size) # this is the uniformly distributed variable that can represent the y-values
        return sigma * ((-2 * np.log(1 - y)) ** .5) # this is the inverse of the cumulative for the rayleigh distribution
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # CHI-SQUARED RVs
    #
    # it can be calculated the inverse of the cdf only for ddof = [2]
    # for all the other I can simply generate k = ddof normal random variables (mu = 0, sigma_squared = 1)
    # and sum their squared values
    def chi_squared(self, ddof, size=1_000):
        if ddof < 1:
            raise KeyError("Degree of Freedom should be greather or equal than 1")        
        elif ddof == 1:
            # this is actually useless as I already use this method in the ddof > 2 case
            # but it is here to show that this method works 
            return self.normal(mu=0, sigma=1, size=size) ** 2 
        elif ddof == 2:
            y = self.uniform(size)
            return -2 * np.log(y)
        else:
            # this calculate the chi-squared variables as sum of n normal random variables, with n==ddof
            # as I already have a method that returns me a number of normal random variables, I use it to
            # obtain an array that can be reshaped to have size row of ddof random variables (so shape=(size,ddof))
            # the sum of the columns of each row is equivalent to a single chi-squared random variable
            # and all the columns at the end will be size chi-squared random variables
            
            
            y = self.normal(mu=0, sigma=1, size=ddof*size) ** 2 # create the numpy.array with shape=(size*ddof,)
            y = y.reshape(size, ddof) # reshaping the array to obtain shape=(size,ddof)
            
            # sum the columns and obtain an array with shape=(size,)
            return np.sum(y, axis=1) # in this way I have size number of chi-squared random variables
        
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # NORMAL RVs
    #
    # we can generate normal random variables with the method explained in the slides: generating before 
    # normal random variables between [0,1) and then scaling to obtain the desider distribution
    def normal(self, mu, sigma, size=1_000):
        # calculating random values for the polar coordinates
        B = (self.chi_squared(ddof=2, size = int(size/2))) ** .5 # the radius
        theta = 2 * np.pi * self.uniform(size = int(size / 2)) # the angle
        
        z = np.concatenate((B * np.cos(theta), B * np.sin(theta)), axis=0) # concatenating all the obtained variables
        return mu + sigma * z # scaling to obtain the desider distribution

    #-----------------------------------------------------------------------------------------------------------------------#
    # LOGNORMAL RVs
    #
    # for generating the lognormal random variables we can generate normal random variables and exponentiate them
    def log_normal(self, mu, sigma, size=1_000):
        return np.exp(self.normal(mu, sigma, size))
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # BETA RVs
    #
    # The beta distribution doesn't have a simple closed-form expression for the inverse cdf
    # we can generate random variables when alpha > 1 and beta > 1
    #
    # The method for generate beta random variables is through 2 independent gamma variate random variables
    # there is a formula to follow
    def beta(self, alpha, beta, size = 1_000):
        if alpha < 1 or beta < 1:
            raise ValueError("Parameters alpha and beta should be greater than 1")
        
        x = self.gamma_variate(shape = alpha, rate = 1, size=size)
        y = self.gamma_variate(shape = beta, rate= 1, size=size)
        
        return x / (x+y)
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # RICE RVs
    #
    # For generating rice random variables we can use two independent normal random variables 
    def rice(self, nu, sigma, size = 1_000):
        if nu < 0 or sigma < 0:
            raise ValueError('Parameters nu and sigma should be greater or equal than 0')
            
        x = self.normal(mu=0, sigma=1, size=size)
        y = self.normal(mu=0, sigma=1, size=size)
        
        r = (x**2 + y**2) ** .5
        return sigma * r + nu
    
class FitAssessment:
    def __init__(self, sign_level=.05):
        self.sign_level = sign_level # is the confidence level in which it can be accept the data as following a given distribution

    def external_ks_test(self, data, distr_name, *params):
        data_sorted = np.sort(data)
        
        if distr_name == 'Rayleigh':
            (sigma,) = params
            ks_statistic, p_value = stats.kstest(data, 'rayleigh', args=(0, sigma))  # Use 'rayleigh' distribution here
        elif distr_name == 'Chi-Squared':
            (k,) = params
            ks_statistic, p_value = stats.kstest(data, 'chi2', args=(k,))  # Use 'chi2' distribution here
        elif distr_name == 'LogNormal':
            mu, sigma = params
            ks_statistic, p_value = stats.kstest(data, 'lognorm', args=(sigma, 0, np.exp(mu)))  # Use 'lognorm' distribution here
        elif distr_name == 'Beta':
            alpha, beta = params
            ks_statistic, p_value = stats.kstest(data, 'beta', args=(alpha, beta))  # Use 'beta' distribution here
        elif distr_name == 'Rice':
            nu, sigma = params
            ks_statistic, p_value = stats.kstest(data, 'rice', args=(nu, sigma))
        elif distr_name == 'GammaVariate':
            shape, rate = params  # Shape and rate parameters for gamma variate
            ks_statistic, p_value = stats.kstest(data, 'gamma', args=(shape, 0, 1 / rate))  # Use 'gamma' distribution here
        elif distr_name == 'Exponential':
            lam, = params  # Rate parameter for exponential
            ks_statistic, p_value = stats.kstest(data, 'expon', args=(0, 1 / lam))  # Use 'expon' distribution here

        # Set your desired significance level (alpha)
        alpha = 0.05

        # Check the p-value against alpha to determine whether to reject the null hypothesis
        if p_value < alpha:
            print(f"Reject the null hypothesis: Data does not follow the {distr_name} distribution.")
        else:
            print(f"Fail to reject the null hypothesis: Data follows the {distr_name} distribution.")
    
    
    def ks_test(self, 
                data, # is the data that should be evaluate
                distr_name, # is the name of the distribution
                *params # here there are the parameters that should describe the distribution of the data
                ):
        data_sorted = np.sort(data)
        
        # ECDF function for the K-S test
        n = len(data_sorted)
        ecdf_values = np.arange(1, n + 1) / n
            
        # CDF values based on the dsitribution we are interested in
        cdf_values = None
        
        if distr_name == 'Rayleigh':
            (sigma,) = params
            cdf_values = 1 - np.exp((-data_sorted**2) / (2*sigma**2))
        elif distr_name == 'Chi-Squared':
            (k,) = params
            cdf_values = stats.chi2.cdf(data_sorted, k)
        elif distr_name == 'LogNormal':
            mu, sigma_squared = params
            cdf_values = .5 * (1 + erf((np.log(data_sorted) - mu) / (np.sqrt(2) * sigma_squared)))
        elif distr_name == 'Beta':
            alpha, beta = params
            cdf_values = stats.beta.cdf(data_sorted, alpha, beta)
        elif distr_name == 'Rice':
            nu, sigma = params
            cdf_values = stats.rice.cdf(data_sorted, nu, sigma)
            #cdf_values = 1 - np.exp(-((data_sorted - nu)**2) / (2 * sigma**2))
        elif distr_name == 'GammaVariate':
            shape, rate = params  # Shape and rate parameters for gamma variate
            cdf_values = stats.gamma.cdf(data_sorted, shape, scale=1/rate)
        elif distr_name == 'Exponential':
            lam, = params  # Rate parameter for exponential
            cdf_values = 1 - np.exp(-lam * data_sorted)
        
        # Max distance between the values and the expected values
        D = np.abs(ecdf_values - cdf_values)
        Dmax = D.max()
        #print(Dmax) # this is the max distance value, now i got to assest if this is inside the critical value
        
        # Critical Value
        # Critical Value for the test
        critical_value = ksone.ppf(1 - self.sign_level / 2, n)
        
        # Compare D with the critical value
        if Dmax <= critical_value:
            print(f"D ({Dmax:.4f}) <= Critical Value ({critical_value:.4f}): Fail to reject the null hypothesis -> data follow the {distr_name} distribution")
        else:
            print(f"D ({Dmax:.4f}) > Critical Value ({critical_value:.4f}): Reject the null hypothesis -> data do not follow the {distr_name} distribution")
         
        
        # Plot the ECDF and CDF
        plt.figure(figsize=(8, 6))
        plt.step(data_sorted, ecdf_values, label='ECDF')
        plt.step(data_sorted, cdf_values, label=f'{distr_name} CDF')
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.title(f'Kolmogorov-Smirnov Test for {distr_name} Distribution')
        plt.show()
        
        # p-value
        #TODO
    
if __name__ == '__main__':
    
    gen = Generator()
    fit = FitAssessment(.01)
    size = 1_000
    """
    #-----------------------------------------------------------------------------------------------------------------------#
    # RAYLEIGH EVALUATION
    #
    sigma = 3
    print(f'\n============================================================\n')
    print(f'Evaluating the Rayleigh random variables:')
    data = gen.rayleigh(sigma,size)    
    fit.ks_test(data, 'Rayleigh', sigma)
    fit.external_ks_test(data, 'Rayleigh', sigma)
    print(f'\n============================================================\n')
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # CHI-SQUARED EVALUATION
    #
    k = 10
    data = gen.chi_squared(ddof=k, size=size)
    print(f'Evaluating the Chi-Squared random variables:')
    fit.ks_test(data, 'Chi-Squared', k)
    fit.external_ks_test(data, 'Chi-Squared', k)
    print(f'\n============================================================\n')
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # LOGNORMAL EVALUATION
    #
    mu = 1
    sigma = 3
    data = gen.log_normal(mu, sigma, size)
    print(f'Evaluating the LogNormal random variables:')
    fit.ks_test(data, 'LogNormal', mu, sigma)
    fit.external_ks_test(data, 'LogNormal', mu, sigma)
    print(f'\n============================================================\n')
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # EXPONENTIAL EVALUATION
    #
    data = gen.exponential(lam=1,size=size)
    print(f'Evaluating the Exponential random variables:')
    fit.ks_test(data, 'Exponential', 1)
    fit.external_ks_test(data, 'Exponential', 1)
    print(f'\n============================================================\n')
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # GAMMA VARIATE EVALUATION
    #
    data = gen.gamma_variate(shape = 5, rate = 1, size=size)
    print(f'Evaluating the Gamma Variate random variables:')
    fit.ks_test(data, 'GammaVariate', 5, 1)
    fit.external_ks_test(data, 'GammaVariate', 5, 1)
    print(f'\n============================================================\n')
    
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # BETA EVALUATION
    #
    alpha = 5
    beta = 10
    data = gen.beta(alpha,beta,size)
    print(f'Evaluating the Beta random variables:')
    fit.ks_test(data, 'Beta', alpha, beta)
    fit.external_ks_test(data, 'Beta', alpha, beta)
    print(f'\n============================================================\n')
    """
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # RICE EVALUATION
    #
    nu = 2
    sigma = 2
    data = gen.rice(nu,sigma,size=size)
    print(f'Evaluating the Rice random variables:')
    fit.ks_test(data, 'Rice', nu, sigma)
    fit.external_ks_test(data, 'Rice', nu, sigma)