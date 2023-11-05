import numpy as np
import matplotlib.pyplot as plt
import numpy as np

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
    # This distribution has an important property that can be used to obtain gamma variate random variables 
    # from a uniform distibution. The scaling property state that if we have a gamma variate random variable X
    # then for every c > 0 also c * X is a gamma variate random variable.
    # Knowing that the exponential distribution is a special case of the gamma distribution with k = 1 we can 
    # generate exponential random variables (using the uniform) and then scale the obtained variables to obtain
    # the gamma variate ones
    def gamma_variate(self, shape, rate, size=1_000):
        x = self.exponential(lam = rate, size = size)
        return shape * x 

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
    # or using the gamma variate distribution and its relation with the chi_squared
    def chi_squared(self, ddof, size=1_000):
        if ddof < 1:
            raise KeyError("Degree of Freedom should be greather or equal than 1")        
        elif ddof == 1:
            return self.normal(mu=0, sigma_squared=1, size=size) ** 2 
        elif ddof == 2:
            y = self.uniform(size)
            return (-2 * np.log(y)) ** .5
        else:
            y = self.gamma_variate(shape = (ddof / 2), rate = 2, size=size) # this is actually applicable to all the ddof
        
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # NORMAL RVs
    #
    # we can generate normal random variables with the method explained in the slides: generating before 
    # normal random variables between [0,1) and then scaling to obtain the desider distribution
    def normal(self, mu, sigma_squared, size=1_000):
        # calculating random values for the polar coordinates
        B = self.chi_squared(ddof=2, size = int(size/2)) # the radius
        theta = 2 * np.pi * self.uniform(size = int(size / 2)) # the angle
        
        z = np.concatenate((B * np.cos(theta), B * np.sin(theta)), axis=0) # concatenating all the obtained variables
        return mu + (sigma_squared ** .5) * z # scaling to obtain the desider distribution

    #-----------------------------------------------------------------------------------------------------------------------#
    # LOGNORMAL RVs
    #
    # for generating the lognormal random variables we can generate normal random variables and exponentiate them
    def log_normal(self, mu, sigma_squared, size=1_000):
        return np.exp(self.normal(mu, sigma_squared, size))
    
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
    # For generating rice random variables we can use two independent normal random variables and a costant 
    # value that can have any value(and represent an angle) thanks to the relation that the rice 
    # distribution has with the normal
    def rice(self, nu, sigma, theta=None, size = 1_000):
        if theta is None:
            theta = 360 * self.uniform(size=1) # it can be a fixed value, I generate a single value between [0°,360°)
        if nu < 0 or sigma < 0:
            raise ValueError('Parameters nu and sigma should be greater or equal than 0')
            
        x = self.normal(mu=nu*np.cos(theta), sigma_squared=sigma**2, size=size)
        y = self.normal(mu=nu*np.sin(theta), sigma_squared=sigma**2, size=size)
        
        return (x**2 + y**2) ** .5
    
class FitAssessment:
    def ks_test(self, data, distr_name=None, *params):
        data_sorted = np.sort(data)
        
        # ECDF function for the K-S test
        n = len(data_sorted)
        ecdf_values = np.zeros(n)
        
        for i in range(n):
            ecdf_values[i] = np.sum(data_sorted <= data_sorted[i]) / n
            
        # CDF values based on the dsitribution we are interested in
        cdf_values = np.zeros(n)
        
        if distr_name == 'rayleigh':            
            for i in range (n):
                cdf_values[i] = 1 - np.exp((-data_sorted[i]**2) / (2*params[0]) ) # this is the cdf of the rayleigh
        
        # Max distance between the values and the expected values
        D = np.abs(ecdf_values - cdf_values)
        Dmax = D.max()
        print(Dmax) # this is the max distance value, now i got to assest if this is inside the critical value
        
        # Critical Value
        # TODO
        
        # p-value
        #TODO
    
if __name__ == '__main__':
    gen = Generator()

    #-----------------------------------------------------------------------------------------------------------------------#
    # RAYLEIGH EVALUATION
    #
    sigma = 3
    size = 1_000
    data = gen.rayleigh(sigma,size)
    mean_pred = data.mean()
    mean_truth = sigma * ((np.pi / 2) ** .5)

    print(f'Prediction mean: {mean_pred}\nAnalytical mean: {mean_truth}')
    
    fit = FitAssessment()
    
    fit.ks_test(data, 'rayleigh', sigma)