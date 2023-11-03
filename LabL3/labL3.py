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
    #
    # also the chi_squared distribution is a special case of gamma variate distribution so maybe has more sense
    # use that distribution (since we already has to have it)
    def gamma_variate(self, shape, rate, size=1_000):
        x = self.exponential(lam = rate, size = size)
        return shape * x 

    #-----------------------------------------------------------------------------------------------------------------------#
    # RAYLEIGH RVs
    #
    # we can dircectly calculate the inverse of the cdf
    def rayleigh(self, sigma, size=1_000):
        y = self.uniform(size=size) # this is the uniformly distributed variable that can represent the y-values
        x = sigma * ((-2 * np.log(1 - y)) ** (1/2)) # this is the inverse of the cumulative for the rayleigh distribution
        
        return x
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # CHI-SQUARED RVs
    #
    # it can be calculated the inverse of the cdf only for ddof = [2,4]
    # for all the other I can simply generate k = ddof normal random variables (mu = 0, sigma_squared = 1)
    # and sum their squared values
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
    def beta(self, alpha, beta, size = 1_000):
        if alpha < 1 or beta < 1:
            raise ValueError("Parameters alpha and beta should be greater than 1")
        
        x = self.gamma_variate(shape = alpha, rate = 1, size=size)
        y = self.gamma_variate(shape = beta, rate= 1, size=size)
        
        return x / (x+y)
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # RICE RVs
    #
    def rice(self, nu, sigma, theta=None, size = 1_000):
        if theta is None:
            theta = 360 * self.uniform(size=1) # it can be a fixed value, I generate a single value between [0°,360°)
            
        x = self.normal(mu=nu*np.cos(theta), sigma_squared=sigma**2)
        y = self.normal(mu=nu*np.sin(theta), sigma_squared=sigma**2)
        
        return (x**2 + y**2) ** .5