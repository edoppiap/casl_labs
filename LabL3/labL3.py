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
    # Generate a uniformly distributed random variable between low and high
    def uniform(self, size=1_000):
        return np.random.uniform(size=size)

    #-----------------------------------------------------------------------------------------------------------------------#
    # RAYLEIGH RVs
    #
    # with the Rayleigh distribution we can dircectly calculate the inverse of the cdf
    def rayleigh(self, sigma, size=1_000):
        y = self.uniform(size=size) # this is the uniformly distributed variable that can represent the y-values
        x = sigma * ((-2 * np.log(1 - y)) ** (1/2)) # this is the inverse of the cumulative for the rayleigh distribution
        
        return x


    #-----------------------------------------------------------------------------------------------------------------------#
    # LOGNORMAL RVs
    #
    # for generating the lognormal distribution variables between [0,1) it can be followed 
    # the same method explained in the slides for the normal distribution but the variables are optained
    # from the exponential form of 
    def log_normal(self, mu, sigma, size=1_000):
        u = self.uniform(size = int(size / 2))
        v = self.uniform(size = int(size / 2))
        
        B = (-2 * np.log(u)) ** .5 # this is the chi-squared inverse of the cdf with ddof=2
        theta = 2 * np.pi * v
        
        z = np.concatencate((B * np.cos(theta), B * np.sen(theta)), axis=0)
        
        return np.exp(mu + sigma * z)

gen = Generator()
gen.rayleigh(sigma=3, size=1_000)