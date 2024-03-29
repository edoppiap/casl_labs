import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ksone
import scipy.stats as stats
from scipy.special import erf
from datetime import datetime
import os

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
    # to generate a gamma variate random variable we use the property of this distribution that the sum of k exponential 
    # random variable with lambda = 1 / rate (the rate of the desired gamma variate distribution) is a gamma variate 
    # random variable we repeat this process size time to optain the desired number of gamma variate random variables
    def gamma_variate(self, shape, rate, size=1_000):
        x = self.exponential(lam =  1/rate, size = size*shape) # array with shape(size*shape,)
        # in this way if we reshape the array to have shape=(size,shape) we can sum the rows to obtain
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
            
            # sum the rows and obtain an array with shape=(size,)
            return np.sum(y, axis=1) # in this way I have size number of chi-squared random variables
        
    
    #-----------------------------------------------------------------------------------------------------------------------#
    # NORMAL RVs
    #
    # we can generate normal random variables with the method explained in the slides: generating before 
    # normal random variables between [0,1) and then scaling to obtain the desider distribution
    def normal(self, mu, sigma, size=1_000):
        num_pairs = size//2
        # this is for andling the case where size is an odd number
        if size % 2:
            num_pairs += 1
        # calculating random values for the polar coordinates
        B = (self.chi_squared(ddof=2, size = num_pairs)) ** .5 # the radius
        theta = 2 * np.pi * self.uniform(size = num_pairs) # the angle
        
        z = np.concatenate((B * np.cos(theta), B * np.sin(theta)), axis=0) # concatenating all the obtained variables
        if size % 2:
            z = z[:-1]
        return mu + sigma * z # scaling to obtain the desider distribution

    #-----------------------------------------------------------------------------------------------------------------------#
    # LOGNORMAL RVs
    #
    # for generating the lognormal random variables we can generate normal random variables and exponentiate them
    def log_normal(self, mu, sigma_squared, size=1_000):
        return np.exp(self.normal(mu, np.sqrt(sigma_squared), size))
    
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
    # For generating rice random variables it can be used two independent normal random variables (x,y)
    # following respectively x~N(nu * cos(theta),sigma) y~N(nu * sin(theta), sigma) for any theta 
    # theta has been choosen to obtain cos(theta)==1 and sin(theta)==0
    # the norm of (x,y) will follow the Rice distribution
    def rice(self, nu, sigma, size = 1_000):
        if nu < 0 or sigma < 0:
            raise ValueError('Parameters nu and sigma should be greater or equal than 0')        
        
        x = self.normal(mu=nu, sigma=sigma, size=size)
        y = self.normal(mu=0, sigma=sigma, size=size)
        
        return (x**2 + y**2) ** .5
    
class FitAssessment:
    def __init__(self, sign_level=.05):
        # is the confidence level in which it can be accept the data as following a given distribution
        self.sign_level = sign_level
        
    def evaluation(self, two_moment, chi_squared, ks):
        av_mean, em_mean, av_var, em_var = two_moment
        mean_diff = abs(av_mean - em_mean) / av_mean
        var_diff = abs(av_var - em_var) / av_var
        
        print(f'Mean Percentage deviation: {mean_diff*100:.2f}%')
        print(f'Variance Percentage deviation: {var_diff*100:.2f}%')
        
        if mean_diff < .2 and var_diff < .2:
            print(f'Both percentage differences are small enough')
        else:
            print(f'At least one of the two percentage differenceas is to large')
        
        if chi_squared > self.sign_level and ks > self.sign_level:
            print(f'The two p_value are greater than the confidence level (α = {self.sign_level:.2f})')
            print('The rangom generated variables follow the given distribution')
        else:
            print(f'At least one of the two p_value is smaller than the confidence level (α = {self.sign_level:.2f})')
            print('The rangom generated variables DO NOT follow the given distribution')
        
    def first_two_moment(self, data, distr_name, *params):
        if distr_name == 'Rayleigh':
            sigma, = params
            mean = sigma * np.sqrt(np.pi / 2)
            variance = ((4 - np.pi) / 2 ) * sigma**2
        
        elif distr_name == 'Chi-Squared':
            (k,) = params
            mean = k
            variance = 2*k
            
        elif distr_name == 'LogNormal':
            mu, sigma_squared = params
            mean = np.exp(mu + (sigma_squared /2))
            variance = (np.exp(sigma_squared) -1) * np.exp(2*mu + (sigma_squared))
            
        elif distr_name == 'Beta':
            alpha, beta = params
            mean = alpha / (alpha+beta)
            variance = alpha*beta / (((alpha + beta)**2) * (alpha + beta + 1))
            
        elif distr_name == 'Rice':
            nu, sigma = params
            mean = stats.rice.mean(b=nu/sigma,loc=0, scale=sigma)
            variance = stats.rice.var(b=nu/sigma, scale=sigma)
            
        return mean, data.mean(), variance, data.var()
            
    def chi_squared_test(self, data, distr_name, *params):
        
        plt.figure(figsize=(8,6))
        
        if distr_name == 'Rayleigh':
            sigma, = params
            plt.title(f'Rayleigh Distribution (σ={sigma})')
            
            x = np.linspace(data.min(), data.max(), 50)
            pdf_plot = (x / (sigma**2)) * np.exp((-x**2) / (2 * sigma ** 2))
            plt.plot(x, pdf_plot, 'r', lw=2, label='Analytical PDF')
            
            hist,bins = np.histogram(data,bins=x, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            pdf = (bin_centers / (sigma**2)) * np.exp((-bin_centers**2) / (2 * sigma ** 2))
            chi2_stat = np.sum((hist - pdf)**2 / pdf)
            df = len(hist) -2 -1
            
        elif distr_name == 'Chi-Squared':
            (k,) = params
            plt.title(f'{distr_name} Distribution (k = {k})')
            
            x = np.linspace(data.min(), data.max(), 50)
            pdf_plot = stats.chi2.pdf(x, k)
            plt.plot(x, pdf_plot, 'r', lw=2, label='Analytical PDF')
            
            hist,bins = np.histogram(data,bins=x, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            pdf = stats.chi2.pdf(bin_centers, k)
            chi2_stat = np.sum((hist - pdf)**2 / pdf)
            df = len(hist) -2 -1
            
        elif distr_name == 'LogNormal':
            mu, sigma_squared = params
            plt.title(f'{distr_name} Distribution (v = {mu}, σ² = {sigma_squared})')
            
            x = np.linspace(data.min(), data.max(), 50)
            pdf_plot = (1 / (x *  np.sqrt(2*np.pi*sigma_squared))) * np.exp(-((np.log(x) - mu)**2) / 2 * sigma_squared )
            plt.plot(x, pdf_plot, 'r', lw=2, label='Analytical PDF')
            plt.yscale('log')
            
            hist,bins = np.histogram(data,bins=x, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            pdf = (1 / (bin_centers * np.sqrt(2*np.pi*sigma_squared))) * np.exp(-((np.log(bin_centers) - mu)**2) / 2 * sigma_squared )
            #pdf = stats.lognorm.pdf(bin_centers, mu, sigma)
            chi2_stat = np.sum((hist - pdf)**2 / pdf)
            df = len(hist) -2 -1
            
        elif distr_name == 'Beta':
            alpha, beta = params
            plt.title(f'{distr_name} Distribution  (α = {alpha}, β = {beta})')
            
            x = np.linspace(data.min(), data.max(), 50)
            pdf_plot = stats.beta.pdf(x, alpha, beta)
            plt.plot(x, pdf_plot, 'r', lw=2, label='Analytical PDF')
            
            hist,bins = np.histogram(data,bins=x, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            pdf = stats.beta.pdf(bin_centers, alpha, beta)
            chi2_stat = np.sum((hist - pdf)**2 / pdf)
            df = len(hist) -2 -1
        
        elif distr_name == 'Rice':
        
            nu,sigma = params
            plt.title(f'Rice Distribution (v={nu}, σ={sigma})')
                        
            x = np.linspace(0, data.max(), 50)
            pdf_plot = stats.rice.pdf(x, b=nu/sigma, loc=0, scale=sigma)
            plt.plot(x, pdf_plot, 'r', lw=2, label='Analytical PDF')
            
            hist,bins = np.histogram(data,bins=x, density=True)        
            bin_centers = (bins[:-1] + bins[1:]) / 2
            pdf = stats.rice.pdf(bin_centers, nu/sigma, scale=sigma)
            chi2_stat = np.sum((hist - pdf)**2 / pdf)
            df = len(hist) -2 -1
        
        p_value = 1 - stats.chi2.pdf(chi2_stat, df)
        
        #print(f"Chi2 statistic: {chi2_stat}")
        
        """
        if p_value < self.sign_level:
            print(f'p value = {p_value:.4f} < significance level = {self.sign_level:.4f} -> Reject the '\
                + f'null hypothesis: Data does not follow the {distr_name} distribution.')
        else:
            print(f"p value = {p_value:.4f} < significance level = {self.sign_level:.4f} -> Fail to "\
                +f"reject the null hypothesis: Data follows the {distr_name} distribution.")"""
        
        plt.hist(data, bins=50, density=True, alpha=0.5, label='Data Histogram')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True)
        file_name = os.path.join(folder_path, 'pdf_'+distr_name)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        #plt.show()
        
        return p_value
    
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
        plt.figure(figsize=(8, 6))
        plt.step(data_sorted, ecdf_values, label='ECDF')
        
        if distr_name == 'Rayleigh':
            (sigma,) = params
            cdf_values = 1 - np.exp((-data_sorted**2) / (2*sigma**2))
            plt.step(data_sorted, cdf_values, label=f'{distr_name} CDF')
            plt.title(f'Kolmogorov-Smirnov Test for {distr_name} Distribution with σ = {sigma}')
            
        elif distr_name == 'Chi-Squared':
            (k,) = params
            cdf_values = stats.chi2.cdf(data_sorted, k)
            plt.step(data_sorted, cdf_values, label=f'{distr_name} CDF')
            plt.title(f'Kolmogorov-Smirnov Test for {distr_name} Distribution with k = {k}')
            
        elif distr_name == 'LogNormal':
            mu, sigma_squared = params
            cdf_values = .5 * (1 + erf((np.log(data_sorted) - mu) / (np.sqrt(2) * np.sqrt(sigma_squared))))
            plt.step(data_sorted, cdf_values, label=f'{distr_name} CDF')
            plt.title(f'Kolmogorov-Smirnov Test for {distr_name} Distribution (v = {mu}, σ = {sigma_squared})')
            
        elif distr_name == 'Beta':
            alpha, beta = params
            cdf_values = stats.beta.cdf(data_sorted, alpha, beta)
            plt.step(data_sorted, cdf_values, label=f'{distr_name} CDF')
            plt.title(f'Kolmogorov-Smirnov Test for {distr_name} Distribution  (α = {alpha}, β = {beta})')
            
        elif distr_name == 'Rice':
            nu, sigma = params
            cdf_values = stats.rice.cdf(data_sorted, (nu/sigma), 0, sigma)
            plt.step(data_sorted, cdf_values, label=f'{distr_name} CDF')
            plt.title(f'Kolmogorov-Smirnov Test for {distr_name} Distribution (ν = {nu}, σ = {sigma})')
            #cdf_values = 1 - np.exp(-((data_sorted - nu)**2) / (2 * sigma**2))
            
        elif distr_name == 'GammaVariate':
            shape, rate = params  # Shape and rate parameters for gamma variate
            cdf_values = stats.gamma.cdf(data_sorted, shape, scale=1/rate)
            plt.step(data_sorted, cdf_values, label=f'{distr_name} CDF')
            plt.title(f'Kolmogorov-Smirnov Test for {distr_name} Distribution with (shape = {shape}, rate = {rate})')
            
        elif distr_name == 'Exponential':
            lam, = params  # Rate parameter for exponential
            cdf_values = 1 - np.exp(-lam * data_sorted)
            plt.step(data_sorted, cdf_values, label=f'{distr_name} CDF')
            plt.title(f'Kolmogorov-Smirnov Test for {distr_name} Distribution (λ = {lam})')
        
        # Max distance between the values and the expected values
        D = np.abs(ecdf_values - cdf_values)
        Dmax = D.max()
        #print(Dmax) # this is the max distance value, now i got to assest if this is inside the critical value
        
        # Critical Value
        # Critical Value for the test
        #critical_value = ksone.ppf(1 - self.sign_level / 2, n)        
        
        # p-value
        p_value = 1 - ksone.sf(Dmax, n)
        
        """
        # Compare D with the critical value
        if Dmax <= critical_value:
            print(f"D ({Dmax:.4f}) <= Critical Value ({critical_value:.4f}): Fail to reject the null hypothesis -> "\
                + f"data follows the {distr_name} distribution")
        else:
            print(f"D ({Dmax:.4f}) > Critical Value ({critical_value:.4f}): Reject the null hypothesis -> "\
                + f"data does not follow the {distr_name} distribution")    """
        
        # Plot the ECDF and CDF        
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.grid(True)
        file_name = os.path.join(folder_path, 'ks_'+distr_name)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        #plt.show()
        plt.close()
        
        return p_value   

def print_table(row):
    an_mean, em_mean, an_var, em_var, chi_p_value, ks_p_value = row

    data = [
        [an_mean, an_var, ""],
        [em_mean, em_var, ""],
        ["", "", chi_p_value],
        ["", "", ks_p_value]
    ]

    headers = ['Mean', 'Variance', 'p_value']
    row_headers = ["Analitical", "Empirical", "Chi-Squared test", "KS-test"]
    # Format the data
    formatted_data = [[f"{item:.4f}" if isinstance(item, float) else str(item) for item in row] for row in data]

    # Calculate the maximum width for each column
    max_widths = [max([len(row[i]) for row in formatted_data] + [len(headers[i])]) for i in range(len(headers))]

    # Calculate the maximum width for the row headers
    max_row_header_width = max(len(header) for header in row_headers)

    # Print the table
    print("-" * (sum(max_widths) + 3 * len(headers) + max_row_header_width + 3))
    print("| " + "".ljust(max_row_header_width) + " | " + " | ".join([headers[i].ljust(max_widths[i]) for i in range(len(headers))]) + " |")
    print("-" * (sum(max_widths) + 3 * len(headers) + max_row_header_width + 3))
    for i in range(len(formatted_data)):
        print("| " + row_headers[i].ljust(max_row_header_width) + " | " + " | ".join([formatted_data[i][j].ljust(max_widths[j]) for j in range(len(formatted_data[i]))]) + " |")
    print("-" * (sum(max_widths) + 3 * len(headers) + max_row_header_width + 3))

    
if __name__ == '__main__':
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'output_images',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    gen = Generator()
    fit = FitAssessment(.05)
    sizes = [100_000, 10_000, 1_000]
    
    for size in sizes:
        
        #-----------------------------------------------------------------------------------------------------------------------#
        # RAYLEIGH EVALUATION
        #
        sigma = 3
        data = gen.rayleigh(sigma,size)
        two_moment = fit.first_two_moment(data, 'Rayleigh', sigma)
        chi_p_value = fit.chi_squared_test(data, 'Rayleigh', sigma)
        ks_p_value = fit.ks_test(data, 'Rayleigh', sigma)
        
        print(f'\nRayleigh distribution (size={size}):')
        print_table(two_moment + (float(chi_p_value),float(ks_p_value)))
        fit.evaluation(two_moment, chi_p_value, ks_p_value)    
        print(f'\n============================================================\n')
        
        #-----------------------------------------------------------------------------------------------------------------------#
        # CHI-SQUARED EVALUATION
        #
        k = 10
        data = gen.chi_squared(ddof=k, size=size)
        two_moment = fit.first_two_moment(data, 'Chi-Squared', k)
        chi_p_value = fit.chi_squared_test(data, 'Chi-Squared', k)
        ks_p_value = fit.ks_test(data, 'Chi-Squared', k)
        
        print(f'Chi-Squared distribution (size={size}):')
        print_table(two_moment + (float(chi_p_value),float(ks_p_value)))
        fit.evaluation(two_moment, chi_p_value, ks_p_value)
        print(f'\n============================================================\n')
        
        #-----------------------------------------------------------------------------------------------------------------------#
        # LOGNORMAL EVALUATION
        #
        mu = 15
        sigma = 1
        data = gen.log_normal(mu, sigma, size)
        two_moment = fit.first_two_moment(data, 'LogNormal', mu, sigma)
        chi_p_value = fit.chi_squared_test(data, 'LogNormal', mu, sigma)
        ks_p_value = fit.ks_test(data, 'LogNormal', mu, sigma)
        
        print(f'LogNormal distribution (size={size}):')
        print_table(two_moment + (float(chi_p_value),float(ks_p_value)))
        fit.evaluation(two_moment, chi_p_value, ks_p_value)
        print(f'\n============================================================\n')    
        
        #-----------------------------------------------------------------------------------------------------------------------#
        # BETA EVALUATION
        #
        alpha = 5
        beta = 10
        data = gen.beta(alpha,beta,size)
        two_moment = fit.first_two_moment(data, 'Beta', alpha, beta)
        chi_p_value = fit.chi_squared_test(data, 'Beta', alpha, beta)
        ks_p_value = fit.ks_test(data, 'Beta', alpha, beta)
        
        print(f'Beta distribution (size={size}):')
        print_table(two_moment + (float(chi_p_value),float(ks_p_value)))
        fit.evaluation(two_moment, chi_p_value, ks_p_value)
        print(f'\n============================================================\n')
        
        
        #-----------------------------------------------------------------------------------------------------------------------#
        # RICE EVALUATION
        #
        nu = 0
        sigma = 1
        data = gen.rice(nu,sigma,size=size)
        two_moment = fit.first_two_moment(data, 'Rice', nu, sigma)
        chi_p_value = fit.chi_squared_test(data, 'Rice', nu, sigma)
        ks_p_value = fit.ks_test(data, 'Rice', nu, sigma)
        
        print(f'Rice distribution (size={size}):')
        print_table(two_moment + (float(chi_p_value),float(ks_p_value)))
        fit.evaluation(two_moment, chi_p_value, ks_p_value)
        print(f'\n============================================================\n')
    print('I saved the graphs directly in a subdirectory for the case size=1_000 (in the other cases '\
        +'the two lines are indistinguishable)')