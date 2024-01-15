"""
The goal of Lab L7 is to define and simulate simple strategies 
to  control an  epidemic (SIR) process through non pharmaceutical interventions
(I.e. by introducing mobility restrictions).

Consider a homogeneous population of 50M individuals.
Fix R(0)=4 and \gamma= 1/14 days (recovering rate).    
Assume that  10% (6%) of the infected individuals  needs to be Hospitalized (H)  (undergo Intensive Treatments (IT).)
  
Fix the fatality rate of the epidemic to 3%.
H/IT places are limited (10k/50 k). Design some  non pharmaceutical intervention strategy that avoids H/IT overloads, 
and limits the number of death in 1 year to 100K.
To design your strategy you can use a mean-field SIR model.

Then, once you have defined your strategy simulate both the stochastic SIR and its mean field.  Are there significant differences, why? 
What happens if you scale down your population N to 10K (of course you have to scale down also other parameters, such as H and IT places)?


For the plagiarism checks, please upload your code here: 
https://www.dropbox.com/request/FSm4b6hTRu8qArbe9ImF
"""

#-----------------------------------------------------------------------------------------------------------#
# SIR MODEL POPULATION
# R(0) = 4 => number of susceptible individuals at time 0
# gamma = 1/14 days => recovery rate
#
#
def generate_sir_model(n):
    return