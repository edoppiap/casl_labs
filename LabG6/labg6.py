"""
    Consider a simulator for natural selection with the following simplified simulation model:

    All the individuals belong to the same species
    The initial population is equal to P
    The reproduction rate for each individual is lambda
    The lifetime LF(k) of individual k whose parent is d(k) is distributed according to the following distribution:
    LF(k)=
    uniform(LF(d(k),LF(d(k)*(1+alpha)) with probability prob_improve   
    uniform(0,LF(d(k)) with probability 1-prob_improve

    where prob_improve  is the probability of improvement for a generation and alpha is the improvement factor (>=0)

    Answer to the following questions:

    Describe some interesting questions to address based on the above simulator.
    List the corresponding output metrics.
    Develop the simulator
    Define some interesting scenario in terms of input parameters.
    Show and comment some numerical results addressing the above questions.
    Upload only the py code and the report (max 2 pages).
"""

from queue import PriorityQueue
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
parser = argparse.ArgumentParser(description='Input parameters for the simulation')

# Population parameters
parser.add_argument('--prob_improve', '--p_i', type=float, default=.01,
                    help='Probability of improvement of the lifetime')
parser.add_argument('--init_population', '--p', type=int, default=50,
                    help='Number of individuals for the 1st generation')
parser.add_argument('--improve_factor', '--alpha', type=float, default=.2,
                    help='Improve factor that an individual can develop at birth')

#-----------------------------------------------------------------------------------------------------------#
# MAIN METHOD
#
#
if __name__ == '__main__':
    args = parser.parse_args()
    print(f'Input parameters: {vars(args)}')
    
    # CREATE OUTPUT FOLDER
    #
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'outputs',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    print(f'Output images will be saved in the folder: {folder_path}')