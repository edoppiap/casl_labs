# -------------------------------------------------------------------------------------------------------#
# IMPORTS
#
#
import itertools
from queue import PriorityQueue
import numpy as np
from tqdm import tqdm
from scipy.stats import t
import pandas as pd
import os
from datetime import datetime

"""
Simulate the career of students attending Politecnico:

The number of students we have to simulate depends by the result of the confidence interval calculation.

Number of session = 3
Number of exam per session = 2
"""
# -------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
PARAMS_GRID = {
    'total_exams' : [12,15,20], # the total number of exams for the graduation
    'succ_prob' : [.2, .4, .6, .8], # the probability that a student success at an exam
    'max_exam_per_sess' : [1,2,3,4], # the number of exams that a student can try each session
    'av_exam_per_sess' : [4,3,2,1] # the number of exams taken in average from each student
    #'var_exam_per_sess' : [0,2] # the variance in the number of exam taken from different student
}

# distribution of the grade based on history data
GRADE_DISTR = np.array([87, 62, 74, 55, 99, 94, 117, 117, 136, 160, 215, 160, 473])
# Confidence level
CONFIDENCE_LEVEL = .95
# Acceptance for the accurace
ACC_ACCEPTANCE = .97
# simulation time
SIM_TIME = 100
# setting the seed
np.random.seed(42)

#--------------------------------------------------------------------------------------------------------#
# UTILS
#
def save(df, *arrays):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'outputs',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file_name = os.path.join(folder_path, 'results.csv')
    df.to_csv(file_name, index=False)
    
    file_name = os.path.join(folder_path, 'arrays.npy')
    with open(file_name, 'wb') as f:
        for array in arrays:
            np.save(f, array)

# -------------------------------------------------------------------------------------------------------#
# GRADE RANDOM GENERATION
#      
# Generation of a random grades based on the input grade distribution
def calculate_grade():
    return np.random.choice(np.arange(18,31), p=grade_probs)

# -------------------------------------------------------------------------------------------------------#
# RANDOM ELEMENT
# 
# for the number of exams that each student try:
# this generate a single random number from the average 
# not greater than the max exams possible and the num exams left
# in this way each student can try a different num of exams 
def generate_num_exams(av_exams, max_exams, exams_left):
    minimum_max = np.minimum(max_exams, exams_left)
    return np.minimum(np.random.poisson(av_exams), minimum_max)

# -------------------------------------------------------------------------------------------------------#
# BERNULLI EXPERIMENT
#
# rule the success/failure of an exam
def take_exams(exam_to_try, succ_prob):
    tried = 0
    passed = []
    
    # the bernulli experiment is repeated n = exam_to_try times
    for _ in range(exam_to_try):
        tried += 1
        if np.random.random() < succ_prob:
            passed.append(calculate_grade())
            
    return tried, passed

# -------------------------------------------------------------------------------------------------------#
# CONDIFENCE INTERVAL METHOD
#
#
def calculate_confidence_interval(data, conf):
    
    mean = data.mean()
    std = data.std(ddof=1)
    se = std / (len(data)**(1/2)) # this is the standard error
    
    interval = t.interval(confidence = conf, # confidence level
                          df = len(data)-1, # degree of freedom
                          loc = mean, # center of the distribution
                          scale = se # spread of the distribution 
                          # (we use the standard error as we use the extimate of the mean)
                          )
    
    MOE = interval[1] - interval[0] # this is the margin of error
    re = (MOE / (2 * abs(mean))) # this is the relative error
    acc = 1 - re # this is the accuracy
    return interval,acc
        
# -------------------------------------------------------------------------------------------------------#
# MAIN FUNCTION
#
#
if __name__ == '__main__':
    
    # Grade probability distribution based on history data
    grade_probs = GRADE_DISTR/ sum(GRADE_DISTR)
    
    # combination of the input parameters
    param_comb = list(itertools.product(*PARAMS_GRID.values()))
    
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} combinations '\
        +'[{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    
    dfs = []
    lst_grades = []
    #lst_grades,lst_periods, lst_tried = [],[],[]
        
    # foor loop for each combination of parameters
    for total_exams, succ_prob, max_exam_per_sess, av_exams_per_sess in \
        tqdm(param_comb, desc='Simulating all combinations of input parameters', 
             bar_format=bar_format):
            
            # choosed because with this value I can reach an accettable accuracy
            # without simulating too many students
            studs_batch = 10
            
            # INITIALIZZATION
            acc_grades, acc_periods, acc_exams_tried = 0,0,0
            grades, lst_session_passed,lst_exam_tried = None, None, None
            
            #------------------------------------#
            # LOOP UNTIL AN ACCETTABLE ACCURACY HAS REACHED
            # 
            # Continue to add students until the accuracies have reached the level of acceptance
            while acc_grades < ACC_ACCEPTANCE or \
                acc_periods < ACC_ACCEPTANCE or \
                acc_exams_tried < ACC_ACCEPTANCE:
            
                # for computation convenience we simulate 'studs_batch' before computing the confidence intervals
                for _ in range(studs_batch):
                    exams_passed = []
                    exam_tried, session_passed = 0,0
                    
                    #------------------------------------#
                    # CARRER OF A SINGLE STUDENT
                    #
                    while len(exams_passed) < total_exams: # exit when all exams have been passed
                        exams_to_try = generate_num_exams(av_exams=av_exams_per_sess,
                                                          max_exams=max_exam_per_sess,
                                                          exams_left= total_exams - len(exams_passed))
                        tried,passed = take_exams(exams_to_try, succ_prob)
                        session_passed +=1
                        exam_tried += tried
                        exams_passed += passed
                        
                    # store the result into np.arrays
                    if grades is None and \
                        lst_session_passed is None and \
                        lst_exam_tried is None:
                        grades = np.array(exams_passed)
                        lst_session_passed = np.array(session_passed)
                        lst_exam_tried = np.array(exam_tried)
                    else:
                        grades = np.vstack([grades, exams_passed])
                        lst_session_passed = np.hstack([lst_session_passed, session_passed])
                        lst_exam_tried = np.hstack([lst_exam_tried, exam_tried])
                
                #------------------------------------#
                # CONFIDENCE LEVEL
                #
                interval,acc_grades = calculate_confidence_interval(grades, CONFIDENCE_LEVEL)
                interval_2, acc_periods = calculate_confidence_interval(lst_session_passed, CONFIDENCE_LEVEL)
                interval_3, acc_exams_tried = calculate_confidence_interval(lst_exam_tried, CONFIDENCE_LEVEL)
                
            # Store the input parameters with the results
            result = {
                'Total exams': total_exams,
                'Success probability': succ_prob,
                'Max Exams per session': max_exam_per_sess,
                'Average exam per session': av_exams_per_sess,
                #'Variance exams per session': var_exams,
                'Grade Mean': grades.mean(),
                'Grade Std': grades.std(ddof=1),
                'Grade Interval low': interval[0],
                'Grade Interval up': interval[1],
                'Accuracy Grade': acc_grades,
                'Period mean': lst_session_passed.mean(),
                'Period Std': lst_session_passed.std(ddof=1),
                'Period Interval low': interval_2[0],
                'Period Interval up': interval_2[1],
                'Accuracy Period': acc_periods,
                'Tried Mean': lst_exam_tried.mean(),
                'Tried Std': lst_exam_tried.std(ddof=1),
                'Tried Interval low': interval_3[0],
                'Tried Interval up': interval_3[1],
                'Accuracy Tried': acc_exams_tried,
                'Num Students': len(lst_session_passed)
            }
            dfs.append(pd.DataFrame([result]))
            lst_grades.append(grades.flatten())
            #lst_periods.append(lst_session_passed)
            #lst_tried.append(lst_exam_tried)
    
    df_result = pd.concat(dfs, ignore_index=True)
    lst_grades = np.hstack(lst_grades)
    #lst_periods = np.concatenate(lst_periods)
    #lst_tried = np.concatenate(lst_tried)
    save(df_result, lst_grades)