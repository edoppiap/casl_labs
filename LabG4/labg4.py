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
    'succ_prob' : [.2, .4, .5, .7, .8], # the probability that a student success at an exam
    'max_exam_per_sess' : [3,4], # the number of exams that a student can try each session
    'av_exam_per_sess' : [3,2], # the number of exams taken in average from each student
    'var_exam_per_sess' : [0,2] # the variance in the number of exam taken from different student
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
def save_df(df):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'outputs',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file_name = os.path.join(folder_path, 'results.csv')
    df.to_csv(file_name, index=False)


# -------------------------------------------------------------------------------------------------------#
# STUDENT DATA STRUCTURE
#
#
class Student:
    def __init__(self, total_exams, av_exam, var_exam):
        self.num_session_passed = 0
        self.total_exams = total_exams
        self.av_exams_per_sess = av_exam
        self.var_exams = var_exam
        self.exams = []
    
    # Bernulli experiment that rule the success/failure of an exam
    def take_exams(self, max_exams, success_prob):
        self.num_session_passed += 1
        
        #------------------------------------------------------------#
        # RANDOM element for the number of exams that each student try:
        # this generate a single random number from the average and 
        # the var, each student can so try a different num of exams 
        exam_to_try = int(np.round(
            np.random.normal(loc=self.av_exams_per_sess, 
                             scale=self.var_exams,
                             size=1))
                          )
        # a student cannot try more than max_exams each session
        exam_to_try = exam_to_try if exam_to_try <= max_exams else max_exams
        
        # function that simulate the student that takes n exams 
        # and calculate the outcome
        for _ in range(exam_to_try):
            if np.random.random() < success_prob:
                self.exams.append(self.calculate_grade())
            if self.has_finished():
                break
            
    # The grade distribution has been generated from the data given in the Lab
    def calculate_grade(self):
        return np.random.choice(np.arange(18,31), p=grade_probs)
    
    # Returns true if the student has no more exams to take
    def has_finished(self):
        return len(self.exams) == self.total_exams
    
    # Returns the grade of the exam taken
    def final_grade(self):
        return sum(self.exams) / len(self.exams)

# -------------------------------------------------------------------------------------------------------#
# SESSION DATA STRUCTURE
#
#
class Session:
    def __init__(self,semester_num, max_exam_per_sess):
        self.semester_num = semester_num
        self.max_exams = max_exam_per_sess
        
    def __lt__(self, other):
        return self.semester_num < other.semester_num
    
    
# -------------------------------------------------------------------------------------------------------#
# CONDIFENCE INTERVAL METHOD
#
#
def calculate_confidence_interval(data, conf):
    """
    This will perform an estimate of the average of the population and return the confidence interval 
    with its accuracy. It use the scipy.stats.t.interval function and it returns a coupla with 
    containing (interval,accuracy)

    :param X: sample for which estimate the average and the confidence interval
    :param C: the choosen confidence intervla
    :return: ((lower_bound,upper_bound),float64)
    """
    
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
    
    dfs = []
    
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} combinations [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    
    # foor loop for each combination of parameters
    for total_exams, succ_prob, max_exam_per_sess, av_exams_per_sess, var_exams in \
        tqdm(param_comb, desc='Simulating all combinations of input parameters', 
             bar_format=bar_format):
        #print('Simulation parameters:')
        #print(f'|  Total exams  | Success probability  |  Max exam per session  | Average exams taken per session | Var exams |')
        #print('|'+f'{total_exams}'.center(15)+'|'+f'{succ_prob*100:.2f}%'.center(22)\
        #    +'|'+f'{max_exam_per_sess}'.center(24)+'|'+f'{av_exams_per_sess}'.center(33)+\
        #        '|'+f'{var_exams}'.center(11)+'|')
        
        studs_batch = 10 # num of students we add each iteration
        
        # INITIALIZZATION
        acc_grades, acc_periods = 0 
        grades, periods = None, None
        
        # Continue to add students until the accuracies have reached the level of acceptance
        while acc_grades < ACC_ACCEPTANCE or acc_periods < ACC_ACCEPTANCE:
            
            # FES DATA STRUCTURE
            sessions = PriorityQueue()
            semester_num = 1
            
            sessions.put(Session(semester_num, max_exam_per_sess))
        
            students = [Student(total_exams, av_exams_per_sess, var_exams) for _ in range(studs_batch)]
            
            #------------------------------------#
            # EVENT LOOP
            #
            while semester_num < SIM_TIME:
                if sessions.empty():
                    break
                
                # get the next session from the FES
                session = sessions.get()
                semester_num = session.semester_num
                
                for stud in students:
                    stud.take_exams(session.max_exams, succ_prob)
                    if stud.has_finished():
                        students.remove(stud) # remove from the student that have to graduate
                        
                        # fill the arrays for the evaluation with its statistics
                        if grades is None and periods is None:
                            # the case where it's the first student
                            grades = np.array([exam for exam in stud.exams])
                            periods = np.array(stud.num_session_passed)
                        else:
                            grades = np.vstack([grades, [exam for exam in stud.exams]])
                            periods = np.hstack([periods, stud.num_session_passed])
                
                if stud: # there are still students that have to graduate
                    # create a new session 
                    sessions.put(Session(semester_num+1, max_exam_per_sess))
            
            #------------------------------------#
            # CONFIDENCE LEVEL
            #
            interval,acc_grades = calculate_confidence_interval(grades, CONFIDENCE_LEVEL)
            interval_2, acc_periods = calculate_confidence_interval(periods, CONFIDENCE_LEVEL)
            
        # Store the input parameters with the results
        result = {
            'Total exams': total_exams,
            'Success probability': succ_prob,
            'Max Exams per session': max_exam_per_sess,
            'Average exam per session': av_exams_per_sess,
            'Variance exams per session': var_exams,
            'Grade Mean': grades.mean(),
            'Grade Std': grades.std(ddof=1),
            'Grade Interval low': interval[0],
            'Grade Interval up': interval[1],
            'Accuracy Grade': acc_grades,
            'Period mean': periods.mean(),
            'Period Std': periods.std(ddof=1),
            'Period Interval low': interval_2[0],
            'Period Interval up': interval_2[1],
            'Accuracy Period': acc_periods,
            'Num Students': len(periods)
        }
        dfs.append(pd.DataFrame([result]))
            
        #mean_grade = sum(stud.calculate_grade() for stud in graduated) / len(graduated)
        #av_time = sum(stud.num_session_passed for stud in graduated)*2 / len(graduated) # each mini-session happen every two months
        #print('Simulation ended.')
        #print(f'Average time for the graduation: {av_time} months ({av_time/12:.2f} years)')
        #print(f'Average grade results: {mean_grade:.2f}')
        #print(f'Total number of sessions: {semester_num}')
        #print('\n------------------------------------------------------------------------\n')
    
    df_result = pd.concat(dfs, ignore_index=True)
    save_df(df_result)