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
import random
import os
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
PARAMS_GRID = {
    'total_exams' : [12,15,20], # the total number of exams for the graduation
    'succ_prob' : [.2, .4, .6, .8], # the probability that a student success at an exam
    'max_exam_per_sess' : [1,2], # the number of exams that a student can try each session
    'av_exam_per_sess' : [2,1] # the number of exams taken in average from each student
    #'var_exam_per_sess' : [0,2] # the variance in the number of exam taken from different student
}

# Confidence level
CONFIDENCE_LEVEL = .95
# Acceptance for the accurace
ACC_ACCEPTANCE = .97
# simulation time
SIM_TIME = 100

TOT_CFU = 132

MAX_APPLIED = 35

NUM_STUDENT = 300
# setting the seed
np.random.seed(42)

# -------------------------------------------------------------------------------------------------------#
# UTILS
#
def read_csv_exams_files():
    exams = []
    df =  pd.read_csv('LabG5/input_exams_.csv', sep=';')
    df['optional'] = df['optional'].replace(['No','no'], np.nan)
    for _,row in df.iterrows():
        exam = Exam(
            name = row['Name'],
            year = row['Year'],
            semester = row['semester'],
            passed = row['passed'],
            tot = row['tot'],
            cfu = row['CFU'],
            optional = row['optional'],
            max_stud = row['max_stud'] if not pd.isna(row['max_stud']) else None,
            grade_distr= [row[f'{i}'] for i in range(18,31)]
        )
        exams.append(exam)
    return exams

def generate_accademic_plan(exams: list):
    exam_dict = {}
    mandatory, chosen = [], []
    for exam in exams:
        optional = exam.optional
        if optional is not np.nan:
            if optional not in exam_dict:
                exam_dict[optional] = [exam]
            else:
                exam_dict[optional].append(exam)
        else:
            mandatory.append(exam)
    for key, exam_table in exam_dict.items():
        exam_table = [exam for exam in exam_table if exam.there_is_place()]
        tots = np.array([exam.tot for exam in exam_table])
        
        # we select the exams based on the past data of the enrollment
        if key != '5' and key != '6':
            chosen_exam = random.choices(exam_table, (tots / sum(tots)))[0]
            chosen_exam.studs += 1
            chosen.append(chosen_exam)
            
        # this is the case for the free credits
        elif key == '5':
            table_5_choice = random.choices(exam_table)[0]
            if table_5_choice.name == 'Free ECTS credits':
                table_6 = list(exam_dict['6'])
                internship = [el for el in table_6 if el.name == 'Internship'][0]
                table_6.remove(internship)
                
                #----------------------------------------------------------------------------------------#
                # RANDOM ELEMENT
                #
                # 40% of students chose the internship while the remain percentage pick the Free credits
                choice = random.choices([internship, table_6], [.40, .60])
                if isinstance(choice, Exam): # it is the intership
                    choice.studs += 1
                    chosen.append(choice)
                    
                elif isinstance(choice, list): # it is the free credits
                    cfus = sum([exam.cfu for exam in mandatory])
                    cfus += sum([exam.cfu for exam in chosen])
                
                    while cfus < TOT_CFU:
                        tots = np.array([exam.tot for exam in table_6])
                        free_credict_choice = random.choices(table_6, (tots / sum(tots)))[0]
                        free_credict_choice.studs += 1
                        cfus += free_credict_choice.cfu
                        table_6.remove(free_credict_choice)
                        chosen.append(free_credict_choice)
            else:
                table_5_choice.studs += 1
                chosen.append(table_5_choice)
    return mandatory + chosen

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
# CONDIFENCE INTERVAL METHOD
#
#
def calculate_confidence_interval(data, conf):
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
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
# STUDENT CLASS
# 
#
#
class Student:
    def __init__(self, all_exams=None):
        # OOP variables
        self.exams_to_take = generate_accademic_plan(all_exams)
        self.exams_taken = []
        self.current_year = 1 # this is because student with same enroll_year and same exams will follow the same courses
        self.current_semester = 1 # each year has 3 semester (Sep-Feb, Mar-Jul, Jul-Sept)
        
        # characteristic variable
        #self.sport = sport
        #self.isee = isee
        #self.tendency_to_study = tendency_to_study
        
        # final_grade
        self.final_grade = None
        self.honours = None
        
    # -------------------------------------------------------------------------------------------------------#
    # RANDOM ELEMENT
    # 
    # for the number of exams that each student try:
    # this generate a single random number from the average 
    # not greater than the max exams possible and the num exams left
    # in this way each student can try a different num of exams 
    def generate_num_exams(self, av_exams, max_exams):
        exams_left = len(self.exams_to_take)
        return min([np.random.poisson(av_exams), max_exams, exams_left])
    
    # -------------------------------------------------------------------------------------------------------#
    # RANDOM ELEMENT
    # 
    # Select the exams with more probability with the one with more than one call
    #
    def select_tryable_exams(self, max_exams):
        tryable_exams = [exam for exam in self.exams_to_take  if exam.year <= self.current_year and exam.semester <= self.current_semester]
        tryable_exams = [(exam, 2) if exam.semester == self.current_semester else (exam, 1) for exam in tryable_exams ]
        
        selected = []

        while max_exams > 0 and tryable_exams:
            exams, attempts = zip(*tryable_exams)
            # the weights for the choices are higher if a session has more than one call for that exam
            weights = [attempt / sum(attempts) for attempt in attempts]
            
            chosen = random.choices(exams, weights=weights)[0]
            index = exams.index(chosen)
            
            if attempts[index] > 0:
                if attempts[index] == 2 and max_exams >= 2:
                    selected.append((chosen, attempts[index]))
                    max_exams -= attempts[index]
                else:
                    selected.append((chosen, 1))
                    max_exams -= 1
            tryable_exams = [pair for pair in tryable_exams if not pair[0] == chosen]
        return selected
        
    def next_semester(self):
        self.current_year += (self.current_semester == 3)
        self.current_semester = (self.current_semester % 3) + 1
        
    
    # -------------------------------------------------------------------------------------------------------#
    # CALCULATE FINAL GRADE
    #
    #
    def calculate_final_grade(self):
        final_points = sum([grade*cfu for grade,cfu in self.exams_taken]) / sum([cfu for _,cfu in self.exams_taken]) * 110/30
        thesis = np.random.uniform(0,4,size=1)
        presentation = np.random.uniform(0,2,size=1)
        bonus = np.random.uniform(0,2,size=1)
        
        final_grade = final_points + thesis + presentation + bonus
        
        self.honours = final_grade[0] > 112.5
        
        self.final_grade = round(np.minimum(110, final_grade[0]))

# -------------------------------------------------------------------------------------------------------#
# EXAM CLASS
#
#        
class Exam:
    def __init__(self, name, year: int, semester: int, cfu: int, passed: int, tot: int, 
                 grade_distr: np.array, max_stud: int = None, group_influence: float = None, optional=None):
        self.name = name
        self.year = year
        self.semester = semester
        self.cfu = cfu
        self.optional = optional
        self.passed = passed
        self.tot = tot
        self.max_stud = max_stud
        self.group_influence = group_influence
        
        self.succ_prob = passed / tot
        self.grade_probs = [grade / sum(grade_distr) for grade in grade_distr]
        self.studs = 0
        
    def attempt(self):
        # --------------------------------------------------------------------------------------------------#
        # BERNOULLI EXPERIMENT
        #
        # rule the success/failure of an exam
        if np.random.binomial(n=1, p=self.succ_prob):
            # ----------------------------------------------------------------------------------------------#
            # GRADE RANDOM GENERATION
            #      
            # Generation of a random grades based on the input grade distribution
            return (np.random.choice(np.arange(18,31), p=self.grade_probs), self.cfu)
        return None
    
    def there_is_place(self):
        if self.max_stud is None:
            return True
        return self.max_stud > self.studs
        
       
#------------------------------------------------------------------------------------------------------#
# SESSION CLASS
# We can have 3 type of session (Winter, Summer, Autumn)
# there can be straordinary sessions but this has to be described in the input file 
#
class Session:
    def __init__(self, year: int, semester: int):
        self.year = year
        self.semester = semester
        
# -------------------------------------------------------------------------------------------------------#
# MAIN FUNCTION
#
#
if __name__ == '__main__':
    
    exams = read_csv_exams_files()
    
    applied = False
    i = 0
    c = 0
    while not applied and i < 200:
        student = Student(all_exams=exams)    
        if 'Applied data science project' in [exam.name for exam in student.exams_to_take]:
            c += 1
        i+=1
    if c:
        print(f'Number of students that choose Applied: {c} ({c/i * 100:.2f}%)')
    else:
        print('Not a single Applied student')
    
    # combination of the input parameters
    param_comb = list(itertools.product(*PARAMS_GRID.values()))
    
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} combinations '\
        +'[{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    
    dfs = []
    simulation_results = []
    #lst_grades = []
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
            results = []
            """
            #------------------------------------#
            # LOOP UNTIL AN ACCETTABLE ACCURACY HAS REACHED
            # 
            # Continue to add students until the accuracies have reached the level of acceptance
            while acc_grades < ACC_ACCEPTANCE or \
                acc_periods < ACC_ACCEPTANCE or \
                acc_exams_tried < ACC_ACCEPTANCE:
                    
                
                
                #------------------------------------#
                # CONFIDENCE LEVEL
                #
                interval, acc_grades = calculate_confidence_interval([result[0] for result in results], CONFIDENCE_LEVEL)
                interval_2, acc_periods = calculate_confidence_interval([result[1] for result in results], CONFIDENCE_LEVEL)
                interval_3, acc_exams_tried = calculate_confidence_interval([result[2] for result in results], CONFIDENCE_LEVEL)
            
            grades = [result[0] for result in results]
            session_passed = [result[1] for result in results]
            exam_tried = [result[2] for result in results]
            
            # Store the input parameters with the results
            result = {
                'Total exams': total_exams,
                'Success probability': succ_prob,
                'Max Exams per session': max_exam_per_sess,
                'Average exam per session': av_exams_per_sess,
                'Grade Mean': np.mean(grades),
                'Grade Std': np.std(grades, ddof=1),
                'Grade Interval low': interval[0],
                'Grade Interval up': interval[1],
                'Accuracy Grade': acc_grades,
                'Period mean': np.mean(session_passed),
                'Period Std': np.std(session_passed, ddof=1),
                'Period Interval low': interval_2[0],
                'Period Interval up': interval_2[1],
                'Accuracy Period': acc_periods,
                'Tried Mean': np.mean(exam_tried),
                'Tried Std': np.std(exam_tried, ddof=1),
                'Tried Interval low': interval_3[0],
                'Tried Interval up': interval_3[1],
                'Accuracy Tried': acc_exams_tried,
                'Num Students': len(results)
            }
            dfs.append(pd.DataFrame([result]))
            simulation_results.append(results)"""
            
    
    #df_result = pd.concat(dfs, ignore_index=True)