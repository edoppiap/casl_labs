# -------------------------------------------------------------------------------------------------------#
# IMPORTS
#
#
import numpy as np
from tqdm import tqdm
from scipy.stats import t
import pandas as pd
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import logistic
import copy

# -------------------------------------------------------------------------------------------------------#
# INPUT PARAMETERS
#
#
PARAMS_GRID = {
    'total_exams' : [12,15,20], # the total number of exams for the graduation
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

STUDENT_PER_YEAR = 300

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

def plot_graphs(students):
    
    graduations = [student.final_grade for student in students]
    
    plt.figure(figsize=(18,5))
                    
    bins = (np.arange(66,112)-.4)
    n, bins, _ = plt.hist(graduations, bins=bins, edgecolor='black', width=0.8)
    plt.title(f'Histogram of final grades')
    plt.xlabel('Grades')
    plt.ylabel('Frequency')
    plt.yticks(np.arange(1,max(n),4))
    plt.xticks(np.arange(66,111))
    plt.grid(which='major', axis='y', linestyle='--', color='gray', alpha=0.6)
    plt.show()
    
    sessions = [student.grade_session for student in students]
    # Create labels for the x-axis (combination of year and semester)
    labels = [f"{session.year}-{session.semester}" for session in sessions]
    
    print(labels)

    # Get unique labels and their counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Plot the histogram using plt.bar
    plt.title('Histogram of years and semesters to graduate')
    plt.bar(unique_labels, label_counts, edgecolor='black', align='center')
    plt.xlabel('Years and Semesters')
    plt.ylabel('Frequency')
    plt.xticks(ha='right')

    plt.grid(True, linestyle='--', color='gray', alpha=0.6)
    plt.show()
    
    intership_studs = [student.final_grade for student in students if student.intership_n_sessions is not None]
    other_studs = [student.final_grade for student in students if student.intership_n_sessions is None]
    
    print(graduations)
    print(intership_studs)
    print(other_studs)
    
    plt.figure(figsize=(12,6))
    plt.title('Histogram of final grades')
    bins = (np.arange(66,112)-.4)
    plt.hist(graduations, bins=bins, edgecolor='black', alpha=.5, width=0.8, label=f'All students')
    plt.hist(intership_studs, bins=bins, edgecolor='black', alpha=.5, width=0.8, label=f'Internship students')
    plt.hist(other_studs, bins=bins, edgecolor='black', alpha=.5, width=0.8, label=f'Non-internship students')
    plt.xlabel('Grades')
    plt.ylabel('Frequency')
    #plt.yticks(np.arange(1,max(n),4))
    plt.xticks(np.arange(66,111))
    plt.grid(which='major', axis='y', linestyle='--', color='gray', alpha=0.6)
    plt.legend()
    plt.show()
    
# -------------------------------------------------------------------------------------------------------#
#
# Read the input file for the exams
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
            exam.studs += 1
            mandatory.append(exam)
    for key, exam_table in exam_dict.items():
        exam_table = [exam for exam in exam_table if exam.has_available_space()]
        tots = np.array([exam.tot for exam in exam_table])
        
        #----------------------------------------------------------------------------------------#
        # RANDOM ELEMENT
        #
        # Select the exams based on the past data of the enrollment
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
        succ_prob = self.succ_prob if not np.isnan(self.succ_prob) else 1
        if np.random.binomial(n=1, p=succ_prob):
            # ----------------------------------------------------------------------------------------------#
            # GRADE RANDOM GENERATION
            #      
            # Generation of a random grades based on the input grade distribution
            if not all(np.isnan(el) for el in self.grade_probs):
                return (np.random.choice(np.arange(18,31), p=self.grade_probs), self.cfu)
            else:
                return (None, self.cfu)                
        return None
    
    def has_available_space(self):
        return self.max_stud is None or self.max_stud > self.studs
        
       
#------------------------------------------------------------------------------------------------------#
# SESSION CLASS
# We can have 3 type of session (Winter, Summer, Autumn)
# there can be straordinary sessions but this has to be described in the input file 
#
class Session:
    def __init__(self, year: int, semester: int):
        self.year = year
        self.semester = semester        
    
    def next_semester(self):
        self.year += (self.semester == 3)
        self.semester = (self.semester % 3) + 1
        
    def __lt__(self, other):
        if self.year == other.year:
            return self.semester < other.semester
        else:
            return self.year < other.year
        
    def __eq__(self, other: object) -> bool:
        return (self.year, self.semester) == (other.year, other.semester)
    
    def more_than(self, n: int, other: object):
        year_difference = abs(self.year - other.year)
        semester_difference = abs(self.semester - other.semester)

        total_semesters_apart = year_difference * 3 + semester_difference

        return total_semesters_apart >= n

# -------------------------------------------------------------------------------------------------------#
# STUDENT CLASS
# 
#
#
class Student:
    def __init__(self, all_exams=None):
        # OOP variables
        self.exams_to_take = []
        self.exams_taken = []
        self.intership_n_sessions = None
        self.challenge_n_session = None
        self.session_start_activity = None
        
        self.generate_accademic_plan(all_exams)
        
        # characteristic variable
        #self.sport = sport
        #self.isee = isee
        #self.tendency_to_study = tendency_to_study
        
        # final_grade
        self.final_grade = None
        self.honours = None
        self.grade_session = None
        
    def generate_accademic_plan(self, all_exams):
        exams = generate_accademic_plan(all_exams)
        names = [exam.name for exam in exams]
        
        if 'Internship' in names:
            exams = [exam for exam in exams if exam.name != 'Internship']
            self.intership_n_sessions = np.random.randint(1,4) # not inclusive
        elif 'Challenge' in names:
            exams = [exam for exam in exams if exam.name != 'Challenge']
            self.challenge_n_session = 1
        
        self.exams_to_take = exams
    
    # -------------------------------------------------------------------------------------------------------#
    # RANDOM ELEMENT
    #
    # This is a function that start the internship or the challenge based on the number of exam that the student
    # has, (the more exam, the less is probable that the students starts one of them)
    def start_internship_or_challenge(self, session: Session):
        # LOGISTIC DISTRIBUTION to calculate the probability to start the intership 
        # based on the number of exam left 
        n_exam_left = len(self.exams_to_take)
        if n_exam_left == 0:
            # add this to be shure it starts
            prob = 1
        else:
            prob = 1 - logistic.cdf(len(self.exams_to_take), loc=2.5, scale=.5)
            
        # BERNULLI OUTCOME for starting the internship (or the challenge)
        outcome =  np.random.binomial(n=1, p=prob)
        if outcome:
            self.session_start_activity = copy.copy(session)

        
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
    def select_tryable_exams(self, num_exams, current_year, current_semester):
        tryable_exams = [exam for exam in self.exams_to_take  if exam.year <= current_year and exam.semester <= current_semester]
        tryable_exams = [(exam, 2) if exam.semester == current_semester else (exam, 1) for exam in tryable_exams]
        
        selected = []

        while num_exams > 0 and tryable_exams:
            exams, attempts = zip(*tryable_exams)
            # the weights for the choices are higher if a session has more than one call for that exam
            weights = [attempt / sum(attempts) for attempt in attempts]
            
            chosen = random.choices(exams, weights=weights)[0]
            index = exams.index(chosen)
            
            if attempts[index] > 0:
                if attempts[index] == 2 and num_exams >= 2:
                    selected.append((chosen, attempts[index]))
                    num_exams -= attempts[index]
                else:
                    selected.append((chosen, 1))
                    num_exams -= 1
            tryable_exams = [pair for pair in tryable_exams if not pair[0] == chosen]
        return selected
    
    def already_graduated(self):
        return self.final_grade != None
    
    def can_graduate(self, current: Session):
        if self.session_start_activity is not None:
            pass
        if not self.challenge_n_session and \
            not self.intership_n_sessions and \
            not self.session_start_activity:
            return len(self.exams_to_take) == 0
        elif self.intership_n_sessions:
            temp_1 = self.session_start_activity is not None and self.session_start_activity.more_than(self.intership_n_sessions, current)
            return len(self.exams_to_take) == 0 and temp_1
        else:
            temp_1 = self.session_start_activity is not None and self.session_start_activity.more_than(1, current)
            return len(self.exams_to_take) == 0 and temp_1
                
    
    # -------------------------------------------------------------------------------------------------------#
    # CALCULATE FINAL GRADE
    #
    #
    def calculate_final_grade(self, session: Session):
        final_points = sum([grade*cfu for grade,cfu in self.exams_taken]) / sum([cfu for _,cfu in self.exams_taken]) * 110/30
        thesis = np.random.uniform(0,4,size=1)
        presentation = np.random.uniform(0,2,size=1)
        bonus = np.random.uniform(0,2,size=1)
        
        final_grade = final_points + thesis + presentation + bonus
        
        self.honours = final_grade[0] > 112.5        
        self.final_grade = round(np.minimum(110, final_grade[0]))
        self.grade_session = session
        
def simulate(students: list[Student]):
    
    max_session = Session(10,1)
    
    current_session = Session(1,1)
    graduaded = []
    
    # Event Loop
    while len(students) > 0 and current_session < max_session:       
        
        for student in students:
            n_exam = student.generate_num_exams(3, 4)
            exams_to_try = student.select_tryable_exams(n_exam, current_session.year, current_session.semester)
            if not student.challenge_n_session and not student.intership_n_sessions:
                student.start_internship_or_challenge(current_session) # Random Element
            
            for exam,n_attempts in exams_to_try:
                attempt = None
                while n_attempts > 0 and attempt is None:
                    attempt = exam.attempt()
                    if attempt:
                        student.exams_to_take.remove(exam)
                        student.exams_taken.append(attempt)
            
            if student.already_graduated():
                print('Già laureato')

            if not student.already_graduated() and student.can_graduate(current_session):
                students.remove(student)
                student.calculate_final_grade(copy.copy(current_session))
                graduaded.append(student)
        
        current_session.next_semester()
        
    return graduaded
        
# -------------------------------------------------------------------------------------------------------#
# MAIN FUNCTION
#
#
if __name__ == '__main__':
    
    exams = read_csv_exams_files()
    
    students = [Student(all_exams=exams) for _ in range(STUDENT_PER_YEAR)]
    
    graduaded = simulate(students)

    plot_graphs(graduaded)
    
    """
    # combination of the input parameters
    param_comb = list(itertools.product(*PARAMS_GRID.values()))
    
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} combinations '\
        +'[{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    
    dfs = []
    simulation_results = []
    #lst_grades = []
    #lst_grades,lst_periods, lst_tried = [],[],[]
        
    # foor loop for each combination of parameters
    for total_exams, max_exam_per_sess, av_exams_per_sess in \
        tqdm(param_comb, desc='Simulating all combinations of input parameters', 
             bar_format=bar_format):
            
            # choosed because with this value I can reach an accettable accuracy
            # without simulating too many students
            studs_batch = 10
            
            # INITIALIZZATION
            acc_grades, acc_periods, acc_exams_tried = 0,0,0
            results = []
            
            #------------------------------------#
            # LOOP UNTIL AN ACCETTABLE ACCURACY HAS REACHED
            # 
            # Continue to simulates a whole course until the accuracies have reached the level of acceptance
            while acc_grades < ACC_ACCEPTANCE or \
                acc_periods < ACC_ACCEPTANCE or \
                acc_exams_tried < ACC_ACCEPTANCE:
                    
                students = [Student(all_exams=exams) for _ in range(STUDENT_PER_YEAR)]
                
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
            simulation_results.append(results)
            
    
    df_result = pd.concat(dfs, ignore_index=True)"""