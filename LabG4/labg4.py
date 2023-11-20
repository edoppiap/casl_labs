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
import pickle
import matplotlib.pyplot as plt

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
    'max_exam_per_sess' : [1,2], # the number of exams that a student can try each session
    'av_exam_per_sess' : [2,1] # the number of exams taken in average from each student
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
def save_outputs(df, results):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'outputs',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} operations '\
        +'[{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        
    with tqdm(total=7, desc=f'Saving the outputs in outputs/{current_time}', bar_format=bar_format) as pbar:
            
        file_name = os.path.join(folder_path, 'results.csv')
        df.to_csv(file_name, index=False)
        
        pbar.update(1)
        
        file_name = os.path.join(folder_path, 'results.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump(results, f)
            
        pbar.update(1)
        
        #-----------------------------------------#
        # PLOT THE FINAL GRADE DISTRIBUTION
        #
        honours = False
        n_simulation_to_plot = 0
        # the prob to have honours is so low that this assure the plotted is one with at least one
        while not honours: 
            for i,simulation in enumerate(results):
                honours = 0
                graduations = []
                for grades,_,_ in simulation:
                    grade, honour = calculate_final_grade(grades)
                    graduations.append(grade)
                    honours += honour
                
                # Simulation to plot found
                if honours and int(df.loc[i][1] >= .6):
                    n_simulation_to_plot = i
                    tot_exams, succ_prob, _, av_exams = df.loc[n_simulation_to_plot][:4]
                    
                    plt.figure(figsize=(18,5))
                    
                    bins = (np.arange(66,112)-.4)
                    n, bins, _ = plt.hist(graduations, bins=bins, edgecolor='black', width=0.8)
                    plt.title(f'Histogram of final grades')
                    plt.xlabel('Grades')
                    plt.ylabel('Frequency')
                    plt.text(82, max(n)-10, f'N. students with honours: {honours}\nNumber of exams = {tot_exams:.0f}' \
                        + f'\nSuccess Probability = {succ_prob*100:.0f}%\nAv. n. exams per session = {av_exams:.0f}', 
                     bbox={'facecolor': 'lightgreen', 'pad': 10}, zorder=2)
                    plt.yticks(np.arange(1,max(n),4))
                    plt.xticks(np.arange(66,111))
                    plt.grid(which='major', axis='y', linestyle='--', color='gray', alpha=0.6)
                    file_name = os.path.join(folder_path, 'final_distr.')
                    plt.savefig(file_name, dpi=300, bbox_inches='tight')
                    plt.close()
                    #plt.show()
                    
                    break
                
        pbar.update(1)
        
        #-----------------------------------------#
        # PLOT THE GRADE DISTRIBUTION
        #
        grades = [grade for grades,_,_ in results[n_simulation_to_plot] for grade in grades ]
        bins = (np.arange(18,32)-.4)
        n,_,_ = plt.hist(grades, bins=bins, edgecolor='black', width=0.8)
        plt.title('Grade distribution ')
        plt.xlabel('Grades')
        plt.ylabel('Frequency')
        plt.xticks(np.arange(18,31))
        if (max(n) // 50) % 2:
            plt.yticks(np.arange(0,max(n)+100, 100))
        else:
            plt.yticks(np.arange(0,max(n), 100))
        plt.grid(True, linestyle='--', color='gray', alpha=0.6)
        plt.text(18, max(n)-100, f'Number of exams = {tot_exams:.0f}'+\
                        f'\nSuccess Probability = {succ_prob*100:.0f}%\nAv. n. exams per session = {av_exams:.0f}', 
                     bbox={'facecolor': 'lightgreen', 'pad': 10}, zorder=2)
        file_name = os.path.join(folder_path, "grades_distr")
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        #plt.show()
        
        pbar.update(1)
        
        #-----------------------------------------#
        # PLOT FOR YEAR TO GRADUATE DISTRIBUTION
        #
        session_per_year = 6
        years = [n_session/session_per_year for _,n_session,_ in results[n_simulation_to_plot]]

        width = (max(years) - min(years)) / 10
        bins = np.linspace(min(years), max(years), num=10)
        n,_,_ = plt.hist(years, bins=bins, edgecolor='black', width=width)
        plt.title('Histogram of years to graduate')
        plt.xlabel('years')
        plt.xticks(bins + width / 2)
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', color='gray', alpha=0.6)
        plt.text(max(years) * 2/3, max(n)-10, f'Number of exams = {tot_exams:.0f}'+\
                    f'\nSuccess Probability = {succ_prob*100:.0f}%\nAv. n. exams per session = {av_exams:.0f}', 
                    bbox={'facecolor': 'lightgreen', 'pad': 10}, zorder=2)
        file_name = os.path.join(folder_path, 'year_to_grad')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        #plt.show()
        
        pbar.update(1)
        
        #---------------------------------------------#
        # PLOT THE YEAR TO GRADUATE BASED ON PROB_SUCC
        #
        session_per_year = 6
        for av_exams in df['Average exam per session'].unique():

            selected_df = df[(df['Total exams'] == tot_exams) & 
                        (df['Average exam per session'] == av_exams) & 
                        (df['Max Exams per session'] == 2) ]
            plt.plot(selected_df['Success probability'], selected_df['Period mean'] / session_per_year, marker='o', zorder=2,
                    label=f'Average number of exam per session = {av_exams}')
            plt.errorbar(selected_df['Success probability'], 
                        selected_df['Period mean'] / session_per_year, 
                        yerr=[(selected_df['Period mean'] - selected_df['Period Interval low']) / session_per_year, 
                                (selected_df['Period Interval up'] - selected_df['Period mean']) / session_per_year],
                        fmt='o', capsize=5, c='black', zorder=1)

        plt.xlabel('Success probability')
        plt.ylabel('Average years for the graduation')
        #plt.yticks(np.arange(13))
        plt.title('Success probability vs Average years for the graduation')
        plt.grid(True)
        plt.legend()
        plt.text(.5, max(selected_df['Period mean'] / session_per_year)-5, f'Number of exams = {tot_exams:.0f}'+\
                        f'\nSuccess Probability = {succ_prob*100:.0f}%\nAv.e n. exams per session = {av_exams:.0f}', 
                     bbox={'facecolor': 'lightgreen', 'pad': 10}, zorder=2)
        file_name = os.path.join(folder_path, 'prob_succ_years')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        #plt.show()
        
        pbar.update(1)
        
        #---------------------------------------------#
        # PLOT THE NUM OF TRIES BASED ON PROB_SUCC
        #
        for av_exams in df['Average exam per session'].unique():
            selected_df = df[(df['Total exams'] == tot_exams) & 
                        (df['Average exam per session'] == av_exams) & 
                        (df['Max Exams per session'] == 2) ]
            plt.plot(selected_df['Success probability'], selected_df['Tried Mean'], marker='o', zorder=2,
                label=f'Average number of exam per session = {av_exams}')
            plt.errorbar(selected_df['Success probability'], 
                        selected_df['Tried Mean'] , 
                        yerr=[selected_df['Tried Mean'] - selected_df['Tried Interval low'], 
                            selected_df['Tried Interval up'] - selected_df['Tried Mean']],
                        fmt='o', capsize=5, c='black', zorder=1)

        plt.xlabel('Success probability')
        plt.ylabel('Average tries before the graduation')
        #plt.yticks(np.arange(13))
        plt.title('Success probability vs Average n_tries \nwith Confidence Interval ')
        plt.grid(True)
        plt.legend()
        plt.text(.5, max(selected_df["Tried Mean"] )-20, f'Number of exams = {tot_exams:.0f}'+\
                        f'\nSuccess Probability = {succ_prob*100:.0f}%\nAv. n.  exams per session = {av_exams:.0f}', 
                     bbox={'facecolor': 'lightgreen', 'pad': 10}, zorder=2)
        file_name = os.path.join(folder_path, 'n_tries')
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        #plt.show()
        
        pbar.update(1)

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
# BERNOULLI EXPERIMENT
#
# rule the success/failure of an exam
def take_exams(exam_to_try, succ_prob):
    return [calculate_grade() for _ in range(exam_to_try) if np.random.binomial(n=1, p=succ_prob)]

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
# CALCULATE FINAL GRADE
#
#
def calculate_final_grade(grades):
    final_points =  np.mean(grades) * 110/30
    thesis = np.random.uniform(0,4,size=1)
    presentation = np.random.uniform(0,2,size=1)
    bonus = np.random.uniform(0,2,size=1)
    
    final_grade = final_points + thesis + presentation + bonus
    
    honours = final_grade[0] > 112.5
    
    final_grade = round(np.minimum(110, final_grade[0]))
    
    return final_grade, honours

# -------------------------------------------------------------------------------------------------------#
# SINGLE STUDENT CARREER
#
#
def simulate_student_career(total_exams, av_exams, max_exams, succ_prob):
    exams_passed = []
    exam_tried, session_passed = 0,0
    while len(exams_passed) < total_exams: # exit when all exams have been passed
        exams_to_try = generate_num_exams(av_exams,
                                            max_exams,
                                            exams_left= total_exams - len(exams_passed))
        passed = take_exams(exams_to_try, succ_prob)
        session_passed += 1
        exam_tried += exams_to_try
        exams_passed += passed
        
    #final_grade, honours = calculate_final_grade(passed)
            
    return exams_passed, session_passed, exam_tried
        
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
            
            #------------------------------------#
            # LOOP UNTIL AN ACCETTABLE ACCURACY HAS REACHED
            # 
            # Continue to add students until the accuracies have reached the level of acceptance
            while acc_grades < ACC_ACCEPTANCE or \
                acc_periods < ACC_ACCEPTANCE or \
                acc_exams_tried < ACC_ACCEPTANCE:
            
                # for computation convenience we simulate 'studs_batch' students before computing the confidence intervals
                for _ in range(studs_batch):                    
                    #------------------------------------#
                    # CARRER OF A SINGLE STUDENT
                    #
                    results.append(
                        simulate_student_career(total_exams, av_exams_per_sess, max_exam_per_sess, succ_prob)
                    )
                
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
            simulation_results.append(results)
            
    
    df_result = pd.concat(dfs, ignore_index=True)
    save_outputs(df_result, simulation_results)