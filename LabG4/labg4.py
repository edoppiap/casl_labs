import itertools
from queue import PriorityQueue
import random
import numpy as np

"""
Simulate the career of students attending Politecnico:

The number of students we have to simulate depends by the result of the confidence interval calculation.

Number of session = 3
Number of exam per session = 2
"""

TOTAL_EXAMS = 12
SUCCESS_PROB = .5
NUM_EXAMS_PER_SESS = 2
GRADE_DISTR = np.array([87, 62, 74, 55, 99, 94, 117, 117, 136, 160, 215, 160, 473])

class Student:
    def __init__(self,enrolled_time, total_exams):
        self.enrolled_time = enrolled_time
        self.total_exams = total_exams
        self.exams = []
    
    # Bernulli experiment that rule the success/failure of an exam
    def take_exams(self,n_exams, success_prob):
        # function that simulate the student that takes n exams 
        # and calculate the outcome
        for _ in range(n_exams):
            if random.random() < success_prob:
                self.exams.append(self.calculate_grade())
            if self.exams_finisched():
                break
            
    # The grade distribution has been generated from the data given in the Lab
    def calculate_grade(self):
        return np.random.choice(np.arange(18,31), p=grade_probs)
    
    def exams_finisched(self):
        return len(self.exams) == self.total_exams
    
    def final_grade(self):
        return sum(self.exams) / len(self.exams)
    
class Session:
    def __init__(self,semester_num, exams_per_sess):
        self.semester_num = semester_num
        self.num_exams = exams_per_sess
        
    def __lt__(self, other):
        return self.semester_num < other.semester_num
    
    
if __name__ == '__main__':
    
    grade_probs = GRADE_DISTR/ sum(GRADE_DISTR)
    
    # input parameters
    params_grid = {
        'total_exams' : [12,15,20], #  represent the total number of exams for the graduation
        'succ_prob' : [.3, .5, .8], # represent the probability that a student success at an exam
        'exams_per_sess' : [2] # represent the number of exams that a student can try each session
    }
    
    param_comb = list(itertools.product(*params_grid.values()))
    
    for total_exams, succ_prob, exams_per_sess in param_comb:
        print('Simulation parameters:')
        print(f'|  Total exams  | Success probability  |  Exam per session  |')
        print('|'+f'{total_exams}'.center(15)+'|'+f'{succ_prob*100:.2f}%'.center(22)+'|'+f'{exams_per_sess}'.center(20)+'|')
        
        sessions = PriorityQueue()
        semester_num = 1
        
        sessions.put(Session(semester_num, exams_per_sess))
        
        studs = [Student(1, total_exams) for _ in range(100)]
        graduated = []
        
        while not sessions.empty():
            session = sessions.get()
            semester_num = session.semester_num
            for stud in studs:
                stud.take_exams(session.num_exams, succ_prob)
                if stud.exams_finisched():
                    graduated.append(stud)
                    studs.remove(stud)
            if not len(studs):
                break
            else:
                sessions.put(Session(semester_num+1, exams_per_sess))
            
        mean_grade = sum(stud.calculate_grade() for stud in graduated) / len(graduated)
        print('Simulation ended.')
        print(f'Average grade results: {mean_grade:.2f}')
        print(f'Total number of sessions: {semester_num}')
        print('\n------------------------------------------------------------------------\n')