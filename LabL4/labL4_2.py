import random
import matplotlib.pyplot as plt
from queue import PriorityQueue, Queue, Full, Empty
import numpy as np
from scipy.stats import t

# arrival process of customers is a poisson process a rate lambda in [5,7,9,10,12,15] (i.e. interarrivals are exponentially distributed with average 1/lambda)
# this means that the arrival process is not a singular exponential distribution but in fact are 6 different distribution

arrival_lambdas = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999]
service_distrs = ['deterministic','exponential','hyper-exponential']

# service times are exponentially distributed with average 1
SERVICE = 1.0

# the simulation time must be great enough to capture the saturaion of the system
SIM_TIME = 100_000

# Maximum capacity of the queue
MAX_QUEUE_CAPACITY = 1000

# 
TYPE1 = 'Client1'

def generate_service_time(lam: float = 1, distr: str = 'deterministic'):
    if distr == 'deterministic':
        return 1
    elif distr == 'exponential':
        return random.expovariate(lam)
    elif distr == 'hyper':
        pass

# half a page of a pdf with the results and a comment
# what will be the condition which I choose

# evaluate the average delay and the dropping probability

# this class will store the info we are interested in about the simulation
class Measure:
    def __init__(self,Narr=0,Ndep=0,NAverageUser=0,OldTimeEvent=0,AverageDelay=0,Dropped=0):
        self.num_arrivals = Narr
        self.num_departures = Ndep
        self.average_utilization = NAverageUser
        self.time_last_event = OldTimeEvent
        self.average_delay_time = AverageDelay
        self.num_dropped = Dropped
        self.av_delays = []
        self.av_no_cust = []

# this class is the representation of a client
class Client:
    def __init__(self,Type,arrival_time):
        self.type_ = Type
        self.arrival_time = arrival_time
        
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
    return mean,interval,acc

def arrival(time, FES, queue, data, lambd, serv_distr):
    global users
    global current
    
    # Create data for measuring simulation
    data.num_arrivals += 1
    data.average_utilization += users*(time - data.time_last_event)
    data.time_last_event = time
    
    # compute the inter-arrival time Tia for next client
    inter_arrival = random.expovariate(lambd)
    
    # schedule an arrival at time Tcurr + Tia
    FES.put((time + inter_arrival, "arrival"))
    
    # Create a record for the client
    client = Client(TYPE1,time)
    
    users += 1
    
    # If the server is idle -> make the server busy
    if current is None:
        # determine the service time Ts
        service_time = generate_service_time(distr=serv_distr)
        current = client # this is for saying that the server is busy
        
        # schedule the end of service at time Tcurr + Ts
        FES.put((time + service_time, 'departure'))
        
    else: # add the client to the queue
        
        try:
            # Insert the record in the queue
            queue.put(client, timeout=0)
        except Full:
            # drop the client because the queue is at capacity
            users -= 1

def departure(time, FES, queue, data, serv_distr):
    global users
    global current
    
    data.num_departures += 1
    data.average_utilization += users*(time - data.time_last_event)
    data.time_last_event = time
    
    users -= 1
    
    # this means that a client is waiting and can be processed (so it will departure after service_time)
    try:
        current = queue.get(timeout=0)
        data.average_delay_time += (time - current.arrival_time)
        
        service_time = generate_service_time(distr=serv_distr)                
        FES.put((time + service_time, 'departure'))
        
    except Empty:
        # the queue is empty
        current = None # this is for saying that the server is back idle
    data.av_delays.append((time, data.average_delay_time/data.num_departures))
    data.av_no_cust.append((time, data.average_utilization/time))

# Event Loop
def simulate(lambd, queue_lenght, serv_distr):

    # Initialization
    data = Measure()
    time = 0

    FES = PriorityQueue()
    queue = Queue(queue_lenght)

    FES.put((time, "arrival"))

    # Event Loop
    while time < SIM_TIME:
        if FES.empty():
            break

        (time, event_type) = FES.get()
        
        if event_type == 'arrival':
            arrival(time, FES, queue, data, lambd, serv_distr)
        elif event_type == 'departure':
            departure(time, FES, queue, data, serv_distr)

    # end of the simulation
    average_delay = data.average_delay_time/data.num_departures
    average_no_cust = data.average_utilization/time
    
    return average_delay, average_no_cust, data

def plot_analitical_empirical_comparation(av_no_cust, data):
    em_waiting = np.mean([delay for _,delay in data.av_delays])
    an_waiting = (average_no_cust / lambd) - 1
    plt.figure(figsize=(12,6))
    plt.plot([time for time,_ in data.av_delays], [delay for _,delay in data.av_delays], 
             linestyle='-', label=f'Ciao 1')
    plt.axhline(y=em_waiting, color='green', linestyle='--', label='Empirical Mean')
    plt.axhline(y=an_waiting, color='red', linestyle='--', label='Analitical Mean')
    plt.ylabel('Delay', fontsize=14)
    plt.xlabel('Times Departure', fontsize=14)
    plt.title('Delays and Times departure', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()
    
    em_no_cust = np.mean([n_cust for _,n_cust in data.av_no_cust])
    
    plt.figure(figsize=(12,6))
    plt.plot([time for time,_ in data.av_no_cust], [n_cust for _,n_cust in data.av_no_cust])
    plt.axhline(y=em_no_cust, color='green', linestyle='--', label='Empirical Number of Customers')
    plt.axhline(y=av_no_cust, color='red', linestyle='--', label='Analitical Number of Customers')
    plt.ylabel('Number of Customers', fontsize=14)
    plt.xlabel('Times Departure', fontsize=14)
    plt.title('Number of Customers and Times departure', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()
    
def plot_r_xs(data):
    delays = [delay for _,delay in data.av_delays]
    ran = range(1,len(delays))
    av = np.mean(data.av_delays)  
    x_ks = [np.mean(delays[k:]) for k in ran]
    r_k = [(x_k - av) / av for x_k in x_ks]
    
    plt.figure(figsize=(12,6))
    plt.plot(ran, x_ks, label=f'x_k')
    plt.plot(ran, r_k, label=f'r_k')
    plt.ylabel('Average', fontsize=14)
    plt.xlabel('k', fontsize=14)
    plt.title('Average and k', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()
    

datas = [] # this is for performing some plot after the simulation
# we run a simulation for each arrival_lambdas
for lambd in arrival_lambdas:
    
    users = 0
    current = None
    print(f'\n\nStarting simulation with arrival lambda = {lambd}')
    average_delay, average_no_cust, data = simulate(queue_lenght=MAX_QUEUE_CAPACITY, 
                                                    lambd=lambd, serv_distr='exponential')
    
    #plot_analitical_empirical_comparation(average_no_cust, data)
    plot_r_xs(data)
    