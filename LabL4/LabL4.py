import random
import matplotlib.pyplot as plt
from queue import PriorityQueue, Queue, Full, Empty
import numpy as np
from tqdm import tqdm
from scipy.stats import t

# arrival process of customers is a poisson process a rate lambda in [5,7,9,10,12,15] (i.e. interarrivals are exponentially distributed with average 1/lambda)
# this means that the arrival process is not a singular exponential distribution but in fact are 6 different distribution

#seeds = np.arange(2,52,5)
seeds = [54651, 54951651, 162198441, 1564841, 54954, 45684123]
#np.random.seed(42)

arrival_lambdas = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999]
#arrival_lambdas = [0.999]

service_distrs = ['exponential']

# service times are exponentially distributed with average 1
SERVICE = 1.0

# the simulation time must be great enough to capture the saturaion of the system
MAX_SIM_TIME = 100_000

# Maximum capacity of the queue
MAX_QUEUE_CAPACITY = 1000

# 
TYPE1 = 'Client1'
TYPE2 = 'Client2'

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

# this class will store the info we are interested in about the simulation
class Measure:
    def __init__(self,lambd=1, serv_distr = None, Narr=0,Ndep=0,NAverageUser=0,OldTimeEvent=0,AverageDelay=0,Dropped=0):
        self.lamdb = lambd
        self.serv_distr = serv_distr
        self.num_arrivals = Narr
        self.num_departures = Ndep
        self.average_utilization = NAverageUser
        self.time_last_event = OldTimeEvent
        self.average_delay_time = AverageDelay
        self.num_dropped = Dropped
        self.delays = []
        self.av_delays = []
        self.lengths = []
        
    def get_conf_intervals(self):
        return calculate_confidence_interval([delay for _,delay in self.av_delays], conf=.97)

# this class is the representation of a client
class Client:
    def __init__(self,Type,arrival_time):
        self.type_ = Type
        self.arrival_time = arrival_time
        
def generate_service_time(lam: float = 1, distr: str = 'deterministic'):
    if distr == 'deterministic':
        return 1
    elif distr == 'exponential':
        return random.expovariate(lam)
    elif distr == 'hyper':
        pass

def arrival(time, FES, queue, data, lambd,  serv_distr):
    global users
    global current
    
    data.num_arrivals += 1
    data.average_utilization += users*(time - data.time_last_event)
    data.time_last_event = time
    
    # compute the inter-arrival time Tia for next client
    inter_arrival = random.expovariate(lambd)
    
    data.lengths.append((time, len(queue.queue)))
    
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
    
    data.lengths.append((time, len(queue.queue)))
    
    users -= 1
    
    # this means that a client is waiting and can be processed (so it will departure after service_time)
    try:
        current = queue.get(timeout=0)
        data.average_delay_time += (time - current.arrival_time)
        
        data.delays.append((time,time - current.arrival_time))
        
        service_time = generate_service_time(distr=serv_distr)                
        FES.put((time + service_time, 'departure'))
        
    except Empty:
        # the queue is empty
        current = None # this is for saying that the server is back idle
    data.av_delays.append((time,data.average_delay_time/data.num_departures))

# Event Loop
def simulate(lambd, queue_lenght, serv_distr):
    
    #------------------------------------#
    # 1) Compute a very long simulation
    # 2) Split the simulation into batches
    time_batch = MAX_SIM_TIME // 20
    
    pbar = tqdm(total=MAX_SIM_TIME,
                    desc=f'Simulating with n_server = {1}, arr_lambda = {lambd} and sim_time = {MAX_SIM_TIME}',
                    bar_format='{l_bar}{bar:30}{n:.0f}s/{total}s [{elapsed}<{remaining}, {rate_fmt}]')
        
    # INITIALIZATION
    time = 0
    datas = []

    FES = PriorityQueue()
    queue = Queue(queue_lenght)

    FES.put((time, "arrival"))
    
    next_batch = time_batch
    
    # Event Loop
    while time < MAX_SIM_TIME:
        if FES.empty():
            break
        batch_data = Measure(serv_distr=serv_distr, lambd=lambd)
        
        #----------------------------
        # Simulate a single batch
        #
        while time < next_batch:
            
            (new_time, event_type) = FES.get()
            
            if time < MAX_SIM_TIME: # to prevent a warning to appear
                pbar.update(new_time - time)
                
            time = new_time
            
            if event_type == 'arrival':
                arrival(time, FES, queue, batch_data, lambd, serv_distr)
            elif event_type == 'departure':
                departure(time, FES, queue, batch_data, serv_distr)
                
        #------------------------------------------------------------------------#
        # 3) For each interval, compute an estimation of the quantity under study
        datas.append(batch_data)
                
        next_batch += time_batch

    print(len(datas))
    pbar.close()   
    return datas
    
    

datas = [] # this is for performing some plot after the simulation
# we run a simulation for each arrival_lambdas
for service_distr in service_distrs:
    for lambd in arrival_lambdas:
        
        users = 0
        current = None
        data = simulate(queue_lenght=MAX_QUEUE_CAPACITY, lambd=lambd, serv_distr=service_distr)
        
        
        plt.figure(figsize=(12,6))
        for i,data_batch in enumerate(data):
            #print(data_batch.av_delays)
            mean,interval,acc = data_batch.get_conf_intervals()
            times = [time for time, _ in data_batch.av_delays]
            plt.plot(times, [delay for _,delay in data_batch.av_delays], 
                     linestyle='-', label=f'batch_n={i+1}')
            plt.plot(times, [mean] * len(times), 
                     linestyle='-')
        plt.ylabel('Delay', fontsize=14)
        plt.xlabel('Times Departure', fontsize=14)
        plt.title('Delays and Times departure', fontsize=16)
        plt.grid(True)
        plt.legend()
        plt.show()

for data in datas:
    em_waiting = np.mean([delay for _,delay in data[3]])
    an_waiting = (data[2] / data[1]) - 1
    plt.figure(figsize=(12,6))
    plt.plot([time for time,_ in data[3]], [delay for _,delay in data[3]], linestyle='-', label=f'ser_dstr={data[0]}, lamdb={data[1]}')
    plt.axhline(y=em_waiting, color='green', linestyle='--', label='Empirical Mean')
    plt.axhline(y=an_waiting, color='red', linestyle='--', label='Analitical Mean')
    plt.ylabel('Delay', fontsize=14)
    plt.xlabel('Times Departure', fontsize=14)
    plt.title('Delays and Times departure', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()