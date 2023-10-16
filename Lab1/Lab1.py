import random

# arrival process of customers is a poisson process a rate lambda in [5,7,9,10,12,15] (i.e. interarrivals are exponentially distributed with average 1/lambda)
# this means that the arrival process is not a singular exponential distribution but in fact are 6 different distribution

arrival_lambdas = [5,7,9,10,12,15]

# service times are exponentially distributed with average 1
SERVICE = 1.0 

# this is not needed since we have a fixed number of clients to be served
SIM_TIME = 200

# Maximum capacity of the queue
MAX_QUEUE_CAPACITY = 100

# 
TYPE1 = 'Client1'
TYPE2 = 'Client2'

k = 10
n_clients = 1000 # fixed number of clients

# half a page of a pdf with the results and a comment
# what will be the condition which I choose

# evaluate the average delay and the dropping probability

# this class will store the info we are interested in about the simulation
class Measure:
    def __init__(self,Narr,Ndep,NAverageUser,OldTimeEvent,AverageDelay,Dropped):
        self.num_arrivals = Narr
        self.num_departures = Ndep
        self.average_utilization = NAverageUser
        self.time_last_event = OldTimeEvent
        self.average_deley_time = AverageDelay
        self.num_dropped = Dropped

# this class is the representation of a client
class Client:
    def __init__(self,Type,arrival_time):
        self.type_ = Type
        self.arrival_time = arrival_time
        
# here we store the clients in the queue
# this implements a FIFO structures (first in - first out)
class Queue:
    def __init__(self):
        self.queue = []
        
    def pop(self): # this will return and delete the first element from the queue
        return self.queue.pop(0)
    
    def append(self,client):
        if not self.is_full():
            self.queue.append(client)
            
    def is_full(self):
        return len(self.queue) >= MAX_QUEUE_CAPACITY
        
# here we store the events that are ment to appen
class PriorityQueue: # this is a list of events in the form: (time,type)
    def __init__(self):
        self.events = []
    
    def put(self,el):
        index = None
        for i,event in enumerate(self.events):
            if el[0] < event[0]:
                index = i
                break
        if index is not None:
            self.events = self.events[:index] + [el] + self.events[index:]
        else:
            self.events.append(el)
        
    def get(self):
        return self.events.pop(0)

def arrival(time, FES, queue, data):
    global users, servers
    
    # Create data for measuring simulation
    data.num_arrivals += 1
    data.average_utilization += users*(time - data.time_last_event)
    data.time_last_event = time
    
    # compute the inter-arrival time Tia for next client
    # this is not needed if I schedule all arrival from the start
    #inter_arrival = random.expovariate(1.0/ARRIVAL) # TODO
    
    # schedule an arrival at time Tcurr + Tia
    # FES.put((time + inter_arrival, "arrival"))
    
    # Create a record for the client
    client = Client(TYPE1,time)
    
    # Insert the record in the queue
    if not queue.is_full():
        users += 1
        #print(f'Clients in queue: {len(queue.queue)} clients in the system: {users} available servers: {servers} full = False')
        queue.append(client)
    else:
        # print(f'Clients in queue: {len(queue.queue)} clients in the system: {users} available servers: {servers}')
        data.num_dropped += 1
    
    # If the server is idle -> make the server busy
    # this means that a client can be removed from the queue (?)
    if servers >= 1:
        client = queue.pop()
        data.average_deley_time += (time - client.arrival_time)
        # determine the service time Ts
        service_time = random.expovariate(1.0 / SERVICE)
        servers -= 1 # this is for saying that the server is busy
        
        # schedule the end of service at time Tcurr + Ts
        FES.put((time + service_time, 'departure'))

def departure(time, FES, queue, data):
    global users, servers
    
    data.num_departures += 1
    data.average_utilization += users*(time - data.time_last_event)
    data.time_last_event = time
    
    users -= 1
    
    #if servers < 10:
    servers += 1 # this is for saying that the server is back idel
    #print(servers)
    
    # this means that a client is waiting and can be processed (so it will live after service_time)
    if len(queue.queue) > 0:
        client = queue.pop()
        
        data.average_deley_time += (time - client.arrival_time)
        
        service_time = random.expovariate(1.0/SERVICE)
        
        FES.put((time + service_time, 'departure'))
        
    
    #print(f'Clients in queue: {len(queue.queue)} clients in the system: {users} available servers: {servers} departure')

# Event Loop

# Initialization
data = Measure(0,0,0,0,0,0)
users = 0
servers = k

FES = PriorityQueue()
queue = Queue()

# Initialize cumulative arrivals for each lambda
cumulative_arrivals = [0] * len(arrival_lambdas)

# Schedule arrival events for each λ value
for i, lambd in enumerate(arrival_lambdas):
    for _ in range(n_clients // len(arrival_lambdas)):
        # Generate inter-arrival time for this type
        interarrival_time = random.expovariate(1.0 / lambd)
        
        # Schedule the arrival event with the corresponding inter-arrival time
        FES.put((cumulative_arrivals[i] + interarrival_time, 'arrival'))
        
        # Update cumulative arrivals for this type
        cumulative_arrivals[i] += interarrival_time

# FES.put((0, "arrival"))

# we have a fixed number of clients so the simulation will run untils they ends
# Event Loop
events = []
while data.num_departures < n_clients:
    if not FES.events:
        break

    (time, event_type) = FES.get()
    events.append((time,event_type))
    
    if event_type == 'arrival':
        arrival(time, FES, queue, data)
    elif event_type == 'departure':
        departure(time, FES, queue, data)
        
    if True:
        True

# end of the simulation
average_delay = data.average_deley_time/data.num_departures
average_no_cust = data.average_utilization/time

print(f'Total number of clients: {data.num_arrivals}\nMax queue capacity: {MAX_QUEUE_CAPACITY}')
print(f'Average time spent waiting: {average_delay:.2f}s\nAverage number of customers in the system: {average_no_cust:.2f}')
print(f'Dropped clients: {data.num_dropped} (Dropping probability: {data.num_dropped / n_clients * 100:.2f}%)')
# print(f'Number of clients in the system at the end: {n_clients - data.num_departures}')