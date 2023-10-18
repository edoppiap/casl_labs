import random
import matplotlib.pyplot as plt

# arrival process of customers is a poisson process a rate lambda in [5,7,9,10,12,15] (i.e. interarrivals are exponentially distributed with average 1/lambda)
# this means that the arrival process is not a singular exponential distribution but in fact are 6 different distribution

arrival_lambdas = [5,7,9,10,12,15]

# service times are exponentially distributed with average 1
SERVICE = 1.0

SIM_TIME = 2_000

# Maximum capacity of the queue
MAX_QUEUE_CAPACITY = 1000

# 
TYPE1 = 'Client1'
TYPE2 = 'Client2'

k = 10 # number of servers

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
        self.average_delay_time = AverageDelay
        self.num_dropped = Dropped

# this class is the representation of a client
class Client:
    def __init__(self,Type,arrival_time):
        self.type_ = Type
        self.arrival_time = arrival_time
        
# here we store the clients in the queue
# this implements a FIFO structures (first in - first out)
class Queue:
    def __init__(self, capacity):
        self.queue = []
        self.capacity = capacity
        
    def get(self): # this will return and delete the first element from the queue
        return self.queue.pop(0)
    
    def append(self,client):
        if not self.is_full():
            self.queue.append(client)
            
    def is_full(self):
        return len(self.queue) >= self.capacity
    
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
            self.events.insert(i,el)
        else:
            self.events.append(el)
        
    def get(self):
        return self.events.pop(0)

def arrival(time, FES, queue, data, lambd):
    global users
    global servers
    
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
    if servers >= 1:
        # determine the service time Ts
        service_time = random.expovariate(SERVICE)
        servers -= 1 # this is for saying that the server is busy
        
        # schedule the end of service at time Tcurr + Ts
        FES.put((time + service_time, 'departure'))
    else: # add the client to the queue
        # Insert the record in the queue
        if not queue.is_full():
            queue.append(client)
            #print(f'Clients in queue: {len(queue.queue)} clients in the system: {users} available servers: {servers} full = False')
        else:
            #print(f'Clients in queue: {len(queue.queue)} clients in the system: {users} available servers: {servers} full = True')
            data.num_dropped += 1
            users -= 1

def departure(time, FES, queue, data):
    global users
    global servers
    
    data.num_departures += 1
    data.average_utilization += users*(time - data.time_last_event)
    data.time_last_event = time
    
    users -= 1
    
    # this means that a client is waiting and can be processed (so it will live after service_time)
    if len(queue.queue) > 0:
        client = queue.get()
        
        data.average_delay_time += (time - client.arrival_time)
        
        service_time = random.expovariate(SERVICE)
                
        FES.put((time + service_time, 'departure'))
    else:
        servers +=1 # this is for saying that the server is back idle
        
    
    #print(f'Clients in queue: {len(queue.queue)} clients in the system: {users} available servers: {servers} departure')

def simulate(lambd, queue_lenght):
    
    # Event Loop

    # Initialization
    data = Measure(0,0,0,0,0,0)
    time = 0

    FES = PriorityQueue()
    queue = Queue(queue_lenght)

    FES.put((time, "arrival"))

    # we have a fixed number of clients so the simulation will run untils they ends
    # Event Loop
    while time < SIM_TIME:
        if not FES.events:
            break

        (time, event_type) = FES.get()
        
        if event_type == 'arrival':
            arrival(time, FES, queue, data, lambd)
        elif event_type == 'departure':
            departure(time, FES, queue, data)

    # end of the simulation
    average_delay = data.average_delay_time/data.num_departures
    average_no_cust = data.average_utilization/time

    print(f'Number of clients \ number of departures: {data.num_arrivals} \ {data.num_departures}')
    print(f'Average time spent waiting: {average_delay:.4f}s\nAverage number of customers in the system: {average_no_cust:.2f}')
    print(f'Dropped clients: {data.num_dropped} (Dropping probability: {data.num_dropped / data.num_arrivals * 100:.2f}%)')
    print(f'Number of clients in the system at the end: {data.num_arrivals - data.num_departures}')
    
    return average_delay,average_no_cust,data

datas = []
for lambd in arrival_lambdas:
    
    users = 0
    servers = k
    print(f'\n\nStarting simulation with arrival lambda = {lambd}')
    datas.append(simulate(queue_lenght=MAX_QUEUE_CAPACITY, lambd=lambd))
    
plt.figure(figsize=(12,6))

plt.plot([lambd for lambd in arrival_lambdas], [data[1] for data in datas], label='Number', color='blue', marker='o', linestyle='-', markersize=5)

plt.ylabel('Average Number of Customer')
plt.xlabel('Lambdas')
plt.xticks(range(arrival_lambdas[0],arrival_lambdas[-1]+1))
plt.title('Average number of clients in the system and lambdas')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,6))

plt.plot([lambd for lambd in arrival_lambdas], [(data[2].num_dropped / data[2].num_arrivals) for data in datas], label = 'Number', color='red', marker='o', linestyle='-', markersize=5)

plt.ylabel('Dropping probability')
plt.xlabel('lambdas')
plt.xticks(range(arrival_lambdas[0],arrival_lambdas[-1]+1))
plt.title('Dropping probabilities and lambdas')
plt.grid(True)
plt.show()