import random
from enum import Enum 

N_DOCTOR = 5
ARRIVAL_LAMBDA = 10
SERVICE_LAMBDA = 10
SIM_TIME = 2_000

class Urgency(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3
    
    def compare_to(self,other: 'Urgency'):
        """
        If returns something < 0 it means that the other is less urgent.
        If returns something > 0 it means that the other is more urgent.
        If returns something == 0 it means that the other has the same urgency.

        Args:
            other (Urgency): It's the urgency to compare this urgency with

        Returns:
            int: Returns the difference between the enum values
        """
        return self.value - other.value    

class Client():
    def __init__(self, arrival_time, urgency=None):
        self.urgency = self.setUrgency() if urgency is None else urgency
        self.arrival_time = arrival_time
        
    def setUrgency(self):
        """
        Randomly assigns an urgency level to a customer based on specified probabilities.

        Returns:
            str: The assigned urgency level, which can be 'Red', 'Yellow', or 'Green'.

        Probability Distribution:
            - 'Red': 1/6 (approximately 16.67%)
            - 'Yellow': 1/3 (approximately 33.33%)
            - 'Green': 1/2 (approximately 50%)

        Usage:
            Use this method to assign urgency levels to customers in a queue system.
        """
        rnd = random.uniform(0,1)
        if rnd < 1/ 6:
            return Urgency.RED
        elif rnd < 1/2: # this means that the Yellow probabilities is 1/6 - 1/2 = 1/3
            return Urgency.YELLOW
        else: # the remaining samples (1/2 of it) fall under this condition
            return Urgency.GREEN
        
    def compare_to(self,other: 'Client'):
        """
        If returns something < 0 it means that the other is less urgent.
        If returns something > 0 it means that the other is more urgent.
        If returns something == 0 it means that the other has the same urgency.

        Args:
            other (Client): It's the client to compare this client with

        Returns:
            int: Returns an int that represent if one is urgent than the other
        """
        compare = self.urgency.compare_to(other.urgency) # this will be 0 only if they have the same urgency
        if compare == 0: # if they have the same urgency we ordered based on their arrival_time
            compare = self.arrival_time - other.arrival_time
        return compare
    
class Event:
    def __init__(self, time:float, type_:str, urgency=None):
        self.time = time
        self.type_ = type_
        self.urgency = urgency
        self.active = True
    
    def deactive(self):
        self.active = False
        
    def is_active(self):
        return self.active
    
    def compare_to(self,other: 'Event'):
        """
        If returns something < 0 it means that the other is after.
        If returns something > 0 it means that the other is before.
        If returns something == 0 it means that they happen in the same exact time (very unlikely).

        Args:
            other (Event): It's the other event to compare this event with

        Returns:
            int: Returns an int that represent if one is happening before of the other
        """
        return self.time - other.time
    
    def is_early_than(self,other: 'Event'):
        return self.compare_to(other) < 0
        
class PriorityQueue: 
    """
    Class where are stored the events that are ment to happen
    """
    def __init__(self):
        self.events = [] # this is a list of events in the form: (time,type)
    
    def put(self,new_event: Event):
        """
        Add an element to the PriorityQueue implementing the insertion sort algorithm.
        This will maintain the list sorted

        Usage:
            Use this method to add event to the ProrityQueue
        """
        index = None
        for i,event in enumerate(self.events):
            if event.is_early_than(new_event):
                index = i
                break
        if index is not None:
            self.events.insert(i,new_event)
        else:
            self.events.append(new_event)
        
    def get(self):
        """
        This method returns the active event that is the nearest in time

        Returns:
            Tuple[float,str]: Time of occurence, type of event
        """
        while True: # this will return only active events
            if len(self.events) > 0:
                event = self.events.pop(0)
                if event.is_active():
                    return event
            else:
                return None # it should neve occurs, but let's prevent some error
    
    def stop_service(self):
        """It finds a departure with a low level of urgency

        Returns:
            _type_: _description_
        """
        for i,event in enumerate(self.events):
            pass # TODO
    
class Queue():
    """Class where are stored the clients waiting
    """
    def __init__(self):
        self.queue = []
        
    def enqueue(self,new_client: Client):
        """Add a Client to the queue implementing the the insertion sort algorithm based on their urgency.
        This means that the most urgent customer who has been waiting the longest would be the first in the queue

        Args:
            new_client (Client): The clients that has to be enqueued
        """
        index = None
        for i,client in enumerate(self.queue):
            # this means that the new is equal or more urgent
            if client.compare_to(new_client) >= 0: # if it's == 0 it means that they arrived at the same time, and it is very unlikely
                index = i
                break
                    
        if index is None:
            self.queue.append(new_client)
        else:
            self.queue.insert(index,new_client)
            
    def dequeue(self):
        """Returns the most urgent client that is waiting the most

        Returns:
            Client: The most urgent customer who has been waiting the longest
        """
        return self.queue.pop(0)
    
class System:
    """Class that store the system variables (so I can avoid using global variables) and measurement parameters
    """
    def __init__(self,arr_lambda: int, serv_lambda: int, n_server: int, 
                 Narr=None,Ndep=None,NAverageUser=None,OldTimeEvent=None,AverageDelay=None,Dropped=None):
        # input parameters
        self.arr_lambda = arr_lambda
        self.serv_lambda = serv_lambda
        self.n_server = n_server
        
        # system variables (so I can avoid using global variables)
        servers = 0
        client = 0
        
        # Measurements paramenters
        self.num_arrivals = 0 if Narr is None else Narr
        self.num_departures = 0 if Ndep is None else Ndep
        self.average_utilization = 0 if NAverageUser is None else NAverageUser
        self.time_last_event = 0 if OldTimeEvent is None else OldTimeEvent
        self.average_delay_time = 0 if AverageDelay is None else AverageDelay
        self.num_dropped = 0 if Dropped is None else Dropped
        
    def arrival(self, time: float, FES: PriorityQueue, queue: Queue):
        """This method calculate what happen when a new client arrives in the system.
        In this case the major things that should happen are:
            - we can schedule the new arrival
            - if there is at least one server available, we can start a new service
            - if clients less urgent are been served, their service must be paused to serve the new arrived
            - if there are no server available, the client must be enqueued

        Args:
            time (float): A float number representing the time
            FES (PriorityQueue): This is the space of events (the event that are ment to happen)
            queue (Queue): This is the class that store the clients that are waiting
        """
        
        # measuring the simulation
        self.num_arrivals += 1
        self.average_utilization += self.users * (time - self.time_last_event)
        self.time_last_event = time
        
        # compute the inter-arrival time Tia for next client
        inter_arrival = random.expovariate(self.arr_lambda)
        
        FES.put(Event(time + inter_arrival, 'arrival'))
        
        # Create a record for the client
        client = Client(arrival_time=time) # the urgency is automatically set
        
        # storing that a client has arrived
        self.clients += 1
        
        # If the server is idle -> make the server busy
        if self.servers >= 1:
            # determine the service time Ts
            service_time = random.expovariate(self.serv_lambda)
            self.servers -= 1 # this is for saying that the server is busy
            
            # schedule the end of service at time Tcurr + Ts
            FES.put(Event(time + service_time, 'departure', urgency=client.urgency))
        else:
            queue.enqueue(client)
    
    def departure(self, time, FES, queue):
        pass # TODO
    
def simulate():
    
    # Initialization
    system = System(arr_lambda=ARRIVAL_LAMBDA, 
                    serv_lambda=SERVICE_LAMBDA, 
                    n_server=N_DOCTOR)
    time = 0
    
    FES = PriorityQueue()
    queue = Queue()
    
    FES.put(Event(time,'arrival'))
    
    # Event loop
    while time < SIM_TIME:
        if not FES.events:
            break
        
        event = FES.get()
        time = event.time
        
        if event.type_ == 'arrival':
            system.arrival(time,FES,queue)
        elif event.type_ == 'departure':
            system.departure(time,FES,queue)