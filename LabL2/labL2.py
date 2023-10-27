import random
from enum import Enum 

N_DOCTOR = 5
ARRIVAL_LAMBDA = 10

class Urgency(Enum):
    RED = 1
    YELLOW = 2
    GREEN = 3
    

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
        
    def compareTo(self,other):
        """
        If returns something < 0 it means that the other is less urgent.
        If returns something > 0 it means that the other is more urgent.
        If returns something == 0 it means that the other has the same urgency.

        Args:
            other (Client): It's the client to compare this client with

        Returns:
            int: Returns the difference between the enum values
        """
        compare = self.urgency.value - other.urgency.value # this will be 0 only if they have the same urgency
        if compare == 0: # if they have the same urgency we ordered based on their arrival_time
            compare = self.arrival_time - other.arrival_time
        return compare
        
class PriorityQueue: 
    """
    Class where are stored the events that are ment to happen
    """
    def __init__(self):
        self.events = [] # this is a list of events in the form: (time,type)
    
    def put(self,el):
        """
        Add an element to the PriorityQueue implementing the insertion sort algorithm.
        This will maintain the list sorted

        Usage:
            Use this method to add event to the ProrityQueue
        """
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
        """
        This method returns the event that is the nearest in time

        Returns:
            Tuple[float,str]: Time of occurence, type of event
        """
        return self.events.pop(0)
    
class Queue():
    """Class where are stored the clients waiting
    """
    def __init__(self):
        self.queue = []
        
    def enqueue(self,new_client):
        """Add a Client to the queue implementing the the insertion sort algorithm based on their urgency.
        This means that the most urgent customer who has been waiting the longest would be the first in the queue

        Args:
            new_client (Client): The clients that has to be enqueued
        """
        index = None
        for i,client in enumerate(self.queue):
            # this means that the new is equal or more urgent
            if client.compareTo(new_client) >= 0: # if it's == 0 it means that they arrived at the same time, and it is very unlikely
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
    def __init__(self,arr_lambda, n_server, Narr=None,Ndep=None,NAverageUser=None,OldTimeEvent=None,AverageDelay=None,Dropped=None):
        # input parameters
        self.arr_lambda = arr_lambda
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
        
    def arrival(self, time, FES, queue):
        pass # TODO
    
    def departure(self, time, FES, queue):
        pass # TODO