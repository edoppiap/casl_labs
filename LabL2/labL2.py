import random
from enum import Enum

N_DOCTOR = 5
ARRIVAL_LAMBDA = 5
SERVICE_LAMBDA = 1
SIM_TIME = 15

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

class Client:
    def __init__(self, arrival_time, urgency: Urgency = None,
                 paused_time: float = None, # this will store the time in which the client has been put on pause
                 time_left: float = None # this will store the time until the end of service (if service has paused)
                ):
        self.urgency = self.setUrgency() if urgency is None else urgency
        self.arrival_time = arrival_time
        self.paused_time = paused_time
        self.time_left = time_left # this is None if the clients has neven been put on pause
        
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
        if compare == 0: # they have the same urgency
            if self.paused_time is None: # we ordered based on their arrival_time
                compare = self.arrival_time - other.arrival_time
            else: # we ordered based on their paused_time
                compare = self.paused_time - other.paused_time
        return compare
    
    def is_less_urgent_than(self,other: 'Client'):
        return self.urgency.compare_to(other.urgency) > 0 # it means that other is more urgent
    
    def is_red(self):
        return self.urgency == Urgency.RED
    
class Event:
    def __init__(self, time:float, type_:str, client: Client = None):
        self.time = time
        self.type_ = type_
        self.client = client
        self.active = True
    
    def deactivate(self):
        self.active = False
        
    def is_active(self):
        return self.active
    
    def compare_to(self,other: 'Event'):
        """
        If returns something < 0 it means that the other is after.
        If returns something > 0 it means that the other is before.
        If returns something == 0 it means that they happen in the same exact time (very unlikely, I'm not handling this case).

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
        
    def has_no_events(self):
        return not self.events
    
    def put(self,new_event: Event):
        """
        Add an element to the PriorityQueue implementing the insertion sort algorithm.
        This will maintain the list sorted

        Usage:
            Use this method to add event to the ProrityQueue
        """
        index = None
        for i,event in enumerate(self.events):
            if new_event.is_early_than(event):
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
            Event: An object representing an event 
        """
        while True: # this will return only active events
            if len(self.events) > 0:
                event = self.events.pop(0)
                if event.is_active():
                    return event
            else:
                return None # it should neve occurs, but let's prevent some error
    
    def stop_service(self, client: Client):
        """It finds the departure schedule for this client and cancel it

        Returns:
            _type_: _description_
        """
        for event in self.events:
            if event.client == client and event.is_active():
                event.deactivate()
                return event.time # it returns the time when the events would take place
    
class Queue:
    """Class where are stored the clients waiting
    """
    def __init__(self):
        self.queue = []
        
    def has_waiting_clients(self):
        return len(self.queue) > 0
    
    def has_red_waiting(self):
        return self.has_waiting_clients() and self.queue[0].urgency == Urgency.RED
        
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

class ServersList:
    def __init__(self, n_servers):
        self.servers = [] # this is a list of clients
        self.capacity = n_servers
        
    def __len__(self):
        return len(self.servers)
    
    def departure(self,client):
        self.servers.remove(client)
    
    def is_server_available(self):
        return len(self) < self.capacity
    
    def put(self,new_client):
        """This method ensure an order to this list in order to check the less urgent client first. 
        In this way when a RED arrives a GREEN is paused and not a YELLOW

        Args:
            new_client (Client): A client that has started a service
        """
        index = None
        for i,client in enumerate(self.servers):
                if new_client.is_less_urgent_than(client):
                    index = i
                    break
        if index is None:
            self.servers.append(new_client)
        else:
            self.servers.insert(index,new_client)
    
    def start_service_if_possible(self, new_client: Client):
        """Method that add a client to the service if there is a server available 
        or there is at least one doctor that is serving a less urgent client

        Args:
            new_client (Client): The client we want to start serve

        Returns:
            True | None | Client: It returns True if we start a service without confict, 
            None if we can't start a service, and a Client if we found one less urgent
        """
        if len(self) < self.capacity: # this means that there is at least one doctor available
            self.put(new_client)
            return True
        else: # we need to check if there is at least one doctor that serve a client less urgent than this one
            paused_client = None
            for i,client in enumerate(self.servers):
                if new_client.is_red() and not client.is_red():
                    paused_client = self.servers.pop(i)
                    self.put(new_client)
                    break # we can exit the loop, we found a less urgent client
            return paused_client # it is None if we do not find any less urgent client
    
class System:
    """Class that store the system variables (so I can avoid using global variables) 
    and it is only an extention of the measurement parameters class
    """
    def __init__(self,arr_lambda: int, serv_lambda: int, n_server: int, 
                 Narr=None,Ndep=None,NAverageUser=None,OldTimeEvent=None,AverageDelay=None, Npau=None):
        # input parameters
        self.arr_lambda = arr_lambda
        self.serv_lambda = serv_lambda
        self.n_server = ServersList(n_server)
        
        # system variables (so I can avoid using global variables)
        self.servers = ServersList(n_servers = n_server)
        self.clients = 0 # clients in the system
        
        # Measurements paramenters
        self.num_arrivals = 0 if Narr is None else Narr
        self.num_departures = 0 if Ndep is None else Ndep
        self.average_utilization = 0 if NAverageUser is None else NAverageUser
        self.time_last_event = 0 if OldTimeEvent is None else OldTimeEvent
        self.average_delay_time = 0 if AverageDelay is None else AverageDelay
        self.num_paused = 0 if Npau is None else Npau
        #self.num_dropped = 0 if Dropped is None else 
        
    # Utils
    def most_urgent_waiting(self, queue: Queue, paused: Queue):
        """This method check if there is a red client waiting and in this case it returns it. 
        Then check if there is a client paused and in this case returns it.
        Then check if there is any other type of client waiting and returns it.
        In case both queue and paused are empty it returns None.
        In this way it prioritize the red client waiting and the green/yellow paused client remain in pause
        but if there are no red clients waiting it prioritize the paused client

        Args:
            queue (Queue): This is the list of clients waiting
            paused (Queue): This is the list of clients that has been put on pause

        Returns:
            Client: The most urgent client to serve
        """
        if queue.has_red_waiting():
            return queue.dequeue()
        elif paused.has_waiting_clients():
            return paused.dequeue()
        elif queue.has_waiting_clients():
            return queue.dequeue()
        return None
        
    def arrival(self, time: float, FES: PriorityQueue, queue: Queue, paused: Queue):
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
            paused (Queue): This is the class that store the clients that has been put on pause
        """
        
        # measuring the simulation
        self.num_arrivals += 1
        self.average_utilization += self.clients * (time - self.time_last_event)
        self.time_last_event = time
        
        # compute the inter-arrival time Tia for next client
        inter_arrival = random.expovariate(self.arr_lambda)
        
        FES.put(Event(time + inter_arrival, 'arrival')) # the arrival event have no client memorized
        
        # Create a record for the client
        client = Client(arrival_time=time) # the urgency is automatically set
        
        # storing that a client has arrived
        self.clients += 1
        
        result = self.servers.start_service_if_possible(client)
        
        if result == True:
            # it means that there were a server available and the service has started
            # determine the service time Ts
            service_time = random.expovariate(self.serv_lambda)
            
            # schedule the end of service at time Tcurr + Ts
            FES.put(Event(time + service_time, 'departure', client=client)) # the departure event has to memorize the client in case we have to stop the service
        elif result == None:
            # it means that there were no server available to process this client
            queue.enqueue(client)
        elif isinstance(result, Client): # it means that the server started processing the new more urgent client and return a paused client
            self.num_paused += 1 # store the number of clients that has been put on pause
            
            # cancel the scheduled departure of the less urgent client
            departure_time = FES.stop_service(result)
            if departure_time is not None:
                result.paused_time = time # we put the time in which the client has been put on old to maintain the list sorted based on that
                result.time_left = departure_time - time # this is the remaining time for the processing
                paused.enqueue(result) # we put the client in the queue for the paused clients 
            
            # schedule the departure of the more urgent client
            service_time = random.expovariate(self.serv_lambda)
            
            FES.put(Event(time + service_time, 'departure', client=client))
    
    def departure(self, time: float, FES: PriorityQueue, queue: Queue, paused: Queue, 
                  client: Client):
        """This method calculate what happens when a client leave the system.
        The major things that should happen are:
        - it can be processed a client that has been put on pause
        - or it can be processed a client that has been waiting in the queue
        - it been decided based on the urgency of the two clients
        - or the server can be put back on idle

        Args:
            time (float): A float number representing the time
            FES (PriorityQueue): This is the space of events (the event that are ment to happen)
            queue (Queue): This is the class that store the clients that are waiting
            paused (Queue): This is the class that store the clients that has been put on pause
            client (Client): The client that has departured
        """
        self.num_departures += 1
        self.average_utilization += self.clients * (time - self.time_last_event)
        self.time_last_event = time
        
        self.clients -= 1
        self.servers.departure(client) # we can free the server        
        
        client = self.most_urgent_waiting(queue,paused)        
        
        if client is not None: # we start a processing only if there is a client waiting
            self.average_delay_time += (time - client.arrival_time)    
            if client.paused_time is None:            
                service_time = random.expovariate(self.serv_lambda)
            else: # this means that the client has been put on pause, we can resume the processing
                service_time = client.time_left
            
            self.servers.start_service_if_possible(client) # this start immediately a service
            
            FES.put(Event(time + service_time, 'departure', client = client))
            
    def print_statistics(self,time):
        average_delay = self.average_delay_time/self.num_departures
        average_no_cust = self.average_utilization/time

        print(f'Number of clients \ number of departures: {self.num_arrivals} \ {self.num_departures}')
        print(f'Average time spent waiting: {average_delay:.4f}s\nAverage number of customers in the system: {average_no_cust:.2f}')
        print(f'Number of clients been put on pause: {self.num_paused} ({self.num_paused / self.num_arrivals * 100:.2f}%)')
        print(f'Number of clients in the system at the end: {self.num_arrivals - self.num_departures}')
        
    
def simulate(arr_lambda:int = ARRIVAL_LAMBDA, 
             serv_lambda: int = SERVICE_LAMBDA, 
             n_server: int = N_DOCTOR):
    
    # Initialization
    system = System(arr_lambda,serv_lambda,n_server)
    time = 0
    
    FES = PriorityQueue()
    queue = Queue()
    paused = Queue() # this will store all the clients that has been stopped
    
    FES.put(Event(time,'arrival'))
    
    # Event loop
    while time < SIM_TIME:
        if FES.has_no_events():
            break
        
        event = FES.get()
        time = event.time
        
        if event.type_ == 'arrival':
            system.arrival(time,FES,queue,paused)
        elif event.type_ == 'departure':
            system.departure(time,FES,queue,paused,event.client)
            
    # end of the simulation
    system.print_statistics(time)
    
if __name__ == '__main__':
    simulate()