#---------------------------------------------------------------------------------------------------------------------------------------------#

# IMPORTS

import random
from enum import Enum
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

#---------------------------------------------------------------------------------------------------------------------------------------------#

"""
There are 7 different classes for managing this queue system:
1) The enum class for the Urgency. I choosed to use an Enum in order to have an object that could 
   be comparable
   
3) The Client class that has all the functionality for managing the client. You can find the code
   that randomly assign (following the professor's request) the Urgency as a method of this class. I
   choosed to give all the clients the same arrival distribution and after the arrival assign each one
   a random Urgency. At the end of the day, the probability for each urgency is the one requested. 
   
4) The Event class: this is usefull for adding some more information about an event. I choosed to link
   a departure event with the client it should departure, this because the system has to know wich client
   is supposed to departure if there is an interruption caused by an arrival of a red client. There was 
   need to link also the arrival with the client so those events are not associated with any client. 
   
5) The PriorityQueue class: this is my personal implementation of the FES. Here we have a simple list
   constantly sorted by the implementation of the insertion sort algorithm. When there is an addition
   the element in inserted into the list at the position that maintains the desired order. The desired
   order is the one that maintains the first element as the closest in time. In this class you can 
   find the code that deals with searching for the scheduled departure for a client and deletes it.
   
6) The Queue class: this is my implementation of the queue. This class is used for the queue but also 
   to store the paused clients. This class is also a simple list constantly sorted by the implementation
   of the insertion sort algorithm. The desidered order is the one that maintains the most urgent clients
   first and the same type of clients are ordere based on their arrival time. I add another order based
   on the paused time because when it is used as paused list of clients they have to be ordered not by
   their arrival time but by the time in which their service has been paused. There are also some simple
   methods for adding some readability to the code. 
   
5) The ServersList class: this is a class of clients even if it's called ServersList, but it represent
   the servers that are busy serving a client. It can reach only a certain capacity and it also mainain 
   sorted but only by the clients urgency. This is done to prevent to put a YELLOW client in pause if there
   is a GREEN client in another server (the green clients are the one first checked in this way). There 
   are also some function to improve readability. 
   
6) The System class is maybe the more useless but it represent the system in general. This class contain
   all the method for running the simulation and the existance of this class allowed me to not have any
   global variables. All the variable that has to be collected are stored inside this class, and also the
   foundamental method for a simulation. 
   
At the end there are two function, the simulate function is the method that contain all the Event Loop and
all its input parameters are the input parameters for the simulation itself. The second function is the one
that plot the graph but I think it's self explanatory. 

At the very end I put the part of the code that start running when the file is called. Here it can be found 
all the code for storing the result and for passing those to the method that plot all the results. 

I hope this introduction can help understand better this code. 
"""

#---------------------------------------------------------------------------------------------------------------------------------------------#

#
#   Class That manage the Urgency of a client
#
class Urgency(Enum):
    """
    This class manage the Urgency color of the clients. It is an enumeration (or Enum).
    The attributes Urgency.RED, Urgency.YELLOW, etc., are enumeration members and are 
    functionally constants. 
    """
    RED = 1
    YELLOW = 2
    GREEN = 3
    
    def mu_adjust(self):
        """
        This method return a value that has to be added to the service lambda.
        The logic behind this is that green patients should take less time than yellow 
        and yellow less time than red patients. This method adjust slightly the lambda following this
        thought

        Returns:
            float: Value that has to be added to the service lambda to slightly adjust the distribution
            based on the urgency of the client
        """
        return self.value - 3 * .5
    
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

#---------------------------------------------------------------------------------------------------------------------------------------------#

#
#   Class that represent a Client
#
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
        - If returns something < 0 it means that the other is less urgent.
        - If returns something > 0 it means that the other is more urgent.
        - If returns something == 0 it means that the other has the same urgency.

        Args:
            other (Client): It's the client to compare this client with

        Returns:
            int: Returns an int that represent if one is urgent than the other
        """
        compare = self.urgency.compare_to(other.urgency) # this will be 0 only if they have the same urgency
        if compare == 0: # they have the same urgency
            if self.paused_time is None: # the order is based on their arrival_time
                compare = self.arrival_time - other.arrival_time
            else: # the order is based on their paused_time
                compare = self.paused_time - other.paused_time
        return compare
    
    def is_less_urgent_than(self,other: 'Client'):
        return self.urgency.compare_to(other.urgency) > 0 # it means that other is more urgent
    
    def is_red(self):
        return self.urgency == Urgency.RED

#---------------------------------------------------------------------------------------------------------------------------------------------#

#
#   Class that represent an Event
#
class Event:
    def __init__(self, time:float, type_:str, client: Client = None):
        self.time = time
        self.type_ = type_
        self.client = client
    
    def compare_to(self,other: 'Event'):
        """
        - If returns something < 0 it means that the other is after.
        - If returns something > 0 it means that the other is before.
        - If returns something == 0 it means that they happen in the same exact time (very unlikely, I'm not handling this case).

        Args:
            other (Event): It's the other event to compare this event with

        Returns:
            int: Returns an int that represent if one is happening before of the other
        """
        return self.time - other.time
    
    def is_early_than(self,other: 'Event'):
        return self.compare_to(other) < 0

#---------------------------------------------------------------------------------------------------------------------------------------------#

#
#   Personal implementation of the PriorityQueue (FES)
#
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
        This method returns the closest event in time. As the list is kept in order, 
        it can simply be the first element.

        Returns:
            Event: An object representing an event 
        """
        while True: 
            if len(self.events) > 0:
                return self.events.pop(0)
            else:
                return None # it should neve occurs, but let's prevent some error
    
    def stop_service(self, client: Client):
        """It finds the departure schedule for this client and deletes it

        Returns:
            float: Is the time when the events would take place
        """
        for event in self.events:
            if event.client == client:
                self.events.remove(event)
                return event.time # it returns the time when the events would take place

#---------------------------------------------------------------------------------------------------------------------------------------------#

#
#   Class that represent a Queue (is it used for both queue and paused client)
#
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

#---------------------------------------------------------------------------------------------------------------------------------------------#

#
#   Class that represent the clients that are in service
#
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

#---------------------------------------------------------------------------------------------------------------------------------------------#

#
#   Class that old all system's measures and parameters
#
class System:
    """Class that store the system variables (so I can avoid using global variables) 
    and it is only an extention of the measurement parameters class
    """
    def __init__(self,arr_lambda: int, serv_mu: int, n_server: int, 
                 Narr=None,Ndep=None,Ndred=None,Ndyel=None,Ndgre=None,AverageQLen=None,NAverageUser=None,OldTimeEvent=None,AverageDelay=None, Npau=None):
        # input parameters
        self.arr_lambda = arr_lambda
        self.serv_mu = serv_mu
        self.n_server = ServersList(n_server)
        
        # system variables (so I can avoid using global variables)
        self.servers = ServersList(n_servers = n_server)
        self.clients = 0 # clients in the system
        
        # Measurements paramenters
        self.num_arrivals = 0 if Narr is None else Narr
        self.num_departures = 0 if Ndep is None else Ndep
        self.red_departures = 0 if Ndred is None else Ndred
        self.yellow_departures = 0 if Ndyel is None else Ndyel
        self.green_departures = 0 if Ndgre is None else Ndgre
        self.average_queue_length = 0 if AverageQLen is None else AverageQLen
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
    
    def calculate_service_time(self,client):
        """This method calculate the service time. I add an adjustment parameter to give slightly 
        different distribution based on the urgency of the customers.  

        Args:
            client (Client): The Client object for which the service time is to be calculated

        Returns:
            float: The random time calculated for the service 
        """
        return random.expovariate(self.serv_mu + client.urgency.mu_adjust())
        
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
        
        if result == True: # it means that there were a server available and the service has started
            # determine the service time Ts
            service_time = self.calculate_service_time(client)
            
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
            service_time = self.calculate_service_time(client)
            
            FES.put(Event(time + service_time, 'departure', client=client))
        
        # measure the average length queue
        self.average_queue_length += len(queue.queue)
    
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
        if client.urgency == Urgency.GREEN:
            self.green_departures += 1
        elif client.urgency == Urgency.YELLOW:
            self.yellow_departures += 1
        else:
            self.red_departures +=1
        
        self.clients -= 1
        self.servers.departure(client) # we can free the server        
        
        client = self.most_urgent_waiting(queue,paused)
        
        # measure the average length queue
        self.average_queue_length += len(queue.queue)
        
        if client is not None: # we start a processing only if there is a client waiting
            self.average_delay_time += (time - client.arrival_time)    
            if client.paused_time is None:            
                service_time = self.calculate_service_time(client)
            else: # this means that the client has been put on pause, we can resume the processing
                service_time = client.time_left
            
            self.servers.start_service_if_possible(client) # this start immediately a service
            
            FES.put(Event(time + service_time, 'departure', client = client))
            
    def print_statistics(self,time):
        average_delay = self.average_delay_time/self.num_departures
        average_no_cust = self.average_utilization/time
        average_queue_len = self.average_queue_length/time

        print(f'Number of clients \ number of departures: {self.num_arrivals} \ {self.num_departures}')
        print(f'Number of green patients served: {self.green_departures} ({self.green_departures/self.num_departures * 100:.2f}%)')
        print(f'Number of yellow patients served: {self.yellow_departures} ({self.yellow_departures/self.num_departures * 100:.2f}%)')
        print(f'Number of red patients served: {self.red_departures} ({self.red_departures/self.num_departures * 100:.2f}%)')
        print(f'Average time spent waiting: {average_delay:.4f}s\nAverage number of customers in the system: {average_no_cust:.2f}')
        print(f'Average queue lenght: {average_queue_len:.2f} patients')
        print(f'Number of clients been put on pause: {self.num_paused} ({self.num_paused / self.num_arrivals * 100:.2f}%)')
        print(f'Number of clients in the system at the end: {self.num_arrivals - self.num_departures}')
        print('\n ===================== \n')
    
#---------------------------------------------------------------------------------------------------------------------------------------------#

#
#   Method that contains the Event Loop
#
def simulate(arr_lambda:int = 5, # input parameter for the arrival_lambda variable
             serv_mu: int = 1, # input parameter for the service_mu variable
             n_server: int = 5, # input parameter for the number of server in the system
             sim_time: int = 200): # input parameter for the simulation time
    
    # Initialization
    system = System(arr_lambda,serv_mu,n_server)
    time = 0
    
    FES = PriorityQueue()
    queue = Queue()
    paused = Queue() # this will store all the clients that has been stopped
    
    FES.put(Event(time,'arrival'))
    
    pbar = tqdm(total=sim_time,
                desc=f'Simulating with n_server = {n_server}, arr_lambda = {arr_lambda}, serv_mu = {serv_mu} and sim_time = {sim_time}',
                bar_format='{l_bar}{bar:30}{n:.0f}s/{total}s [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Event loop
    while time < sim_time:
        if FES.has_no_events():
            break
        
        event = FES.get()
        
        if event.time < sim_time: # to prevent a warning to appear
            pbar.update(event.time - time)
        time = event.time
        
        if event.type_ == 'arrival':
            system.arrival(time,FES,queue,paused)
        elif event.type_ == 'departure':
            system.departure(time,FES,queue,paused,event.client)
    
    pbar.close()
            
    # end of the simulation
    system.print_statistics(time)
    
    return [system.num_arrivals, system.num_departures, system.red_departures, system.yellow_departures, system.green_departures, \
        system.average_queue_length, system.average_utilization, system.time_last_event, system.average_delay_time, system.num_paused, time]

#---------------------------------------------------------------------------------------------------------------------------------------------#

#
#   Method for the plot 1
#
def plot_graph(df,selected_arrival_lambda=12):
    """
    This method plots 4 different plots for analyze some results after the simulation

    Args:
        df (DataFrame): DataFrame that contains the measures after the simulations
        selected_arrival_lambda (int, optional): This is the selected arrival lambda fixed to show the results. Defaults to 12.
    """
    
    df_1 = df[df['ArrivalLambda'] == selected_arrival_lambda]
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    folder_path = os.path.join(script_directory, 'output_images',current_time)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Plot 1 -> Service Mu vs Percentages of processed client with fixed arrival lambda, it plots how the system handle different rate 
    plt.plot(df_1['ServiceMu'], df_1['RedDepartures'] / df_1['TotDepartures'], color='r', linestyle='-', marker='o', label='Red Client')
    plt.plot(df_1['ServiceMu'], df_1['YellowDepartures'] / df_1['TotDepartures'], color='y', linestyle='-', marker='o', label='Yellow Client')
    plt.plot(df_1['ServiceMu'], df_1['GreenDepartures'] / df_1['TotDepartures'], color='g', linestyle='-', marker='o', label='Green Client')
    plt.title(f'Service μ vs Percentages of processed client with Arrival λ = {selected_arrival_lambda}')
    plt.xlabel('μ')
    plt.xticks(df_1['ServiceMu'])
    plt.ylabel('Percentages')
    plt.grid(True)
    plt.legend()
    file_name = os.path.join(folder_path, 'output_1.png')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2 -> Service Mu vs Total Number of processed clients with fixed arrival lambda, it plots the increasing number of clients that the system can handle increasing mu
    plt.plot(df_1['ServiceMu'], df_1['TotDepartures'], color='black', linestyle='-', marker='o')
    plt.title(f'Service μ vs Total Number of processed clients with Arrival λ = {selected_arrival_lambda}')
    plt.xlabel('μ')
    plt.xticks(df_1['ServiceMu'])
    plt.ylabel('Number of processed clients')
    plt.grid(True)
    file_name = os.path.join(folder_path, 'output_2.png')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3 -> Service Mu vs Average Queue length with fixed arrival lambda, it plots the decreasing number of clients waiting on the queue as the mu increases
    plt.plot(df_1['ServiceMu'], df_1['AverageQueueLenght'] / df_1['FinalTime'], color='purple', linestyle='-', marker='o')
    plt.title(f'Service μ vs Average Queue Length with Arrival λ = {selected_arrival_lambda}')
    plt.xlabel('μ')
    plt.xticks(df_1['ServiceMu'])
    plt.ylabel('Average Queue Length')
    plt.grid(True)
    file_name = os.path.join(folder_path, 'output_3.png')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4 -> Service Mu vs Total number of paused clients with fixed arrival lambda, it plots how many service the system interrupt to process red client with different mu
    plt.plot(df_1['ServiceMu'], df_1['TotPaused'], color='orange', linestyle='-', marker='o')
    plt.title(f'Service μ vs Total Number of paused Clients with Arrival λ = {selected_arrival_lambda}')
    plt.xlabel('μ')
    plt.xticks(df_1['ServiceMu'])
    plt.ylabel('Total Paused Clients')
    plt.grid(True)
    file_name = os.path.join(folder_path, 'output_4.png')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    
#---------------------------------------------------------------------------------------------------------------------------------------------#

#
#   Main method
#
if __name__ == '__main__':
    
    # DataFrame to store the measure for plotting
    df = pd.DataFrame(columns=['ArrivalLambda','ServiceMu','TotArrivals','TotDepartures','RedDepartures','YellowDepartures','GreenDepartures',\
        'AverageQueueLenght','AverageUtilization','TimeLastEvent','AverageDelayTime','TotPaused', 'FinalTime'])
 
    # Combinations of parameters to simulate the system with
    param_dict = {
        'arrival_lambdas': [3,5,7,9,12,15],
        'service_mus': [3,5,7,9,12,15]
    }
    
    # Loop for trying the different combinations
    for arrival_lambda in param_dict['arrival_lambdas']:
        for service_mu in param_dict['service_mus']:
            df.loc[len(df)] = [arrival_lambda, service_mu] + simulate(arr_lambda=arrival_lambda,\
                serv_mu=service_mu, n_server=1, sim_time=2_000)
        
    plot_graph(df)