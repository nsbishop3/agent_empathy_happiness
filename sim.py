# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:32:47 2022

@author: strel
"""
from agents import agent
import numpy as np
import random
import copy

def run_sim(popsize,nconns,num_steps,dividing_pressure,
            time_empathy_mode,ups_downs=None,apply_updown_ingroup=False,empathetic_agents=[],
            create_logs=[],seed=0,return_pop=False):
    '''
    Performs one run with specified parameters for agent empathy and happiness simulation

    Parameters
    ----------
    popsize : int
        Number of agents.
    nconns : int
        Number of friends per agent.
    num_steps : int
        Max number of steps to take if a steady state is not detected. When reached, the simulation
        finishes.
    dividing_pressure : float
        This value is directly assigned to each agent's maximum invite probability. As this value
        increases, the happiness generally decreases and the overall structure of the social network
        becomes more divided.
    time_empathy_mode : str of only 'None', 'in-type', or 'all'
        Determines if time emapthy is applied and how it is applied.
    ups_downs : [float,float], optional
         A list of length two that determines how much fondness increases and decreases for each 
         interaction. The first element determines increase in fondness for accepted invites, and 
         the second element determines decrease in fondness for declined invites. If set to None,
         the simulation will use [0.05,0.05]. The default is None.
         Adjusting these values allows for the implementation of event-based empathy.
    apply_updown_ingroup : bool, optional
        Determines application type of event based empathy. If set to True, the values for up_downs
        will only apply to agents of the same person type, while agents not of the same agent
        person type will use [0.05,0.05] as the up_downs value. The default is False.
    empathetic_agents : [ints], optional
        List of integers, pertaining to each agent's agent_id, specifying which agents will apply
        the emapthy rules specified in the input parameters. If set to [], all agents will apply
        empathy as specified. The default is [].
    create_logs : [ints], optional
        List of integers, pertaining to each agent's agent_id, that will generate log files. If an
        agent generates log files, then a list will be generated with each element showing all events
        pertaining to the specified agents in chronological order as an attribute of each agent. The default is [].
        It is not implemented in this function, but a user of the code could easily get the log of each 
        agent and convert it to a text file to use as a traditional log file.
    seed : int, optional
        Seed to set for random number pulls. This allows for consistency and more comparability between
        runs using different parameters. The default is 0.
    return_pop : bool, optional
        Determines if the initial and final population should be returned as a dataframe. 
        DO NOT USE THIS IF YOU PLAN ON DOING MORE THAN A FEW RUNS AT A TIME, YOU WILL RUN OUT OF MEMORY.
        The default is False.

    Returns
    -------
    pop_init :  [agent objects]
        List of agent objects. Can be used to create a pandas dataframe then export information 
        about the initial population.
    pop : [agent objects]
        List of agent objects. Can be used to create a pandas dataframe then export information 
        about the final population.
    trace_hlvl : [floats]
        List with float elements representing the average happiness of the network at each timestep.
    trace_hlvl_ae : [floats]
        List with float elements representing the average happiness of the network of all the agents
        that apply empathy at each timestep.
    trace_hlvl_ane : [floats]
        List with float elements representing the average happiness of the network of all the agents
        that do not apply empathy at each timestep.
    ss : [floats]
        List of last 1000 timestep happiness values in trace_hlvl, which is used for the calculation
        of the steady state.

    '''
    if empathetic_agents == []:
        empathetic_agents = list(range(popsize))
    random.seed(seed)
    np.random.seed(seed)
    #%% initialize population
    pop = []
    pop_emp = []
    pop_no_emp = []
    for i in range(popsize):
        ind = agent(i,nfriends=nconns) # nfriends is social surface area
        if i < popsize//2:
            ind.person_type = 1
        else:
            ind.person_type = 0
        pop.append(ind)
        if ind.agent_id in empathetic_agents:
            pop_emp.append(ind)
        else:
            pop_no_emp.append(ind)
    
    #%% initialize relationships
    
    # create init_fondness lookup table
    def get_fond(agent,other):
        diff = 0
        for my_trait,their_trait in zip(agent,other):
            diff_tmp = (their_trait - my_trait)**2 # r squared
            diff += diff_tmp/len(agent) # normalizing to be 0 - 1
        return 1-diff
    
    fond_lookup = {}
    for i in range(len(pop)-1):
        ind1 = pop[i]
        for j in range(i+1,len(pop)):
            ind2 = pop[j]
            fond = get_fond(ind1.person_traits,ind2.person_traits)
            fond_lookup[(ind1,ind2)] = fond
            fond_lookup[(ind2,ind1)] = fond
    
    for ind in pop:
        ind.fondness_lookup = fond_lookup
    
    for ind in np.random.choice(pop,size=len(pop),replace=False):
        ind.set_init_fondness(pop)
        ind.inv_max_prob = dividing_pressure # this is sim dividing pressure
        if ups_downs is not None:
            ind.accept_increase = ups_downs[0]
            # take care of which agents will perform event empathy
            if ind.agent_id in empathetic_agents: 
                # if not meant for event empathy, leave as default value
                ind.decline_reduce = ups_downs[1]
    
    #%% steady state functions
    def check_steady_state(vals,counter,seglen=1000,sustain=1000):
        segment = vals[-1000:]
        mins,maxs = min(segment),max(segment)
        past = segment[0]
        now = segment[-1]
        if len(vals) < seglen:
            return False,0,segment
        if counter >= sustain:
            return True, counter, segment
        elif (mins<now<maxs) or \
            (past == segment[1] and now == segment[-2]):
            return False,counter+1,segment
        else:
            return False,0,segment
    
    #%% run sim
    # create trace vars #
    pop_init = copy.deepcopy(pop)
    for ind in pop:
        ind.update_happiness()
    hlvl = [[xx.happiness for xx in pop]]
    hlvl_ae = [[xx.happiness for xx in pop_emp]]
    hlvl_ane = [[xx.happiness for xx in pop_no_emp]]
    trace_hlvl = [np.average(hlvl)]
    trace_hlvl_ae = [np.average(hlvl_ae)]
    trace_hlvl_ane = [np.average(hlvl_ane)]
    blvl = [[xx.battery_lvl for xx in pop]]
    steps = num_steps
    ss_counter = 0
    # begin sim #
    for i in range(steps):
        # all agents perform actions at once 
        for action in ['send','receive']:
            for ind in pop:
                ind.battery_action(action)
        # update agents based on interactions #
        for ind in pop:
            ind.update_battery_level()
            # event empathy application
            if apply_updown_ingroup:
                ind.update_friends(pop,apply_ingroup=True)
            else:
                ind.update_friends(pop)
            # housekeeping
            ind.reset_invs()
            ind.update_happiness()
            # apply time empathy
            if ind in pop_emp:
                ind.apply_time_empathy(mode=time_empathy_mode)
            ind.log.append(['--------------------------------------------------'])
            # remove unnecessary logs. will overrun memory on a large sensitivity analysis.
            if ind not in create_logs:
                ind.log = []
        # perform possible new friend search for each agent
        for ind in pop:
            ind.try_new_things(pop)
        # update trace vars
        happiness = [xx.happiness for xx in pop]
        happiness_ae = [xx.happiness for xx in pop_emp]
        happiness_ane = [xx.happiness for xx in pop_no_emp]
        trace_hlvl.append(np.average(happiness))
        trace_hlvl_ae.append(np.average(happiness_ae))
        trace_hlvl_ane.append(np.average(happiness_ane))
        blvl.append([xx.battery_lvl for xx in pop])
        # check for steady state and stop early if that is the case
        is_ss,ss_counter,ss = check_steady_state(trace_hlvl,ss_counter)
        if is_ss:
            break
    if return_pop==True:
        return pop_init,pop,trace_hlvl,trace_hlvl_ae,trace_hlvl_ane,ss
    else:
        return [],[],trace_hlvl,trace_hlvl_ae,trace_hlvl_ane,ss
