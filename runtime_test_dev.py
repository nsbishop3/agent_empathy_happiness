# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:48:34 2022

@author: strel

This script is used for testing the model while developing it.
This script may not be completely current with the model as it is in sim.py with regards
to modeling the percentage of agents applying empathy. Because it is not in the scope of a funciton,
it is easy to add it in as a one-off test once the logic of the model flow is understood.
"""

from agents import agent
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
tqdm.format_interval(10)
#%% initialize population
pop = []
popsize = 100
for i in range(popsize):
    ind = agent(i,nfriends=10) # nfriends is social surface area
    if i < popsize//2:
        ind.person_type = 1
    else:
        ind.person_type = 0
    pop.append(ind)

#%% initialize relationshiops
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
    ind.inv_max_prob = 0.9 # this is sim dividing pressure

#%% test interactions ; invites/accept/decline - successful!
def network_graph_ax(axn,pop,popsize,show_conns=[0,1],title=''):
    # randlistx = [random.random() for xx in pop]
    # randlisty = [random.random() for xx in pop]
    coors = {}
    ring = popsize
    ringpl = 360 / ring
    for ind in pop:
        aid = ind.agent_id
        cx = np.cos(np.deg2rad(aid*ringpl))
        cy = np.sin(np.deg2rad(aid*ringpl))
        coors[ind] = [cx,cy]
        
    x = []
    y = []
    lbl = []
    arrow_locs = []
    for k,v in coors.items():
        x.append(v[0])
        y.append(v[1])
        lbl.append(k.person_type)
        conns = []
        for other,flvl in k.fondness.items():
            if flvl > 0:
                x2,y2 = coors[other]
                conns.append([x2-x[-1],y2-y[-1]])
        arrow_locs.append(conns)
    colors = ['blue','red']
    true_lbls = ["Intorvert","Extrovert"]
    axn.set_title(title)
    for i,j,k,arrs in zip(x,y,lbl,arrow_locs):
        if (i,j) == (x[0],y[0]) or (i,j) == (x[-1],y[-1]):
            axn.scatter(i,j,label=true_lbls[k],color=colors[k])
        else:
            axn.scatter(i,j,color=colors[k])
        for arr in arrs:
            if k in show_conns:
                axn.arrow(i,j,arr[0]*.99,arr[1]*.99,head_width=0.01,alpha=0.5,rasterized=True)
    axn.legend(title="Person Type",fontsize='x-small',title_fontsize='x-small')

def check_steady_state(vals,counter,seglen=1000,sustain=1000):
    segment = vals[-1000:]
    mins,maxs = min(segment),max(segment)
    past = segment[0]
    now = segment[-1]
    if len(vals) < seglen:
        return False,0,np.average(segment)
    if counter >= sustain:
        print('past',past)
        print('now',now)
        print('min seg',min(segment))
        print('max seg',max(segment))
        print('seg endpoints',segment[0],segment[-1])
        return True, counter, np.average(segment)
    # elif (mins<past<maxs and mins<now<maxs) or \
    elif (mins<now<maxs) or \
        (past == segment[1] and now == segment[-2]):
        return False,counter+1,np.average(segment)
    else:
        return False,0,np.average(segment)

#%% test battery action
pop_init = copy.deepcopy(pop)
for ind in pop:
    ind.update_happiness()
hlvl = [[xx.happiness for xx in pop]]
trace_hlvl = [np.average(hlvl)]
blvl = [[xx.battery_lvl for xx in pop]]
steps = 20000
ss_counter = 0
print('\n')
for i in trange(steps,mininterval=5,desc='Run sim:'):
    for action in ['send','receive']:
        for ind in pop:
            ind.battery_action(action)
    for ind in pop:
        ind.update_battery_level()
        ind.update_friends(pop)
        ind.reset_invs()
        ind.update_happiness()
        ind.apply_time_empathy(mode='all')
        ind.log.append(['--------------------------------------------------'])
    for ind in pop:
        ind.try_new_things(pop)
    happiness = [xx.happiness for xx in pop]
    hlvl.append(happiness)
    trace_hlvl.append(np.average(happiness))
    blvl.append([xx.battery_lvl for xx in pop])
    
    is_ss,ss_counter,ss_avg = check_steady_state(trace_hlvl,ss_counter)
    if is_ss:
        break
print('Happiness Result:',ss_avg)
#%% happiness across entire sim
fig,ax = plt.subplots()
x = range(len(hlvl))
y = trace_hlvl
ax.plot(x,y)
ax.vlines(x[-1000],min(y),max(y),color='black',linestyle='--')
ax.set_title('Average Happiness in Network')
ax.set_ylabel('Happiness Level')
ax.set_xlabel('Simulation Step')
ax.grid()

#%% network graph and degree vis
lbl_col = ['origin all ','origin introvert ', 'origin extrovert ']
lbl_row = ['before', 'after']
conns = [[0,1],[0],[1]]
nrows = 2
ncols = len(conns)
pops = [pop_init,pop]
fig, ax = plt.subplots(nrows=nrows,ncols=ncols)
for row in range(nrows):
    for col in range(ncols):
        cax = ax[row][col]
        cax.axes.get_xaxis().set_visible(False)
        cax.axes.get_yaxis().set_visible(False)
        network_graph_ax(cax,pops[row],popsize,show_conns=conns[col],title=lbl_col[col]+lbl_row[row])

ind_sends = []
ind_recs = []
for ind in pop:
    ind_sends.append(0)
    ind_recs.append(0)
for ind in pop:
    ind_sends[ind.agent_id] = len(ind.friends)
    for other,fond in ind.fondness.items():
        ind_recs[other.agent_id] += 1
        
figsr,axsr = plt.subplots()
x = np.arange(len(ind_sends))
axsr.bar(x-0.2,ind_sends,0.4,label='Degree out')
axsr.bar(x+0.2,ind_recs,0.4,label='Degree in')
axsr.set_yticks(np.arange(max(max(ind_sends),max(ind_recs))+1))
axsr.legend()
axsr.set_title('Node Degree of Each Agent')
axsr.grid(axis='y')
axsr.set_axisbelow(True)
axsr.set_xlabel('Agent ID - Lowest half are extroverts')
axsr.set_ylabel('Degree Number')

#%% uncomment and specify which idx in pop to create log file for agent. put this in a loop for multiple agents
# with open("log.log", "w") as log:
#     for line in pop[6].log:
#         log.write(str(line)+'\n')