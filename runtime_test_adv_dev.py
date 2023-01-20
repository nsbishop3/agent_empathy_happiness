# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:48:34 2022

@author: strel

This script serves as a runtime test script for the model as it became more advanced in development.

The model is used within the scope of a function, but the code layout used in the multi-processing scripts
is the same. This script allowed for verification of the output of sim.py, as well as the data processing
that comes after all the parameters are pushed through sim.py
"""


import sim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% scenario vars
num_steps = 20000
num_agents = 100
empathy_modes = ['None'] # time empathy
num_friends_per_agent = [10] # social surface area
dividing_pressure = [0.9] # max inv probability
ups_downs_list = [[0.05,0.025]] # event empathy
updown_ig = [False] # event empathy in-group application
# create list of agents to be empathetic
emp_agents_list = [[0,num_agents-1]]
for i in range(1,26):
    bot = i
    top = 99-i
    emp_ag = emp_agents_list[-1] + [bot,top]
    emp_ag.sort()
    emp_agents_list.append(emp_ag)
emp_agents_list = [emp_agents_list[-1]]

# create all params to iterate over
params = []
for empathy_mode in empathy_modes:
    for nconns in num_friends_per_agent:
        for div_pressure in dividing_pressure:
            for ups_downs in ups_downs_list:
                for updown_ingroup in updown_ig:
                    for emp_agents in emp_agents_list:
                        params.append([num_agents,
                                       nconns,
                                       num_steps,
                                       div_pressure,
                                       empathy_mode,
                                       ups_downs,
                                       updown_ingroup,
                                       emp_agents,
                                       ])

pb = tqdm(range(len(params)),desc='Analysis')

sim_results = []
ss_stats = []
for pars in params:
    res = sim.run_sim(*pars)
    sim_results.append(res)
    ss_stats.append([round(np.average(res[3]),2),round(np.std(res[3]),2)])
    pb.update()
    

for p,stat in zip(params,ss_stats):
    print([p[1],p[3],p[4],p[5]],stat)

# main sentivitiy analysis results
df = pd.DataFrame({
    'nfriends': [xx[1] for xx in params],
    'div_pres': [xx[3] for xx in params],
    'empathy': [xx[4] for xx in params],
    'ups': [xx[5][0] for xx in params],
    'downs': [xx[5][1] for xx in params],
    'updowns_ingroup': [xx[6] for xx in params],
    'emp_agents': [len(xx[7]) for xx in params], # get num of emp agents, can reconstruct later
    })
df.to_csv('data/test_select_agents_empathy_analysis.csv',index=False)

# hold steady states (last 1000 iterations)
df_ss = pd.DataFrame({
    str(param):ss for param,ss in zip(params,[xx[5] for xx in sim_results])
    })
df_ss.to_hdf('data/test_steady_state_sae.hdf','df')
#%%
# hold avg happiness levels of pop for each step
df_hlvl = pd.DataFrame({
    str(param):hlvl + [0]*(1+num_steps-len(hlvl)) for param,hlvl in zip(params,[xx[2] for xx in sim_results])
    })
df_hlvl.to_hdf('data/test_hlvl_sae.hdf','df')

df_hlvl_ae = pd.DataFrame({
    str(param):hlvl + [0]*(1+num_steps-len(hlvl)) for param,hlvl in zip(params,[xx[3] for xx in sim_results])
    })
df_hlvl_ae.to_hdf('data/test_hlvl_ae_sae.hdf','df')

df_hlvl_ane = pd.DataFrame({
    str(param):hlvl + [0]*(1+num_steps-len(hlvl)) for param,hlvl in zip(params,[xx[4] for xx in sim_results])
    })
df_hlvl_ane.to_hdf('data/test_hlvl_ane_sae.hdf','df')
#%%
# hold population obj info
df_popi = pd.DataFrame({})
df_pop = pd.DataFrame({})
for i,simres in enumerate(sim_results):
    df_popitmp = pd.DataFrame({})
    df_poptmp = pd.DataFrame({})
    
    popi = simres[0]
    pop = simres[1]
    
    df_popitmp['run'] = [str(params[i]) for xx in popi]
    df_popitmp['run_idx'] = [i for xx in popi]
    df_popitmp['agent_id'] = [xx.agent_id for xx in popi]
    df_popitmp['person_type'] = [xx.person_type for xx in popi]
    df_popitmp['friends'] = [[yy.agent_id for yy in xx.friends] for xx in popi]
    
    df_poptmp['run'] = [str(params[i]) for xx in pop]
    df_poptmp['run_idx'] = [i for xx in pop]
    df_poptmp['agent_id'] = [xx.agent_id for xx in pop]
    df_poptmp['person_type'] = [xx.person_type for xx in pop]
    df_poptmp['friends'] = [[yy.agent_id for yy in xx.friends] for xx in pop]
    
    df_popi = df_popi.append(df_popitmp)
    df_pop = df_pop.append(df_poptmp)

df_popi.to_csv('data/test_init_pop_sae.csv',index=False)
df_pop.to_csv('data/test_pop_sae.csv',index=False)

#%% graphing
def network_graph_ax(axn,pop,popsize,show_conns=[0,1],title=''):
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
    axn.set_title(title)
    for i,j,k,arrs in zip(x,y,lbl,arrow_locs):
        axn.scatter(i,j,label=k,color=colors[k])
        for arr in arrs:
            if k in show_conns:
                axn.arrow(i,j,arr[0]*.99,arr[1]*.99,head_width=0.01,alpha=0.5,rasterized=True)
# %% happiness graph
figh,axh = plt.subplots()
x = range(len(df_hlvl.loc[df_hlvl[df_hlvl.columns[0]]>0]))
y = df_hlvl.loc[df_hlvl[df_hlvl.columns[0]]>0]
y2 = df_hlvl_ae.loc[df_hlvl[df_hlvl.columns[0]]>0]
y3 = df_hlvl_ane.loc[df_hlvl[df_hlvl.columns[0]]>0]
axh.plot(x,y)
axh.plot(x,y2)
axh.plot(x,y3)

#%% network graph
lbl_col = ['origin all ','origin introvert ', 'origin extrovert ']
lbl_row = ['before', 'after']
conns = [[0,1],[0],[1]]
nrows = 2
ncols = len(conns)
pops = [popi,pop]
fig, ax = plt.subplots(nrows=nrows,ncols=ncols)
for row in range(nrows):
    for col in range(ncols):
        cax = ax[row][col]
        cax.axes.get_xaxis().set_visible(False)
        cax.axes.get_yaxis().set_visible(False)
        network_graph_ax(cax,pops[row],num_agents,show_conns=conns[col],title=lbl_col[col]+lbl_row[row])


#%% network degree check
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
axsr.bar(x+0.2,ind_recs,0.4,label='Degree in')
axsr.bar(x-0.2,ind_sends,0.4,label='Degree out')
axsr.set_yticks(np.arange(max(max(ind_sends),max(ind_recs))+1))
axsr.legend()
axsr.set_title('Node Degree of Each Agent')
axsr.grid(axis='y')
axsr.set_axisbelow(True)
axsr.set_xlabel('Agent ID - Lowest half are extroverts')
axsr.set_ylabel('Degree Number')
