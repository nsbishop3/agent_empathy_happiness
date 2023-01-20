# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:48:34 2022

@author: strel

This script is run for sensitivity analysis 2, that iterates over the number of agents applying empathy
rules in the network.
"""


import sim

def run_sim_parallel(params):
    results = sim.run_sim(*params)
    return results


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count()//3) # use all CPUs available
    suffix = '_16_24'

    #%% scenario vars
    num_steps = 20000
    num_agents = 100
    time_empathy_modes = ['None','in-type','all'] # time empathy
    num_friends_per_agent = [10] # social surface area
    dividing_pressure = [0.5] # max inv probability
    ups_downs_list = [[0.05,0.05],[0.05,0.025]] # event empathy
    updown_ig = [False,True] # apply event empathy in-group
    seed_vals = list(range(16,24))
    # create list of lists of agents to be empathetic in population
    emp_agents_list = [[0,num_agents-1]]
    for i in range(1,49):
        bot = i
        top = 99-i
        emp_ag = emp_agents_list[-1] + [bot,top]
        emp_ag.sort()
        emp_agents_list.append(emp_ag)
    
    # create parameters to sweep over
    params = []
    for seed in seed_vals:
        for time_empathy_mode in time_empathy_modes:
            for nconns in num_friends_per_agent:
                for div_pressure in dividing_pressure:
                    for ups_downs in ups_downs_list:
                        for updown_ingroup in updown_ig:
                            for emp_agents in emp_agents_list:
                                params.append([num_agents,
                                               nconns,
                                               num_steps,
                                               div_pressure,
                                               time_empathy_mode,
                                               ups_downs,
                                               updown_ingroup,
                                               emp_agents,[],
                                               seed
                                               ])
    
    pb = tqdm(range(len(params)),desc='Analysis')
    
    results = [pool.apply_async(run_sim_parallel, args=[pars]) for pars in params]
    
    sim_results = []
    ss_stats = []
    for i,r in enumerate(results):
        res = r.get()
        sim_results.append(res)
        ss_stats.append([round(np.average(res[-1]),2),round(np.std(res[-1]),2)])
        pb.update()
    
    for p,stat in zip(params,ss_stats):
        print(p,stat)
    
    # main sentivitiy analysis results
    df = pd.DataFrame({
        'seed': [xx[-1] for xx in params],
        'nfriends': [xx[1] for xx in params],
        'div_pres': [xx[3] for xx in params],
        'empathy': [xx[4] for xx in params],
        'ups': [xx[5][0] for xx in params],
        'downs': [xx[5][1] for xx in params],
        'updowns_ingroup': [xx[6] for xx in params],
        'emp_agents': [len(xx[7]) for xx in params], # get num of emp agents, can reconstruct later
        })
    df.to_csv('data/select_agents_empathy_analysis'+suffix+'.csv',index=False)
    
    # hold steady states (last 1000 iterations)
    df_ss = pd.DataFrame({
        str(param):ss for param,ss in zip(params,[xx[5] for xx in sim_results])
        })
    df_ss.to_hdf('data/steady_state_sae'+suffix+'.hdf','df')
    #%%
    # hold avg happiness levels of pop for each step
    df_hlvl = pd.DataFrame({
        str(param):hlvl + [0]*(1+num_steps-len(hlvl)) for param,hlvl in zip(params,[xx[2] for xx in sim_results])
        })
    df_hlvl.to_hdf('data/hlvl_sae'+suffix+'.hdf','df')
    
    df_hlvl_ae = pd.DataFrame({
        str(param):hlvl + [0]*(1+num_steps-len(hlvl)) for param,hlvl in zip(params,[xx[3] for xx in sim_results])
        })
    df_hlvl_ae.to_hdf('data/hlvl_ae_sae'+suffix+'.hdf','df')
    
    df_hlvl_ane = pd.DataFrame({
        str(param):hlvl + [0]*(1+num_steps-len(hlvl)) for param,hlvl in zip(params,[xx[4] for xx in sim_results])
        })
    df_hlvl_ane.to_hdf('data/hlvl_ane_sae'+suffix+'.hdf','df')
    #%% COMMENT THIS SECTION IF YOU ARE PERFORMING RUNS OVER MORE THAN ONE SEED VALUE
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
    
    df_popi.to_csv('data/init_pop_sae'+suffix+'.csv',index=False)
    df_pop.to_csv('data/pop_sae'+suffix+'.csv',index=False)
    #%%
    pool.close()
    pool.join()