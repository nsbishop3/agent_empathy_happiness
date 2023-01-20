# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 19:48:34 2022

@author: strel

This function is used to run the parameter sweep sensitivity analysis
"""

# from agents import agent

import sim

def run_sim_parallel(params):
    results = sim.run_sim(*params)
    return results


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count()//3) # use all CPUs available.
    
    suffix = '_test' # appened to filenames for organizational purposes of multiple runs
    # np.random.seed(0)
    #%% scenario vars
    num_steps = 20000
    num_agents = 100
    empathy_modes = ['None','in-type','all'] # time empathy
    num_friends_per_agent = [5,10,20] # social surface area
    dividing_pressure = [0.1,0.25,0.5,0.75,0.9] # max inv probability 
    ups_downs_list = [[0.05,0.05],[0.05,0.04],[0.05,0.03],[0.05,0.02],[0.05,0.01],[0.05,0.0]] # event empathy
    seed_vals = list(range(24,30))
    
    params = []
    for seed in seed_vals:
        for empathy_mode in empathy_modes:
            for nconns in num_friends_per_agent:
                for div_pressure in dividing_pressure:
                    for ups_downs in ups_downs_list:
                        for updown_ingroup in [False,True]:
                            params.append([num_agents,
                                           nconns,
                                           num_steps,
                                           div_pressure,
                                           empathy_mode,
                                           ups_downs,
                                           updown_ingroup,[],[],
                                           seed
                                           ])
    
    pb = tqdm(range(len(params)),desc='Analysis')
    
    results = [pool.apply_async(run_sim_parallel, args=[pars]) for pars in params[:4]]
    
    sim_results = []
    ss_stats = []
    for i,r in enumerate(results):
        res = r.get()
        sim_results.append(res)
        ss_stats.append([round(np.average(res[3]),2),round(np.std(res[3]),2)])
        pb.update()
    
    
    for p,stat in zip(params,ss_stats):
        print([p[1],p[3],p[4],p[5]],stat)
    
    # main sentivitiy analysis results
    df = pd.DataFrame({
        'seed': [xx[9] for xx in params],
        'nfriends': [xx[1] for xx in params],
        'div_pres': [xx[3] for xx in params],
        'empathy': [xx[4] for xx in params],
        'ups': [xx[5][0] for xx in params],
        'downs': [xx[5][1] for xx in params],
        'updowns_intype': [xx[6] for xx in params],
        })
    df.to_csv('data/sa_results_all'+suffix+'.csv',index=False)
    
    # hold steady states (last 1000 iterations)
    df_ss = pd.DataFrame({
        str(param):ss for param,ss in zip(params,[xx[5] for xx in sim_results])
        })
    df_ss.to_hdf('data/steady_state_all'+suffix+'.hdf','df')
    #%%
    # hold avg happiness levels of pop for each step
    df_hlvl = pd.DataFrame({
        str(param):hlvl + [0]*(1+num_steps-len(hlvl)) for param,hlvl in zip(params,[xx[2] for xx in sim_results])
        })
    df_hlvl.to_hdf('data/hlvl_all'+suffix+'.hdf','df')
    #%%
    # hold population obj info
    # df_popi = pd.DataFrame({})
    # df_pop = pd.DataFrame({})
    # for i,simres in enumerate(sim_results):
    #     df_popitmp = pd.DataFrame({})
    #     df_poptmp = pd.DataFrame({})
        
    #     popi = simres[0]
    #     pop = simres[1]
        
    #     df_popitmp['run'] = [str(params[i]) for xx in popi]
    #     df_popitmp['run_idx'] = [i for xx in popi]
    #     df_popitmp['agent_id'] = [xx.agent_id for xx in popi]
    #     df_popitmp['person_type'] = [xx.person_type for xx in popi]
    #     df_popitmp['friends'] = [[yy.agent_id for yy in xx.friends] for xx in popi]
        
    #     df_poptmp['run'] = [str(params[i]) for xx in pop]
    #     df_poptmp['run_idx'] = [i for xx in pop]
    #     df_poptmp['agent_id'] = [xx.agent_id for xx in pop]
    #     df_poptmp['person_type'] = [xx.person_type for xx in pop]
    #     df_poptmp['friends'] = [[yy.agent_id for yy in xx.friends] for xx in pop]
        
    #     df_popi = df_popi.append(df_popitmp)
    #     df_pop = df_pop.append(df_poptmp)
    
    # df_popi.to_csv('data/init_pop'+suffix+'.csv',index=False)
    # df_pop.to_csv('data/pop'+suffix+'.csv',index=False)
    #%%
    pool.close()
    pool.join()