# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 22:59:03 2022

@author: strel

This script is created to analyze the results from sensitivity analysis 2

Many of the graph elements are set up specifically for the data shown in the final paper. This would
need to be adjusted for any other analysis run of similar type.

Once the script has run, make the window fullscreen, select the options in the matplotlib window
and select "tight layout" to obtain the same layout ratio as seen in the paper.
"""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

def multirun_metrics(hlvl,params):
    runs = hlvl[params]
    run_metrics = []
    for ncol in range(runs.shape[1]):
        runcol = runs.iloc[:,ncol]
        run_last = runcol[runcol>0][-1000:]
        run_ss = run_last.mean()
        run_metrics.append(run_ss)
    return run_metrics

def multirun_metrics_sparkline(hlvl,params,params_baseline):
    # baseline run
    runs = hlvl[params_baseline]
    run_metrics_bl = []
    for ncol in range(runs.shape[1]):
        runcol = runs.iloc[:,ncol]
        run_last = runcol[runcol>0][-1000:]
        run_ss = run_last.mean()
        run_metrics_bl.append(run_ss)
    # current run
    runs = hlvl[params]
    run_metrics = []
    for ncol in range(runs.shape[1]):
        runcol = runs.iloc[:,ncol]
        run_last = runcol[runcol>0][-1000:]
        run_ss = run_last.mean()
        run_metrics.append(run_ss)
        
    # get perc diff
    per_diff = [100 * (cr-bl)/bl for cr,bl in zip(run_metrics,run_metrics_bl)]
    
    return per_diff

df1 = pd.read_csv('data/select_agents_empathy_analysis_0_8.csv')
ss1 = pd.read_hdf('data/steady_state_sae_0_8.hdf')
df2 = pd.read_csv('data/select_agents_empathy_analysis_8_16.csv')
ss2 = pd.read_hdf('data/steady_state_sae_8_16.hdf')
df3 = pd.read_csv('data/select_agents_empathy_analysis_16_24.csv')
ss3 = pd.read_hdf('data/steady_state_sae_16_24.hdf')
df4 = pd.read_csv('data/select_agents_empathy_analysis_24_30.csv')
ss4 = pd.read_hdf('data/steady_state_sae_24_30.hdf')

df = pd.concat([pd.concat([pd.concat([df1,df2],axis=0),df3],axis=0),df4],axis=0)
ss = pd.concat([pd.concat([pd.concat([ss1,ss2],axis=1),ss3],axis=1),ss4],axis=1)

newcols = []
for cols in ss.columns:
    lst = ast.literal_eval(cols)
    lst_edit = [xx for xx in lst if xx != []]
    num_emp = len(lst_edit[7])
    newcols.append(str(lst_edit[:-1]+[num_emp]))

hlvl1 = pd.read_hdf('data2/hlvl_sae_0_8.hdf')
hlvl_ae1 = pd.read_hdf('data2/hlvl_ae_sae_0_8.hdf')
hlvl_ane1 = pd.read_hdf('data2/hlvl_ane_sae_0_8.hdf')
hlvl2 = pd.read_hdf('data/hlvl_sae_8_16.hdf')
hlvl_ae2 = pd.read_hdf('data/hlvl_ae_sae_8_16.hdf')
hlvl_ane2 = pd.read_hdf('data/hlvl_ane_sae_8_16.hdf')
hlvl3 = pd.read_hdf('data/hlvl_sae_16_24.hdf')
hlvl_ae3 = pd.read_hdf('data/hlvl_ae_sae_16_24.hdf')
hlvl_ane3 = pd.read_hdf('data/hlvl_ane_sae_16_24.hdf')
hlvl4 = pd.read_hdf('data/hlvl_sae_24_30.hdf')
hlvl_ae4 = pd.read_hdf('data/hlvl_ae_sae_24_30.hdf')
hlvl_ane4 = pd.read_hdf('data/hlvl_ane_sae_24_30.hdf')

hlvl = pd.concat([pd.concat([pd.concat([hlvl1,hlvl2],axis=1),hlvl3],axis=1),hlvl4],axis=1)
hlvl_ae = pd.concat([pd.concat([pd.concat([hlvl_ae1,hlvl_ae2],axis=1),hlvl_ae3],axis=1),hlvl4],axis=1)
hlvl_ane = pd.concat([pd.concat([pd.concat([hlvl_ane1,hlvl_ane2],axis=1),hlvl_ane3],axis=1),hlvl4],axis=1)

hlvl.rename(columns={cols:newcols[i] for i,cols in enumerate(hlvl.columns)},inplace=True)
hlvl_ae.rename(columns={cols:newcols[i] for i,cols in enumerate(hlvl_ae.columns)},inplace=True)
hlvl_ane.rename(columns={cols:newcols[i] for i,cols in enumerate(hlvl_ane.columns)},inplace=True)
ss.rename(columns={cols:newcols[i] for i,cols in enumerate(ss.columns)},inplace=True)


df['happiness_ae'] = 0
df['happiness_ae_std'] = 0
df['happiness_ae_sl'] = 0
df['happiness_ae_std_sl'] = 0

df['happiness_ane'] = 0
df['happiness_ane_std'] = 0
df['happiness_ane_sl'] = 0
df['happiness_ane_std_sl'] = 0

emp_agents_list = [[0,99]]
emp_ag = []
for i in range(1,49):
    bot = i
    top = 99-i
    emp_ag = emp_agents_list[-1] + [bot,top]
    emp_ag.sort()
    emp_agents_list.append(emp_ag)
# get data for plot
params = []
for nrow,row in df.iterrows():
    ss_idx = [row.nfriends,row.div_pres,row.empathy,row.ups,row.downs,row.updowns_ingroup,
              emp_agents_list[row.emp_agents//2-1],row.emp_agents]#,row.seed]
    ss_idx[2] = "'{}'".format(ss_idx[2])
    param = "[100, {}, 20000, {}, {}, [{}, {}], {}, {}, {}]".format(*ss_idx)
    params.append(ast.literal_eval(param))
    # empathetic
    ae_ss = multirun_metrics(hlvl_ae,param)
    avg_hlvl_ae = np.average(ae_ss)
    std_hlvl_ae = np.std(ae_ss)
    # non-empathetic
    ane_ss = multirun_metrics(hlvl_ane,param)
    avg_hlvl_ane = np.average(ane_ss)
    std_hlvl_ane = np.std(ane_ss)
    
    df.happiness_ae.iloc[nrow] = avg_hlvl_ae
    df.happiness_ae_std.iloc[nrow] = std_hlvl_ae
    df.happiness_ane.iloc[nrow] = avg_hlvl_ane
    df.happiness_ane_std.iloc[nrow] = std_hlvl_ane
    
    # get sparkline data
    param_bl = params[nrow%49] # there are 49 data points in the sparkline data for each line
    ae_sl = multirun_metrics_sparkline(hlvl_ae,param,str(param_bl))
    avg_hlvl_ae_sl = np.average(ae_sl)
    std_hlvl_ae_sl = np.std(ae_sl)
    # non-empathetic
    ane_sl = multirun_metrics_sparkline(hlvl_ane,param,str(param_bl))
    avg_hlvl_ane_sl = np.average(ane_sl)
    std_hlvl_ane_sl = np.std(ane_sl)
    
    df.happiness_ae_sl.iloc[nrow] = avg_hlvl_ae_sl
    df.happiness_ae_std_sl.iloc[nrow] = std_hlvl_ae_sl
    df.happiness_ane_sl.iloc[nrow] = avg_hlvl_ane_sl
    df.happiness_ane_std_sl.iloc[nrow] = std_hlvl_ane_sl

df_igt = df.loc[df.updowns_ingroup==True]
df_igt.reset_index(inplace=True,drop=True)
df_igf = df.loc[df.updowns_ingroup==False]
df_igf.reset_index(inplace=True,drop=True)

#%% plot single case -- only in-group empathy = False

fig2, cax = plt.subplots()
fig2.suptitle('Happiness Level of Empathetic Agents and Non-Empathetic Agents',fontsize=32)
plt.tight_layout()
labels = ['Empathetic Agent Happiness','Non-Empathetic Agent Happiness']
colors = ['black','b','orange','g','purple','r']
for window in range(6): # 6 total param space
    start = window*49 # 49 combinations of emp agents
    finish = (window+1)*49
    cols = [str(xx) for xx in params[start:finish]]
    one_ss = ss[cols]
    x = np.arange(4,0+len(cols)*2,2)
    pars = df_igf.iloc[start][:6].tolist()
    te = pars[3].capitalize()
    ee = 100 * (pars[4]-pars[5]) / pars[4]
    lbl = 'Time Empathy: {} | Event Empathy: {}%'.format(te,ee)
    lbl_add = [' | Apply Empathy',' | No Empathy']
    line_types = ['-','--']
    for itype,hap_type in enumerate(['happiness_ae','happiness_ane']):
        y = df_igf.iloc[start+1:finish-1][hap_type]
        y_std = df_igf.iloc[start+1:finish-1][hap_type+'_std']
        cax.plot(x,y,label=lbl+lbl_add[itype],linestyle=line_types[itype],c=colors[window])
        cax.fill_between(x,y-y_std*1,y+y_std*1,alpha=0.25,color=colors[window])
    cax.set_xlim(0,100)
    cax.set_xticks(np.arange(0,101,10))
    cax.legend(fontsize=10,loc='lower center',ncol=3)
    cax.set_xlabel('Pecentage of Agents Showing Empathy (%)')
    cax.set_ylabel('Happiness Level')
cax.grid()

#%% plot sparkline of difference from baseline

figsl, cax2 = plt.subplots()
figsl.suptitle('Percent Happiness Change from Zero Empathy Baseline',fontsize=32)
labels = ['Empathetic Agent Happiness','Non-Empathetic Agent Happiness']
for window in range(6): # 6 total param space
    start = window*49 # 49 combinations of emp agents
    finish = (window+1)*49
    cols = [str(xx) for xx in params[start:finish]]
    x = np.arange(4,0+len(cols)*2,2)
    pars = df_igf.iloc[start][:6].tolist()
    te = pars[3].capitalize()
    ee = 100 * (pars[4]-pars[5]) / pars[4]
    lbl = 'Time Empathy: {} | Event Empathy: {}%'.format(te,ee)
    lbl_add = [' | Apply Empathy',' | No Empathy']
    line_types = ['-','--']
    for itype,hap_type in enumerate(['happiness_ae','happiness_ane']):
        y = df_igf.iloc[start+1:finish-1][hap_type+'_sl']
        y_std = df_igf.iloc[start+1:finish-1][hap_type+'_std_sl']
        cax2.plot(x,y,label=lbl+lbl_add[itype],linestyle=line_types[itype],c=colors[window])
        cax2.fill_between(x,y-y_std*1,y+y_std*1,alpha=0.25,color=colors[window])
        
    cax2.hlines(0,0.0,100,linestyle='-',color='black')
    cax2.set_yticks(np.arange(-5,30,5))
    cax2.set_ylim(-7.5,28)
    cax2.set_xlim(0,100)
    cax2.set_xticks(np.arange(0,101,10))
    cax2.legend(fontsize=10,loc='lower center',ncol=3)
    cax2.set_xlabel('Pecentage of Agents Showing Empathy (%)')
    cax2.set_ylabel('Pecentage Happiness Change (%)')
cax2.grid()
