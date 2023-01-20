# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 22:59:03 2022

@author: strel

This script is used to analyze results from sensitivity analysis 1.

The visuals are hard coded for the results of that particular analysis, with the formatting and
axis windows adjusted for better graphics in the final paper.

Once the script has run, make the window fullscreen, select the options in the matplotlib window
and select "tight layout" to obtain the same layout ratio as seen in the paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv('data/sa_results_all_0_12.csv')
ss1 = pd.read_hdf('data/steady_state_all_0_12.hdf')
df2 = pd.read_csv('data/sa_results_all_12_24.csv')
ss2 = pd.read_hdf('data/steady_state_all_12_24.hdf')
df3 = pd.read_csv('data/sa_results_all_24_30.csv')
ss3 = pd.read_hdf('data/steady_state_all_24_30.hdf')

df = pd.concat([pd.concat([df1,df2],axis=0),df3],axis=0)
ss = pd.concat([pd.concat([ss1,ss2],axis=1),ss3],axis=1)
seeds = df.seed.unique().tolist()


# create placeholder happiness column to fill with values
df['happiness'] = 0
df['happiness_norm'] = 0
df['happiness_perc_diff'] = 0

df_igt = df.loc[df.updowns_intype==True] # df with event empathy ingroup
df_igf = df.loc[df.updowns_intype==False] # df with event empathy all

igt_vals = {}
igf_vals = {}

#%% Plot raw data from sa1

for df,stitle in zip((df_igt,df_igf),('In-type','All')):
    df.reset_index(inplace=True,drop=True)
    
    ig = True
    if stitle=='All':
        ig = False
    
    params = []
    for nrow,row in df.iterrows():
        ss_idx = [row.nfriends,row.div_pres,row.empathy,row.ups,row.downs,row.updowns_intype,[],[],row.seed]
        ss_idx[2] = "'{}'".format(ss_idx[2])
        param = "[100, {}, 20000, {}, {}, [{}, {}], {}, {}, {}, {}]".format(*ss_idx)
        params.append(params)
        avg_hlvl = ss[param].mean()
        max_ss_idx = [row.nfriends,row.div_pres,row.empathy,row.ups,0.0,False,[],[],row.seed]
        max_ss_idx[2] = "'{}'".format(max_ss_idx[2])
        max_avg_hlvl = ss["[100, {}, 20000, {}, {}, [{}, {}], {}, {}, {}, {}]".format(*max_ss_idx)].mean()
        df.happiness.iloc[nrow] = avg_hlvl
        df.happiness_norm.iloc[nrow] = avg_hlvl/max_avg_hlvl
        
    nrows = 3
    ncols = 3
    fig,ax = plt.subplots(nrows=nrows,ncols=ncols)
    max_h = {5:3.61,10:6.74,20:12.19}
    row_ymins = [2.5,5,9]
    row_ymaxs = [3.85,7.1,12.85]
    col_lbls = df.empathy.unique().tolist()
    row_lbls = df.nfriends.unique().tolist()
    for r in range(nrows):
        clbl = row_lbls[r]
        for c in range(ncols):
            rlbl = col_lbls[c]
            cax = ax[r][c]
            cax.annotate('Max Happiness',xy=(0.05,max_h[clbl]),xytext=(.0497,max_h[clbl]*1.01))
            
            dplot = df.loc[(df.empathy==rlbl) & (df.nfriends==clbl) & (df.updowns_intype==ig)]
            
            plot_lines = dplot.div_pres.unique().tolist()
            for line in plot_lines:
                data = dplot.loc[dplot.div_pres==line]
                x = data.downs
                y = data.happiness_norm
                y_raw = data.happiness
                
                err_df = pd.DataFrame({'x':x.values,'y':y.values,'y_raw':y_raw.values})
                x = x.unique().tolist()
                ymin,ymax,yavg,ystd = [],[],[],[]
                for val in x:
                    curr_y = err_df.loc[err_df.x==val,'y']
                    ymin.append(curr_y.min())
                    ymax.append(curr_y.max())
                    ystd.append(curr_y.std())
                    yavg.append(curr_y.mean())
                    # record values for percent change later
                    curr_y = err_df.loc[err_df.x==val,'y_raw']
                    if stitle == 'In-type':
                        if (rlbl,clbl,val,line) in list(igt_vals.keys()):
                            igt_vals[(rlbl,clbl,val,line)] = igt_vals[(rlbl,clbl,val,line)] + curr_y.tolist()
                        else:
                            igt_vals[(rlbl,clbl,val,line)] = curr_y.tolist()
                    elif stitle == 'All':
                        if (rlbl,clbl,val,line) in list(igf_vals.keys()):
                            igf_vals[(rlbl,clbl,val,line)] = igf_vals[(rlbl,clbl,val,line)] + curr_y.tolist()
                        else:
                            igf_vals[(rlbl,clbl,val,line)] = curr_y.tolist()
                        
                y = np.array(yavg)
                ystd = np.array(ystd)
                cax.plot(x,y,linestyle='-',label=line)
                cax.fill_between(x,y-ystd*2,y+ystd*2,alpha=0.25)
                # else:
                #     cax.scatter(x,y,label=str(line),marker='x')
                
            cax.set_title('Time Empathy:'+rlbl.capitalize()+' | nFriends:'+str(clbl),fontsize=16)
            cax.grid()
            cax.set_ylim(0.72,1.01)
            cax.set_xlim(0.05,0.01)
            cax.set_xticks(np.arange(.05,-0.001,-0.01))
            cax.set_xticklabels(np.arange(0,101,20))
            cax.tick_params(axis='both', which='major', labelsize=12)
            cax.set_xlabel('Event Empathy (%)',fontsize=12)
            cax.set_ylabel('Happiness Level',fontsize=12)
    lines, labels = cax.get_legend_handles_labels()
    fig.legend(lines,labels,title='Max Invite Probability',fontsize='medium',
                loc='upper right',title_fontsize='medium',ncol=5,bbox_to_anchor=(0.95,0.99))
    fig.suptitle('Event Empathy Type: '+stitle,fontsize=32)


#%% get diff b/w empathy application types

nrows = 3
ncols = 3
fig2,ax2 = plt.subplots(nrows=nrows,ncols=ncols)
fig2.suptitle('Percentage Change in Happiness from Event Empathy Type "In-Type" to "All"',fontsize=32)
max_h = {5:3.61,10:6.74,20:12.19}
col_lbls = df.empathy.unique().tolist()
row_lbls = df.nfriends.unique().tolist()
for r in range(nrows):
    clbl = row_lbls[r]
    for c in range(ncols):
        rlbl = col_lbls[c]
        cax = ax2[r][c]
        cax.hlines(0,-1,6,color='black')
        
        dplot = df.loc[(df.empathy==rlbl) & (df.nfriends==clbl)]
        plot_lines = dplot.div_pres.unique().tolist()
        vals = []
        for iline,line in enumerate(plot_lines):
            data = dplot.loc[dplot.div_pres==line]
            fullys = []
            ys,yerr = [],[]
            xs = []
            for down in data.downs[:-1].unique():
                x = np.round(down + .002 - .001*iline,3)
                igt = np.array(igt_vals[(rlbl,clbl,down,line)])
                igf = np.array(igf_vals[(rlbl,clbl,down,line)])
                diff = igf - igt
                per_diff = np.divide(diff,igt)
                avg_per_diff = np.average(per_diff)
                std_per_diff = np.std(per_diff)
                y = avg_per_diff * 100
                xs.append(x)
                ys.append(y)
                yerr.append(std_per_diff*200)
                fullys.append(100*per_diff)
                x_list = [down]*len(per_diff)
                for xi,yi in zip(x_list,per_diff):
                    xiconv = 100 - xi*100*20
                    vals.append([xiconv,100*yi,str(line)])
                
        box_df = pd.DataFrame({
            "x":[xx[0] for xx in vals],
            "y":[xx[1] for xx in vals],
            "line":[xx[2] for xx in vals],
            })
        g = sns.boxplot(data=box_df,x='x',y='y',hue='line',ax=cax,orient='v',
                        width=.75,linewidth=1.1,medianprops=dict(color='yellow'))
        g.legend_.remove()
        cax.set_title('Time Empathy:'+rlbl.capitalize()+' | nFriends:'+str(clbl),fontsize=14)
        cax.grid()
        cax.set_axisbelow(True)
        ymin,ymax = -2,3
        # ymin,ymax = -9,14
        cax.set_ylim(ymin,ymax)
        cax.set_yticks(np.arange(ymin,ymax+.001,0.5))
        cax.set_axisbelow(True)
        cax.set_xlabel('Event Empathy (%)', fontsize=12)
        cax.set_ylabel('Percent Change (%)', fontsize=12)
        cax.tick_params(axis='both', which='major', labelsize=12)
lines, labels = fig2.axes[-1].get_legend_handles_labels()
fig2.legend(lines,labels,title='Max Invite Probability',fontsize='medium',
            loc='upper center',title_fontsize='medium',ncol=5,bbox_to_anchor=(0.5,0.95))
