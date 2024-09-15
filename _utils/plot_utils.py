# -*- coding: utf-8 -*-
"""
Created on 2023-06-07 (Wed) 10:07:04

@author: I.Azuma
"""
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from logging import getLogger
logger = getLogger('plot_utils')

def plot_scatter(df,x='Macrophage',y='Dendritic cell'):
    fig,ax = plt.subplots(figsize=(6,6))
    plt.scatter(df[x].tolist(),df[y].tolist(),alpha=0.9)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    plt.show()

def plot_multi(data=[[11,50,37,202,7],[47,19,195,117,74],[136,69,33,47],[100,12,25,139,89]],
               names=["A","B","C","D"], value="ALT (U/I)", title="", grey=True, dpi=300,
               figsize=(8,6), lw=1.5, capthick=1.5, capsize=6):
    sns.set(style="whitegrid")
    #plt.rcParams['font.family'] = 'serif'
    
    if grey:
        sns.set_palette('Greys')
    else:
        sns.set_palette('coolwarm')

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    df = pd.DataFrame()
    for i in range(len(data)):
        tmp_df = pd.DataFrame({names[i]: data[i]})
        df = pd.concat([df, tmp_df], axis=1)

    error_bar_set = dict(lw=lw, capthick=capthick, capsize=capsize)
    ax.bar([i for i in range(len(data))], df.mean(), yerr=df.std(), tick_label=df.columns, 
           error_kw=error_bar_set, color=sns.color_palette())

    # Jitter plot with larger markers and edgecolor
    df_melt = pd.melt(df)
    sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, size=6, 
                  edgecolor='black', linewidth=0.6, ax=ax)
    
    # Draw lines between corresponding jitter points with slightly thicker lines
    """
    for i in range(len(data[0])):
        y_values = [data[j][i] for j in range(len(data)) if i < len(data[j])]
        ax.plot(range(len(y_values)), y_values, color='gray', linewidth=0.75, alpha=0.7)
    """
    for i in range(len(data[0])):
        x1, y1 = ax.collections[1].get_offsets()[i] 
        x2, y2 = ax.collections[2].get_offsets()[i]
        ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.75, alpha=0.7) 
        

    ax.set_xlabel('')
    ax.set_ylabel(value, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=10)
    
    # Add grid and adjust its appearance
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_radar(data=[[0.3821, 0.6394, 0.8317, 0.7524],[0.4908, 0.7077, 0.8479, 0.7802]],labels=['Neutrophils', 'Monocytes', 'NK', 'Kupffer'],conditions=['w/o addnl. topic','w/ addnl. topic'],title='APAP Treatment',dpi=100):
    # preprocessing
    dft = pd.DataFrame(data,index=conditions)

    # Split the circle into even parts and save the angles
    # so we know where to put each axis.
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    # and append the start value to the end.
    angles += angles[:1]

    # ax = plt.subplot(polar=True)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=dpi)

    # Helper function to plot each car on the radar chart.
    def add_to_radar(name, color):
        values = dft.loc[name].tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=0.25)
    # Add each car to the chart.
    add_to_radar(conditions[0], '#429bf4')
    add_to_radar(conditions[1], '#ec6e95')

    # Fix axis to go in the right order and start at 12 o'clock.
    ax.set_theta_offset(np.pi/4)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label.
    ax.set_thetagrids(np.degrees(angles)[0:len(labels)], labels)

    # Go through labels and adjust alignment based on where
    # it is in the circle.
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('left')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    # Ensure radar range
    #ax.set_ylim(0, 0.9)
    ax.set_rlabel_position(180 / len(labels))
    ax.tick_params(colors='#222222')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(color='#AAAAAA')
    ax.spines['polar'].set_color('#222222')
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, y=1.02, fontsize=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()

# %% Legacy
def plot_multi_legacy(data=[[11,50,37,202,7],[47,19,195,117,74],[136,69,33,47],[100,12,25,139,89]],names=["+PBS","+Nefopam","+Ketoprofen","+Cefotaxime"],value="ALT (U/I)",title="",grey=True,dpi=100,figsize=(12,6),lw=1,capthick=1,capsize=5):
    sns.set()
    sns.set_style('whitegrid')
    if grey:
        sns.set_palette('gist_yarg')
        
    fig,ax = plt.subplots(figsize=figsize,dpi=dpi)
    
    df = pd.DataFrame()
    for i in range(len(data)):
        tmp_df = pd.DataFrame({names[i]:data[i]})
        df = pd.concat([df,tmp_df],axis=1)
    error_bar_set = dict(lw=lw,capthick=capthick,capsize=capsize)
    if grey:
        ax.bar([i for i in range(len(data))],df.mean(),yerr=df.std(),tick_label=df.columns,error_kw=error_bar_set)
    else:
        ax.bar([i for i in range(len(data))],df.mean(),yerr=df.std(),tick_label=df.columns,error_kw=error_bar_set)
    # jitter plot
    df_melt = pd.melt(df)
    sns.stripplot(x='variable', y='value', data=df_melt, jitter=True, color='black', ax=ax, size=3)
        
    ax.set_xlabel('')
    ax.set_ylabel(value)
    plt.title(title)
    plt.xticks(rotation=60)
    plt.show()
