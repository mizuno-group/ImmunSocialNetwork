# -*- coding: utf-8 -*-
"""
Created on 2024-05-09 (Thu) 14:22:39

Utils for benchmarking of skeleton estimation methods.

@author: I.Azuma
"""
# %%
import copy
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error

def g2array(g):
    return nx.adjacency_matrix(g).todense()

def pair_melt(orig_df):
    """
    Args:
        df (dataframe): Adjacency matrix
                            Hepatocyte  Hepatoblast  Cholangiocyte
        Hepatocyte            0.0     1.683933            0.0
        Hepatoblast           0.0     0.000000            0.0
        Cholangiocyte         0.0     1.549065            0.0

    Returns:
        dataframe: Melted dataframe as follows:
                Source         Target     value
        8  Cholangiocyte  Cholangiocyte  0.000000
        5  Cholangiocyte    Hepatoblast  1.549065
        2  Cholangiocyte     Hepatocyte  0.000000
        7    Hepatoblast  Cholangiocyte  0.000000
        4    Hepatoblast    Hepatoblast  0.000000
        1    Hepatoblast     Hepatocyte  0.000000
        6     Hepatocyte  Cholangiocyte  0.000000
        3     Hepatocyte    Hepatoblast  1.683933
        0     Hepatocyte     Hepatocyte  0.000000
    """
    df = copy.deepcopy(orig_df)
    df['Source']=df.index.tolist()
    df_melt = df.melt(id_vars='Source',var_name='Target').sort_values('value')
    df_melt = df_melt.sort_values(['Source','Target'])
    
    return df_melt

def convert2binary(df):
    mat = np.array(df)
    for i in range(mat.shape[0]):
        for j in range(i+1,mat.shape[1]):
            if mat[i][j] > mat[j][i]:
                mat[i][j] = 1
                mat[j][i] = 0
            else:
                mat[j][i] = 1
                mat[i][j] = 0
    binary_df = pd.DataFrame(mat,index=df.index,columns=df.columns)
    return binary_df

def triu2flatten(df):
    mat = np.array(df)
    res = []
    st_list = []
    for i in range(mat.shape[0]):
        for j in range(i+1,mat.shape[1]):
            res.append(mat[i][j])
            st_list.append((df.index.tolist()[i], df.columns.tolist()[j]))
    return res, st_list

def mat2flatten(df):
    mat = np.array(df)
    res = []
    st_list = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == j:
                pass
            else:
                res.append(mat[i][j])
                st_list.append((df.index.tolist()[i], df.columns.tolist()[j]))
    return res, st_list

# %% Visualization

def plot_scatter(y_true:list, y_score:list, xlabel="sc RNA-Seq CCC Reference",ylabel="Estimated Adj Weight",title="",c="tab:blue"):
    assert len(y_true)==len(y_score), "! y_true and y_score must be the same length !"

    total_cor, pvalue = stats.pearsonr(y_score,y_true)  # Pearson correlation
    rmse = np.sqrt(mean_squared_error(y_score, y_true))  # RMSE
    if pvalue < 0.01:
        pvalue = '{:.2e}'.format(pvalue)
    else:
        pvalue = round(pvalue,3)

    fig,ax = plt.subplots(figsize=(5,4))
    plt.scatter(y_true,y_score,color=c)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.text(.8,0.15,'R = {}'.format(str(round(total_cor,3))), transform=ax.transAxes)
    plt.text(.8,0.10,'P = {}'.format(str(pvalue)), transform=ax.transAxes)
    plt.text(.8,0.05,'RMSE = {}'.format(str(round(rmse,3))), transform=ax.transAxes)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    ax.set_axisbelow(True)
    ax.grid(color="#ababab",linewidth=0.5)
    plt.title(title)
    plt.show()

def plot_sns_scatter(y_true:list, y_score:list, xlabel="sc RNA-Seq CCC Reference",ylabel="Estimated Adj Weight",title="",c="tab:blue"):
    assert len(y_true)==len(y_score), "! y_true and y_score must be the same length !"

    total_cor, pvalue = stats.pearsonr(y_score,y_true)  # Pearson correlation
    rmse = np.sqrt(mean_squared_error(y_score, y_true))  # RMSE
    if pvalue < 0.01:
        pvalue = '{:.2e}'.format(pvalue)
    else:
        pvalue = round(pvalue,3)

    tmp_df = pd.DataFrame({"y_true":y_true,"y_score":y_score})
    
    ax = sns.jointplot(data=tmp_df,x="y_true",y="y_score",kind="reg",color=c)
    xmax, xmin = max(y_true), min(y_true)
    ymax, ymin = max(y_score), min(y_score)

    plt.text(xmin+.8*(xmax-xmin),ymin+0.35*(ymax-ymin),'R = {}'.format(str(round(total_cor,3))))
    plt.text(xmin+.8*(xmax-xmin),ymin+0.30*(ymax-ymin),'P = {}'.format(str(pvalue)))
    plt.text(xmin+.8*(xmax-xmin),ymin+0.25*(ymax-ymin),'RMSE = {}'.format(str(round(rmse,3))))
    plt.title(title)
    plt.show()
