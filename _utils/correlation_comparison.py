# -*- coding: utf-8 -*-
"""
Created on 2023-12-06 (Wed) 13:42:09

Correlation comparison
Reference: https://github.com/groovy-phazuma/ML_DL_Notebook/tree/main/Correlation_Comparison

@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

# %%
def Fisher_transformation(df=None,x='sepal length (cm)',y='petal length (cm)',p=0.05,verbose=True,do_plot=True):
    if df is None:
        raise ValueError('!! Set Data !!')
    
    n = len(df)
    r = np.corrcoef(df[x], df[y])[0,1]
    z = np.log((1 + r) / (1 - r)) / 2

    eta_min = z - stats.norm.ppf(q=1-p/2, loc=0, scale=1) / np.sqrt(n - 3)
    eta_max = z - stats.norm.ppf(q=p/2, loc=0, scale=1) / np.sqrt(n - 3)

    rho_min = (np.exp(2 * eta_min) - 1) / (np.exp(2 * eta_min) + 1)
    rho_max = (np.exp(2 * eta_max) - 1) / (np.exp(2 * eta_max) + 1)

    if verbose:
        print(r)
        print(f'95% confident interval: {rho_min}〜{rho_max}')

    if do_plot:
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

    return z

def Cor_diff_test(df1,df2,x='sepal length (cm)',y='petal length (cm)',verbose=True,do_plot=True):
    z1 = Fisher_transformation(df1,x=x,y=y,verbose=verbose,do_plot=do_plot)
    z2 = Fisher_transformation(df2,x=x,y=y,verbose=verbose,do_plot=do_plot)

    T = (z1 - z2) / np.sqrt((1/(len(df1)-3)) + (1/(len(df2) - 3)))
    p_value = stats.norm.cdf(T,loc=0,scale=1)

    return z1, z2, T, p_value

def permutation_test(df1,df2,x='sepal length (cm)',y='petal length (cm)',n_perm=1000,alternative="less",do_plot=True):
    if alternative in ['less','greater']:
        pass
    else:
        raise ValueError("!! Inappropriate alternative type !!")
    original_t,original_p = ft.Cor_diff_test(df1,df2,x=x,y=y,verbose=False)

    n1 = len(df1)
    concat_df = pd.concat([df1,df2])

    # permutation
    perm_res = [original_t]
    for i in tqdm(range(n_perm)):
        shuffle_df = concat_df.sample(frac=1, random_state=i)
        u_df1 = shuffle_df.iloc[0:n1,:]
        u_df2 = shuffle_df.iloc[n1:,:]
        ut,up = ft.Cor_diff_test(u_df1,u_df2,x=x,y=y,verbose=False)
        perm_res.append(ut)
    
    # calc p-value
    if alternative == "less":
        perm_res = sorted(perm_res)
    else:
         perm_res = sorted(perm_res,reverse=True)

    original_idx = perm_res.index(original_t)
    perm_p = original_idx / n_perm

    if do_plot:
        # visualization
        fig,ax = plt.subplots(figsize=(6,4))
        plt.hist(perm_res,bins=int(n_perm/10),alpha=0.9)
        plt.vlines(x=original_t,ymin=0,ymax=10,color="red",ls="dashed",linewidth=2)

        plt.xlabel('statistics value')
        plt.ylabel('frequency')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.show()

    return perm_p


