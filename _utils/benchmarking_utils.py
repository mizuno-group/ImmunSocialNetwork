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
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1 import make_axes_locatable

DPI = 100

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

def convert2binary(raw_df):
    fxn = lambda x : abs(x)
    df = raw_df.applymap(fxn)  # 
    mat = np.array(df)
    for i in range(mat.shape[0]):
        for j in range(i+1,mat.shape[1]):
            if mat[i][j] == mat[j][i]:
                mat[i][j] = 0
                mat[j][i] = 0
            elif mat[i][j] > mat[j][i]:
                mat[i][j] = 1
                mat[j][i] = 0
            else:
                mat[j][i] = 1
                mat[i][j] = 0
        mat[i][i] = 0  # Convert the diagonal component to 0.
    binary_df = pd.DataFrame(mat,index=df.index,columns=df.columns)
    return binary_df

def triu2flatten(df:pd.DataFrame):
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

def eval_with_w(adj_df,W,do_abs=True,title="[]",weight_threshold=0.3,common=[],W_minmax=True,do_plot=True,
                xlabel='sc RNA-Seq CCC Reference',ylabel='Estimated Adj Weight'):
    res_range = adj_df.index.tolist()
    W_pro = pd.DataFrame(np.zeros((adj_df.shape[1], adj_df.shape[1])),index=res_range,columns=res_range)
    if len(common) == 0:
        common = list(set(res_range) & set(W.index))
    for c1 in common:
        for c2 in common:
            if c1 == c2:  # Ignore diagonal components
                pass
            else:
                if do_abs:
                    W_pro[c1].loc[c2] = abs(W[c1].loc[c2])  # abs
                else:
                    W_pro[c1].loc[c2] = W[c1].loc[c2]

    # minmax scaling
    if W_minmax:
        vmax = W_pro.max().max()
        vmin = W_pro.min().min()
        fxn = lambda x : (x-vmin)/(vmax-vmin)
        final_W = W_pro.applymap(fxn)
    else:
        final_W = W_pro

    # intersecion cell types
    common = sorted(common)
    adj_common = adj_df[common].loc[common]
    W_common = final_W[common].loc[common]
    adj_melt = pair_melt(adj_common)
    w_melt = pair_melt(W_common)

    if do_plot:
        vis_dual_heatmaps(W_common, adj_common)

    # scatter plots for overall relationships
    y_true = w_melt['value'].tolist()
    y_score = adj_melt['value'].tolist()

    # scatterplots with distribution
    y_true_posi, y_score_posi, overall_score = customized_scatter_plot(y_true, y_score, xlabel, ylabel, title=title, weight_threshold=weight_threshold, do_plot=False)
    local_score = plot_sns_scatter(y_true_posi, y_score_posi, 
                                   xlabel=xlabel,ylabel=ylabel,
                                   title=title,do_plot=do_plot,dpi=DPI)

    return adj_melt, w_melt, overall_score, local_score


def vis_dual_heatmaps(W_common, adj_common):

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI,
                           gridspec_kw={"wspace": 0.75})

    cmaps = ["Oranges", "Purples"]
    titles = ["Estimated Weight Matrix", "Reference Adjacency Matrix"]

    xlabel = "Target"
    ylabel = "Source"

    for i, (mat, cmap) in enumerate(zip([W_common, adj_common], cmaps)):

        # seaborn heatmap（cbar=False）で描画
        hm = sns.heatmap(
            mat,
            ax=ax[i],
            cmap=cmap,
            square=True,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            linewidths=0.4,
            linecolor="white",
            cbar=False
        )

        # 軸ラベルなど
        ax[i].set_xlabel(xlabel, fontsize=12)
        ax[i].set_ylabel(ylabel, fontsize=12)
        ax[i].set_title(titles[i], fontsize=14, pad=10)

        # スパインを細く
        for spine in ax[i].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("gray")

        # --- ★ 等高 colorbar を右に配置 ★ ---
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="4%", pad=0.1)
        plt.colorbar(hm.collections[0], cax=cax)

    # tight_layout は square を崩す → 使用禁止
    fig.subplots_adjust(
        left=0.06, right=0.95,
        top=0.92, bottom=0.10,
        wspace=0.25
    )

    plt.show()

def eval_confusion_matrix(w_res, a_res, do_plot=True):
    accuracy = round(metrics.accuracy_score(w_res,a_res),3)
    precision = round(metrics.precision_score(w_res,a_res),3)
    recall = round(metrics.recall_score(w_res,a_res),3)
    f1 = round(metrics.f1_score(w_res,a_res),3)

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    cm = metrics.confusion_matrix(w_res, a_res)
    if do_plot:
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.xlabel("Estimated label")
        plt.ylabel("True label")
        plt.title(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        plt.show()

    score_dict = {"cm":cm,"accuracy":accuracy,"precision":precision,"recall":recall,"f1":f1}

    return score_dict


# %% Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def customized_scatter_plot(y_true, y_score, xlabel, ylabel, title, weight_threshold, do_plot=True):
    """
    Creates an scatter plot with customized aesthetics.
    
    Parameters:
    - y_true: List of true values
    - y_score: List of predicted scores
    - xlabel: Label for the x-axis
    - ylabel: Label for the y-axis
    - title: Title of the plot
    - total_cor: Correlation coefficient to be displayed
    - rmse: RMSE value to be displayed
    - weight_threshold: Threshold for distinguishing colors
    - do_plot: Boolean flag to show plot
    """

    y_true_posi = []
    y_score_posi = []
    c = []

    # Assign colors based on the threshold
    for i in range(len(y_score)):
        if abs(y_score[i]) > weight_threshold:
            c.append('tab:orange')
            y_score_posi.append(y_score[i])
            y_true_posi.append(y_true[i])
        else:
            c.append('tab:blue')

    total_cor, pvalue = stats.pearsonr(y_score_posi,y_true_posi)
    rmse = np.sqrt(mean_squared_error(y_score_posi, y_true_posi))
    if pvalue < 0.01:
        pvalue = '{:.2e}'.format(pvalue)
    else:
        pvalue = round(pvalue,3)
    
    if do_plot:
        # Set the style and color palette
        sns.set(style="whitegrid", context="talk")
        palette = sns.color_palette(["#1f77b4", "#ff7f0e"])  # Define consistent color scheme

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(6, 5),dpi=DPI)
        
        # Plot scatter with custom colors and aesthetics
        scatter = ax.scatter(y_true, y_score, color=c, s=100, edgecolor='w', alpha=0.8)

        # Set axis labels
        ax.set_xlabel(xlabel, fontsize=16, weight='bold')
        ax.set_ylabel(ylabel, fontsize=16, weight='bold')

        # Place text in the bottom right
        ax.text(0.95, 0.15, f'R = {round(total_cor, 3)}', 
                fontsize=14, weight='semibold', 
                horizontalalignment='right', transform=ax.transAxes)
        ax.text(0.95, 0.1, f'RMSE = {round(rmse, 3)}', 
                fontsize=14, weight='semibold', 
                horizontalalignment='right', transform=ax.transAxes)
        plt.text(0.95, 0.05, f'P = {pvalue}', 
                    fontsize=14, weight='semibold', 
                    horizontalalignment='right', transform=ax.transAxes, fontstyle='italic')

        # Customize spines, ticks and grids
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.grid(color="#ababab", linewidth=0.5)
        ax.set_axisbelow(True)

        ax.set_title(title, fontsize=16, weight='bold')
        plt.show()
        
    overall_score = {"R":round(total_cor,3),"RMSE":round(rmse,3)}
    return y_true_posi, y_score_posi, overall_score


def plot_sns_scatter(y_true:list, y_score:list, 
                     xlabel="sc RNA-Seq CCC Reference",
                     ylabel="Estimated Adj Weight",
                     title="",do_plot=True, dpi=100):
    assert len(y_true)==len(y_score), "! y_true and y_score must be the same length !"

    # Pearson correlation
    total_cor, pvalue = stats.pearsonr(y_score,y_true)
    rmse = np.sqrt(mean_squared_error(y_score, y_true))  # RMSE
    if pvalue < 0.01:
        pvalue = '{:.2e}'.format(pvalue)
    else:
        pvalue = round(pvalue,3)

    tmp_df = pd.DataFrame({"y_true":y_true,"y_score":y_score})

    if do_plot:
        # style settings
        sns.set(style="whitegrid", context="talk")
        palette = sns.color_palette("deep")

        # scatter plots and regression line
        plt.figure(figsize=(6, 5),dpi=dpi)
        ax = sns.regplot(data=tmp_df, x="y_true", y="y_score", color=palette[0])

        # Set text alignment to bottom right.
        plt.text(0.95, 0.15, f'R = {round(total_cor, 3)}', 
                fontsize=14, color='black', weight='semibold', 
                horizontalalignment='right', transform=ax.transAxes)
        plt.text(0.95, 0.1, f'RMSE = {round(rmse, 3)}', 
                fontsize=14, color='black', weight='semibold', 
                horizontalalignment='right', transform=ax.transAxes)
        plt.text(0.95, 0.05, f'P = {pvalue}', 
                fontsize=14, color='black', weight='semibold', 
                horizontalalignment='right', transform=ax.transAxes, fontstyle='italic')

        # set axis labels
        plt.xlabel(xlabel, fontsize=16, weight='bold')
        plt.ylabel(ylabel, fontsize=16, weight='bold')

        # grid settings
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.grid(color="#ababab", linewidth=0.5)
        plt.title(title, fontsize=18, weight='bold')
        plt.show()
    
    # score
    score_dict = {"R":round(total_cor,3),"P":pvalue,"RMSE":round(rmse,3)}
    return score_dict


# %% legacy codes
def plot_sns_scatter_legacy(y_true:list, y_score:list, 
                     xlabel="sc RNA-Seq CCC Reference",ylabel="Estimated Adj Weight",title="",c="tab:blue", do_plot=True):
    assert len(y_true)==len(y_score), "! y_true and y_score must be the same length !"

    total_cor, pvalue = stats.pearsonr(y_score,y_true)  # Pearson correlation
    rmse = np.sqrt(mean_squared_error(y_score, y_true))  # RMSE
    if pvalue < 0.01:
        pvalue = '{:.2e}'.format(pvalue)
    else:
        pvalue = round(pvalue,3)

    tmp_df = pd.DataFrame({"y_true":y_true,"y_score":y_score})
    
    if do_plot:
        ax = sns.jointplot(data=tmp_df,x="y_true",y="y_score",kind="reg",color=c)
        xmax, xmin = max(y_true), min(y_true)
        ymax, ymin = max(y_score), min(y_score)

        plt.text(xmin+.8*(xmax-xmin),ymin+0.35*(ymax-ymin),'R = {}'.format(str(round(total_cor,3))))
        plt.text(xmin+.8*(xmax-xmin),ymin+0.30*(ymax-ymin),'P = {}'.format(str(pvalue)))
        plt.text(xmin+.8*(xmax-xmin),ymin+0.25*(ymax-ymin),'RMSE = {}'.format(str(round(rmse,3))))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    score_dict = {"R":round(total_cor,3),"P":pvalue,"RMSE":round(rmse,3)}
    return score_dict
