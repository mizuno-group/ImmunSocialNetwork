# -*- coding: utf-8 -*-
"""
Created on 2024-04-04 (Thu) 14:13:21

Augmentation for deconvolution on small sample size.

@author: I.Azuma
"""
# %%
import random
import itertools
import numpy as np
import pandas as pd

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent

import sys
sys.path.append(str(BASE_DIR))
from _utils import processing

# %%
class Aug4Deconv():
    def __init__(self):
        self.df = None
        self.__processing = processing

    def set_data(self,df,marker_dic:dict):
        df.index = [t.upper() for t in df.index.tolist()]
        upper_v = []
        for i,k in enumerate(marker_dic):
            upper_v.append([t.upper() for t in marker_dic.get(k)])
        new_dic = dict(zip(list(marker_dic.keys()), upper_v))
        self.df = df
        self.marker_dic = new_dic
    
    def calc_cv(self,outlier=True):
        self.marker_genes = list(itertools.chain.from_iterable(list(self.marker_dic.values())))
        other_genes = sorted(list(set(self.df.index) - set(self.marker_genes)))
        other_df = self.df.loc[other_genes]

        if outlier:
            log_df = self.__processing.log2(other_df)
            common = set(log_df.index.tolist())
            for sample in log_df.columns.tolist():
                log_sample = log_df[sample].replace(0,np.nan).dropna()
                mu = log_sample.mean()
                sigma = log_sample.std()
                df3 = log_sample[(mu - 2*sigma <= log_sample) & (log_sample <= mu + 2*sigma)]
                common = common & set(df3.index.tolist())
            target_df = other_df.loc[sorted(list(common))]
        else:
            target_df = other_df
        
        self.other_df = other_df
        self.target_df = target_df

        var_df = pd.DataFrame(target_df.T.var())
        mean_df = pd.DataFrame(target_df.T.mean())
        cv_df = var_df/mean_df # coefficient of variance

        self.cv_df = cv_df.sort_values(0,ascending=False)
    
    def build_block(self,marker_genes:list,other_genes:list):
        # e.g. marker + high CV genes
        target_genes = sorted(list(set(marker_genes) | set(other_genes)))
        block_df = self.df.loc[self.df.index.isin(target_genes)]

        return block_df
    
    def stack_multi_blocks(self,marker_genes=None,cv_size=100,stack_n=5,allow_overlap=False):
        if marker_genes is None:
            marker_genes = self.marker_genes
        candi_size = cv_size*stack_n
        high_cv_genes = self.cv_df.index.tolist()[0:candi_size]

        random.seed(123)
        shuffled_genes = random.sample(high_cv_genes,len(high_cv_genes)) # shuffle

        stacked_df = pd.DataFrame()
        start = 0
        for sn in range(stack_n):
            if allow_overlap:
                random.seed(sn)
                add_genes = random.sample(shuffled_genes,cv_size)
            else:
                add_genes = shuffled_genes[start:start+cv_size]
            # collect each block
            tmp_df = self.build_block(marker_genes,add_genes)
            re_col = [t+'_{}'.format(str(sn)) for t in tmp_df.columns]
            tmp_df.columns = re_col

            stacked_df = pd.concat([stacked_df,tmp_df],axis=1)

            start += cv_size # shift
        
        self.other_genes = sorted(list(set(stacked_df.index) - set(marker_genes))) 
        self.stacked_df = stacked_df

        