# -*- coding: utf-8 -*-
"""
Created on 2023-12-20 (Wed) 11:49:19

data loader

@author: I.Azuma
"""
# %%
import collections
import numpy as np
import pandas as pd

from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
print(BASE_DIR)

import sys
sys.path.append(BASE_DIR)
from _utils import archs4_utils as a4u
from _utils import processing as pc

# %%
class ImmneNet_Dataloader():
    def __init__(self,info_path=None,raw_info=None):
        if info_path is None:
            self.raw_info = raw_info
        else:
            self.raw_info = pd.read_csv(info_path,index_col=0)

        """
        	geo_accession	series_id	characteristics_ch1                                 source_name_ch1     singlecellprobability	treatment
            GSM1067318	    GSE43631	TISSUE: LIVER,GENOTYPE: CTKO,CELL CYCLE STATE:...	LIVER	            0.001154	            HRS AFTER CCL4
            GSM1067319	    GSE43631	TISSUE: LIVER,GENOTYPE: CTKO,CELL CYCLE STATE:...	LIVER	            -0.006098	            36 HRS AFTER 2/3RD PARTIAL HEPATECTOMY
            GSM1067324	    GSE43631	TISSUE: LIVER,GENOTYPE: CTKO;ALBCREER/+,CELL C...	LIVER	            0.003910	            2 WEEKS AFTER TAMOXIFEN
            GSM1067325	    GSE43631	TISSUE: LIVER,GENOTYPE: CTKO;ALBCREER/+,CELL C...	LIVER	            0.003910	            2 WEEKS AFTER TAMOXIFEN
            GSM1088288	    GSE44640	AGE: 28 WKS AT THE TIME OF RNA ISOLATION,BACKG...	LIVER_CZ_9V/NULL	0.000516	            CZ (IMIGLUCERASE)
        """

    def sample_selection(self,condition:list=[],accession_list:list=[]):
        if len(condition) > 0:
            target_info = self.raw_info[self.raw_info['treatment'].isin(condition)]
        elif len(accession_list) > 0:
            target_info = self.raw_info[self.raw_info['geo_accession'].isin(accession_list)]

        # selection based on batch size
        batchsize_threshold = 3
        count_dict = collections.Counter(target_info['series_id'].tolist())
        selected_series = []
        for i,k in enumerate(count_dict):
            if count_dict.get(k)>=batchsize_threshold:
                selected_series.append(k)
            else:
                pass

        target_info = target_info[target_info['series_id'].isin(selected_series)]
        self.target_info = target_info
        self.target_samples = target_info.index.tolist() # 183
        print(len(self.target_samples),"samples were selected.")
        self.target_info = target_info
    
    def load_expression(self,tpm_path='/workspace/datasource/ARCHS4/mouse_tpm_v2.2.h5',ref_path='/workspace/datasource/Biomart/mouse_transcriptID2MGI.csv',gene_list_path='/workspace/datasource/MsigDB/231204_mouse_gene_lst.pickle',processing=True):
        # load expression data
        df = a4u.samples_local(file=tpm_path, sample_ids=self.target_samples, row_type='transcript')

        # annotation (for transcript data)
        ref_df = pd.read_csv(ref_path,index_col=0)
        ann_df = pc.annotation(df,ref_df,method='median')
        ann_df.index = [t.upper() for t in ann_df.index.tolist()]

        # load mouse gene list
        mouse_gene_list = pd.read_pickle(gene_list_path)
        mouse_gene_list = [t.upper() for t in mouse_gene_list]
        common = sorted(list(set(ann_df.index.tolist()) & set(mouse_gene_list)))
        self.ann_df = ann_df.loc[common]

        if processing:
            # remove redundant genes
            trim_df = self.ann_df.replace(0.0,np.nan).dropna(how='all').replace(np.nan,0) # (18266, 183)
            # 1. combat
            try:
                series_list = self.target_info.loc[trim_df.columns.tolist()]['series_id'].tolist()
                series_batch = pc.batch_id_converter(series_list)

                comb_df = pc.batch_norm(trim_df,series_batch)
                df = comb_df.dropna(how='all',axis=1)
            except:
                print("Bathc Normalization was not conducted.")

            # trimming
            fxn = lambda x : 0 if x < 0 else x
            df = df.applymap(fxn)

            # 2. QN
            qn_df = pc.quantile(df) # (18266, 183)

            return qn_df
        else:
            pass
