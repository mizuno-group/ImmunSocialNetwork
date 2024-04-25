# -*- coding: utf-8 -*-
"""
Created on 2023-12-15 (Fri) 11:06:37

Marker processing for CellMarker

@author: I.Azuma
"""
# %%
import pandas as pd
import codecs
import collections
import itertools

# %%
class CellMarkerHandler():
    def __init__(self,raw_path='/workspace/github/ImmunSocialNetwork/data/raw_info/Cell_marker_All.csv',species='Mouse'):
        self.raw_path = raw_path
        with codecs.open(self.raw_path, "r", "Shift-JIS", "ignore") as file:
            total_ref = pd.read_table(file, delimiter=",")
        self.total_ref = total_ref[total_ref["species"].isin(["Mouse"])]

        print(f'{len(total_ref)} marker genes were ready for {species}.')
    
    def narrow_samples(self,marker_source=['Experiment', 'Review', 'Single-cell sequencing', 'Company']):
        total_ref = self.total_ref
        # marker source selection
        tmp_ref = total_ref[total_ref['marker_source'].isin(marker_source)]
        print(f'{len(self.total_ref)} --> {len(tmp_ref)}')

        # update
        self.total_ref = tmp_ref

    
    def load_summary_table(self,summary_path='/path/to/Summary_Table_for_Liver.csv',
                                target_tissues=['Liver','Blood']):
        """
        Type	Sub-type	Registered Name
        0	Hepatocyte	Hepatocyte	Hepatocellular cell
        1	NaN	NaN	Hepatocyte
        2	NaN	NaN	Mature hepatocyte
        3	NaN	Hepatoblast	Hepatoblast
        4	Bile duct	Bile duct	Biliary cell
        ...	...	...	...
        86	NaN	NaN	Memory-like CD8+ T cell
        87	NaN	Regulatory T (Treg) cell	Regulatory T (Treg) cell
        88	NaN	NaN	Regulatory T(Treg) cell
        89	NaN	NaN	Induced regulatory T (Treg) cell
        90	NaN	NaN	Activated regulatory T cell

        Args:
            summary_path (str, optional): Path to summary csv file.
                e.g. '/workspace/github/ImmunSocialNetwork/data/Summary_Table_for_Liver.csv'.
            target_tissues (list, optional): Target tissues to be analyzed.
                e.g. ['Liver','Bone marrow', 'Blood', 'Lymph node','lymph node','Lymph', 'Spleen'].
        """

        summary = pd.read_csv(summary_path)
        all_names = summary['Registered Name'].tolist()

        # comprehensive cell types
        marker_res = []
        for name in all_names:
            tmp_res = self.total_ref[self.total_ref['cell_name']==name]
            tmp_res = tmp_res[tmp_res['tissue_class'].isin(target_tissues)]
            tmp_res = tmp_res.dropna(subset=['Symbol'])
            if len(tmp_res)==0:
                print(name,' has no marker gene symbol')
            marker_res.append(sorted(tmp_res['Symbol'].tolist())) # Note: remove unique setting

        self.raw_dic = dict(zip(all_names,marker_res)) # 91 cell types
    
    def merge_update(self,merge_cell_list=[['Memory T cell','Memory T(Tm) cell'],['Naive T cell','Naive T(Th0) cell']]):
        """_summary_

        Args:
            merge_cell_list (list, optional): Synonyms list to be merged.
                e.g. [['Memory T cell','Memory T(Tm) cell'],['Naive T cell','Naive T(Th0) cell']]
        """
        # merge synonyms
        merge_targets = list(itertools.chain.from_iterable(merge_cell_list))
        other_merge = [] # not merging in this process
        for i,k in enumerate(self.raw_dic):
            if k in merge_targets:
                pass
            else:
                other_merge.append(k)
        merged_name = []
        merged_v = []
        for m in merge_cell_list:
            merged_name.append(m[0]) # rename
            tmp = []
            for t in m:
                tmp.extend(self.raw_dic.get(t))
            merged_v.append(tmp)
        merged_dic = dict(zip(merged_name,merged_v))

        # update
        update_k = []
        update_v = []
        for i,k in enumerate(self.raw_dic):
            if k in other_merge:
                update_k.append(k)
                update_v.append(self.raw_dic.get(k))
            elif k in merged_name:
                update_k.append(k)
                update_v.append(merged_dic.get(k))
            else:
                pass
        self.update_dic = dict(zip(update_k,update_v)) # 83 cell types

    def prep_fine_dict(self,remove_cells=['T cell', 'B cell']):
        """Create a subdivided cell dictionary. You may need to remove hypernym or other redundant names.

        Args:
            remove_cells (list): Cell list to be removed from the update_dic.
        """
        update_dic = self.update_dic
        fine_dic = {k:v for k,v in update_dic.items() if k not in remove_cells}
        fine_dic = {k:v for k,v in fine_dic.items() if len(v) > 0}
        self.fine_dic = fine_dic
    
    def upper_cut(self,threshold=50):
        """If many marker genes are registered, cut at threshold.

        Args:
            threshold (int): Defaults to 50.
        """
        k_list = []
        v_list = []
        for i,k in enumerate(self.fine_dic):
            s = len(set(self.fine_dic.get(k))) # Update to be unique
            k_list.append(k)
            if s > threshold:
                print(s)
                print(k,self.fine_dic.get(k))
                c_item = dict(collections.Counter(self.fine_dic.get(k))).items()
                c_item = sorted(c_item,key=lambda x : x[1],reverse=True)
                tmp = []
                for t in c_item:
                    if t[1]==1:
                        break
                    else:
                        tmp.append(t[0])
                v_list.append(tmp)
            else:
                v_list.append(self.fine_dic.get(k))
        self.final_dic = dict(zip(k_list,v_list))
            