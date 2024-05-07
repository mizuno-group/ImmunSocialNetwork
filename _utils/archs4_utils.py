# -*- coding: utf-8 -*-
"""
Created on 2023-11-05 (Sun) 14:12:30

ARCHS4 handler for TPM normalized data
- TPM transcript expression files for mouse and human in HDF5 format. All transcript TPM are on ensembl_id level.

ref: https://github.com/MaayanLab/archs4py (for gene counts and transcript counts data)

@author: I.Azuma
"""
# %%
import numpy as np
import pandas as pd

import h5py as h5
#import s3fs
import tqdm
import re

import multiprocessing
import random

SC_THRESHOLD = 0.1 # 240225 changed 0.01 --> 0.3

# %%
def meta_local(file, search_term, meta_fields=["geo_accession", "series_id", "characteristics_ch1", "extract_protocol_ch1", "source_name_ch1", "title"],remove_sc=True, silent=False):
    f = h5.File(file, "r")
    idx = []
    for field in meta_fields:
        if field in f["meta"]["samples"].keys():
            meta = [x.decode("UTF-8") for x in list(np.array(f["meta"]["samples"][field]))]
            idx.extend([i for i, item in enumerate(meta) if re.search(search_term, re.sub(r"_|-|'|/| |\.", "", item.upper()))])

    library_source =np.array([x.decode("UTF-8") for x in np.array(f["meta/samples/library_source"])])
    if remove_sc:
        try:
            """
            Gene level and Transcript level
            """
            target_idx = np.where(np.array(f["meta/samples/singlecellprobability"]) < SC_THRESHOLD)[0]
            print("Gene selection from singlecellprobability")
        except:
            """TPM level"""
            target_idx = np.where((library_source == 'transcriptomic'))[0]
            print("Gene selection from library_source")
    else:
        target_idx = np.where((library_source == 'transcriptomic') | (library_source == 'transcriptomic single cell'))[0]
    idx = sorted(list(set(idx).intersection(set(target_idx))))
    counts = index(file, idx, silent=silent)
    return counts

def meta_info(file, search_term, meta_fields=["geo_accession", "series_id", "characteristics_ch1", "extract_protocol_ch1", "source_name_ch1", "title"],remove_sc=True, silent=False):
    """
    Search for samples in a file based on a search term in specified metadata fields.

    Args:
        file (str): The file path or object containing the data.
        search_term (str): The term to search for. Case-insensitive.
        meta_fields (list, optional): The list of metadata fields to search within.
            Defaults to ["geo_accession", "series_id", "characteristics_ch1", "extract_protocol_ch1", "source_name_ch1", "title"].
        remove_sc (bool, optional): Whether to remove single-cell samples from the results.
            Defaults to False.
        silent (bool, optional): Print progress bar.

    Returns:
        pd.DataFrame: DataFrame containing the extracted metadata, with metadata fields as columns and samples as rows.
    """
    search_term = search_term.upper()
    with h5.File(file, "r") as f:
        meta = []
        idx = []
        mfields = []
        for field in tqdm.tqdm(meta_fields, disable=not silent):
            if field in f["meta"]["samples"].keys():
                try:
                    meta.append([x.decode("UTF-8").upper() for x in list(np.array(f["meta"]["samples"][field]))])
                    mfields.append(field)
                except Exception:
                    x=0
        meta = pd.DataFrame(meta, index=mfields ,columns=[x.decode("UTF-8").upper() for x in list(np.array(f["meta"]["samples"]["geo_accession"]))])
        for i in tqdm.tqdm(range(meta.shape[0]), disable=silent):
            idx.extend([i for i, item in enumerate(meta.iloc[i,:]) if re.search(search_term, item.upper())])

        library_source =np.array([x.decode("UTF-8") for x in np.array(f["meta/samples/library_source"])])
        if remove_sc:
            try:
                """
                Gene level and Transcript level
                """
                target_idx = np.where(np.array(f["meta/samples/singlecellprobability"]) < SC_THRESHOLD)[0]
                print("Gene selection from singlecellprobability")
            except:
                """TPM level"""
                target_idx = np.where((library_source == 'transcriptomic'))[0]
                print("Gene selection from library_source")
        else:
            target_idx = np.where((library_source == 'transcriptomic') | (library_source == 'transcriptomic single cell'))[0]
        idx = sorted(list(set(idx).intersection(set(target_idx))))
    return meta.iloc[:,idx].T

def characteristics_ch1_selection(file, search_term, meta_fields=["geo_accession", "series_id", "characteristics_ch1", "extract_protocol_ch1", "source_name_ch1", "title"],remove_sc=True, silent=False):
    search_term = search_term.upper()
    f = h5.File(file, "r")
    meta = []
    mfields = []
    for field in tqdm.tqdm(meta_fields, disable=not silent):
        if field in f["meta"]["samples"].keys():
            try:
                meta.append([x.decode("UTF-8").upper() for x in list(np.array(f["meta"]["samples"][field]))])
                mfields.append(field)
            except Exception:
                try:
                    meta.append(list(np.array(f["meta"]["samples"][field])))
                    mfields.append(field)
                except:
                    print(field,": something is wrong.")
                    x=0
    meta = pd.DataFrame(meta, index=mfields ,columns=[x.decode("UTF-8").upper() for x in list(np.array(f["meta"]["samples"]["geo_accession"]))])

    library_source =np.array([x.decode("UTF-8") for x in np.array(f["meta/samples/library_source"])])
    if remove_sc:
        try:
            """
            Gene level and Transcript level
            """
            target_idx = np.where(np.array(f["meta/samples/singlecellprobability"]) < SC_THRESHOLD)[0]
            print("Gene selection from singlecellprobability")
        except:
            """TPM level"""
            target_idx = np.where((library_source == 'transcriptomic'))[0]
            print("Gene selection from library_source")
    else:
        target_idx = np.where((library_source == 'transcriptomic') | (library_source == 'transcriptomic single cell'))[0]
    meta = meta.iloc[:,target_idx].T

    # search term selection
    target_meta = meta[meta["characteristics_ch1"].str.contains(search_term)]

    return target_meta

def add_treatment(target_meta):
    treatment_info = target_meta[target_meta['characteristics_ch1'].str.contains('TREATMENT')]

    treatment_list = []
    target_keys = ['TREATMENT MOLECULE', 'PRE-TREATMENT','TREATMENT/CONDITION', 'TREATMENT_SHORT_DESCRIPT', 'CHEMICAL TREATMENT', 'TREATMENT_2','TREATMENT','AAV TREATMENT', 'TREATMENT REPONSE','TREATMENT_SHORT_NAME', 'DIET AND TREATMENT','TREATMENT_(DIET+TREATMENT)', 'PATERNAL TREATMENT', 'TREATMENT DIET', 'TREATMENT_FULL_NAME', 'TREATMENT 2', 'DRUG TREATMENT','TREATMENT(1)', 'TREATMENT_1','TREATMENT_COMPOUND', 'TREATMENT GROUP', 'TREATMENT 1']

    key_list = set()
    final_info = []
    final_idx = []
    for idx, txt in enumerate(treatment_info['characteristics_ch1'].tolist()):
        tmp_list = txt.split(',')
        info_dict = each_sample_dict(tmp_list)
        key_list = key_list | set(info_dict.keys())
        tmp_treatment = []
        for i,k in enumerate(info_dict):
            if k in target_keys:
                treatment = info_dict.get(k)
                tmp_treatment.append(treatment)
            else:
                pass
        if len(tmp_treatment)>0:
            tmp_treatment = sorted(tmp_treatment,key=len,reverse=True)
            final_info.append(tmp_treatment[0])
            final_idx.append(idx)
        else:
            pass
    # Add treatment info
    target_info = treatment_info.iloc[final_idx]
    target_info['treatment']=final_info
    
    return target_info

    
def update(tmp_list):
    """ Concat the unexpected separation.
    Args:
        tmp_list (list): ['TISSUE: LIVER', 'CELL TYPE: TRNK', 'GENOTYPE: WT', 'TREATMENT: 50 NG/ML PMA', ' 1 UG/ML IONOMYCIN']

    Returns:
        list: ['TISSUE: LIVER','CELL TYPE: TRNK','GENOTYPE: WT','TREATMENT: 50 NG/ML PMA 1 UG/ML IONOMYCIN']
    """
    updated_list = []
    for t in tmp_list:
        if ': ' in t:
            updated_list.append(t)
        else:
            updated_list[-1] = updated_list[-1]+t
    return updated_list

def each_sample_dict(tmp_list):
    """
    Args:
        tmp_list (list): comma separated list
            ['AGE: 28 WKS AT THE TIME OF RNA ISOLATION',
            'BACKGROUND STRAIN: C57BL/129SV/FVB',
            'GENOTYPE: 9V/NULL',
            'TREATMENT: CZ (IMIGLUCERASE)',
            'TISSUE: LIVER']
    """
    tmp_list = update(tmp_list)
    k_list = []
    v_list = []
    for t in tmp_list:
        k = t.split(': ')[0]
        v = t.split(k+': ')[1] # 'treatment: AAA: BBB' --> AAA: BBB
        k_list.append(k)
        v_list.append(v)
    info_dict = dict(zip(k_list,v_list))

    return info_dict

# %%
def samples_local(file,sample_ids,silent=False,row_type='transcript'):
    sample_ids = set(sample_ids)
    f = h5.File(file, "r")
    samples = [x.decode("UTF-8") for x in np.array(f["meta/samples/geo_accession"])]
    f.close()
    idx = [i for i,x in enumerate(samples) if x in sample_ids]
    if len(idx) > 0:
        return index(file, idx, silent=silent, row_type=row_type)

def index(file,sample_idx,gene_idx=[],silent=False,row_type='transcript'):
    """
    Retrieve gene expression data from a specified file for the given sample and gene indices.

    Args:
        file (str): The file path or object containing the data.
        sample_idx (list): A list of sample indices to retrieve expression data for.
        gene_idx (list, optional): A list of gene indices to retrieve expression data for. Defaults to an empty list (return all).
        silent (bool, optional): Whether to disable progress bar. Defaults to False.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the gene expression data.
    """
    sample_idx = sorted(sample_idx)
    gene_idx = sorted(gene_idx)
    if row_type == 'transcript':
        row_encoding = "meta/transcripts/ensembl_id"
    elif row_type == 'gene':
        row_encoding = "meta/genes/symbol"
    else:
        raise ValueError("!! Maybe the row_type is inappropriate !!")
        pass

    f = h5.File(file, "r")
    try:
        genes = np.array([x.decode("UTF-8") for x in np.array(f[row_encoding])])
    except:
        raise ValueError("!! Maybe the row_type is inappropriate !!")
    if len(gene_idx) == 0:
        gene_idx = list(range(len(genes)))
    if len(sample_idx) == 0:
        return pd.DataFrame(index=genes[gene_idx])
    gsm_ids = np.array([x.decode("UTF-8") for x in np.array(f["meta/samples/geo_accession"])])[sample_idx]
    f.close()

    exp = []
    PROCESSES = 16
    with multiprocessing.Pool(PROCESSES) as pool:
        results = [pool.apply_async(get_sample, (file, i, gene_idx)) for i in sample_idx]
        for r in tqdm.tqdm(results, disable=silent):
            res = r.get()
            exp.append(res)
    exp = np.array(exp).T
    exp = pd.DataFrame(exp, index=genes[gene_idx], columns=gsm_ids, dtype=np.float32)
    return exp

def get_sample(file, i, gene_idx):
    try:
        f = h5.File(file, "r")
        temp = np.array(f["data/expression"][:,i], dtype=np.float32)[gene_idx]
        f.close()
    except Exception:
        dd = np.array([0]*len(gene_idx))
        return dd
    return temp

# %%
