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
import s3fs
import tqdm
import re

import multiprocessing
import random

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
        target_idx = np.where((library_source == 'transcriptomic'))[0]
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
            target_idx = np.where((library_source == 'transcriptomic'))[0]
        else:
            target_idx = np.where((library_source == 'transcriptomic') | (library_source == 'transcriptomic single cell'))[0]
        idx = sorted(list(set(idx).intersection(set(target_idx))))
    return meta.iloc[:,idx].T

def samples_local(file, sample_ids, silent=False):
    sample_ids = set(sample_ids)
    f = h5.File(file, "r")
    samples = [x.decode("UTF-8") for x in np.array(f["meta/samples/geo_accession"])]
    f.close()
    idx = [i for i,x in enumerate(samples) if x in sample_ids]
    if len(idx) > 0:
        return index(file, idx, silent=silent)

def index(file, sample_idx, gene_idx = [],silent=False):
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
    row_encoding = "meta/transcripts/ensembl_id"

    f = h5.File(file, "r")
    genes = np.array([x.decode("UTF-8") for x in np.array(f[row_encoding])])
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
