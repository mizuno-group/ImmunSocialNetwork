# -*- coding: utf-8 -*-
"""
Created on 2024-04-25 (Thu) 21:11:03

Helper functions for CellPhoneDB.

@author: I.Azuma
"""
# %%
from pybiomart import Server # pip install git+https://github.com/vitkl/orthologsBioMART.git

# %%
# Convert to HGNC (ortholog)

# find correct dataset name for your species

server = Server(host='http://www.ensembl.org')
server.marts['ENSEMBL_MART_ENSEMBL'].list_datasets()

from pyorthomap import findOrthologsMmHs, findOrthologsHsMm

mgi_symbols = pd.read_csv('/workspace/datasource/Biomart/mouse_transcriptID2MGI.csv')['mgi_symbol'].tolist()
mouse_genes = sorted(set(log_df.columns) & set(mgi_symbols)) # 19286
chunk = 100
merge_res = pd.DataFrame()
for i in range(len(mouse_genes)//chunk + 1):
    print(i)
    res = findOrthologsMmHs(from_filters='mgi_symbol',from_values=mouse_genes[chunk*i:chunk*(i+1)]).map()
    merge_res = pd.concat([merge_res,res])

merge_res.to_csv('/workspace/datasource/Tabula_Muris/processed/Liver/for_cellphonedb/mgi_hgnc_ortholog.csv')