# -*- coding: utf-8 -*-
"""
Created on 2024-02-15 (Thu) 16:29:45

@author: I.Azuma
"""
# %%
import copy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from causalnex.structure.notears import from_pandas
from causalnex.structure.pytorch.notears import from_pandas as from_pandas_pytorch


# %%
class NOTEARS_Analyzer():
    def __init__(self):
        self.deconv_res = None
        self.sm_l = None
        self.dag = None
        self.input_data = None

    def set_data(self,deconv_res:pd.DataFrame):
        """ Set deconvolution result as input dataframe

        Args:
            deconv_res (pd.DataFrame): Samples are in rows and cell types are in columns.
            	        Hepatocyte	Hepatoblast	Cholangiocyte
            GSM1400569	0.026937	0.007902	0.012417
            GSM1400571	0.026574	0.011447	0.005951
            GSM1400573	0.026255	0.014530	0.008820
        """
        self.deconv_res = deconv_res
        print(self.deconv_res.shape)
    
    def binning(self,bins=10):
        df_new = self.deconv_res.copy()
        for col in self.deconv_res.columns.tolist():
            df_new[col] = pd.cut(df_new[col], bins, labels=False) # Note labels=False
        self.input_data = df_new
    
    def create_skelton(self,do_plot=True,tabu_child_nodes=['Hepatocyte'],pytorch=False):
        if self.input_data is None:
            self.input_data = self.deconv_res
        # DAGs with NO TEARS (Dense)
        if pytorch:
            self.sm = from_pandas_pytorch(self.input_data,tabu_child_nodes=tabu_child_nodes)
        else:
            self.sm = from_pandas(self.input_data,tabu_child_nodes=tabu_child_nodes)

        if do_plot:
            # Visualize
            fig, ax = plt.subplots(figsize=(16, 16))
            nx.draw_circular(self.sm,
                            with_labels=True,
                            font_size=10,
                            node_size=3000,
                            arrowsize=20,
                            alpha=0.5,
                            ax=ax)
            plt.plot()
            plt.show()
        
        # Edge weight distribution
        fig,ax = plt.subplots()
        weight_list = [d['weight'] for (u,v,d) in self.sm.edges(data=True)]
        plt.plot(sorted(weight_list))
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.title("Edge Weight Distribution")
        plt.show()

        # construct raw graph
        node_names = self.input_data.columns.tolist()
        raw_dag = nx.DiGraph()
        new_idx = []
        pn_labels = []
        source_labels = []
        target_labels = []
        for (u,v,d) in self.sm.edges(data=True):
            new_u = node_names.index(u)
            new_v = node_names.index(v)
            raw_dag.add_edge(new_u, new_v, weight=abs(d['weight']))
            new_idx.append('{} (interacts with) {}'.format(new_u,new_v))
            source_labels.append(new_u)
            target_labels.append(new_v)
        
        self.raw_dag = raw_dag

    
    def create_dags(self,weight_threshold=0.3,do_plot=True):
        # Trimming
        self.sm.remove_edges_below_threshold(weight_threshold)
        edge_width = [d['weight']*1 for (u,v,d) in self.sm.edges(data=True)]

        if do_plot:
            fig, ax = plt.subplots(figsize=(16, 16))
            nx.draw_circular(self.sm,
                            with_labels=True,
                            font_size=10,
                            node_size=3000,
                            arrowsize=20,
                            alpha=0.5,
                            width=edge_width,
                            ax=ax)
            plt.plot()
            plt.show()
        
        # Extract largest subgraph
        self.sm_l = self.sm.get_largest_subgraph()
        fig, ax = plt.subplots(figsize=(16, 16))
        nx.draw_circular(self.sm_l,
                        with_labels=True,
                        font_size=10,
                        node_size=3000,
                        arrowsize=20,
                        alpha=0.5,
                        width=edge_width,
                        ax=ax)
        plt.show()
    
    def save_dag(self,save_dir=None,weight_threshold=0.0,edge_limit=1000000):
        if self.sm_l is None:
            self.sm_l = self.sm.get_largest_subgraph()

        node_names = self.input_data.columns.tolist()
        dag = nx.DiGraph()

        new_idx = []
        pn_labels = []
        source_labels = []
        target_labels = []
        sorted_edge_list = sorted(self.sm_l.edges(data=True),key=lambda x : x[-1]['weight'],reverse=True)  # sort by weight
        for (u,v,d) in sorted_edge_list:
            if abs(d['weight']) < weight_threshold:
                continue

            if d['weight'] > 0:
                pn_labels.append('positive')
            else:
                pn_labels.append('negative')
            new_u = node_names.index(u)
            new_v = node_names.index(v)
            dag.add_edge(new_u, new_v, weight=abs(d['weight']))
            new_idx.append('{} (interacts with) {}'.format(new_u,new_v))
            source_labels.append(new_u)
            target_labels.append(new_v)

            if len(new_idx) == edge_limit:
                print("Reached the upper size.")
                break

        nx.draw(dag, arrows=True, with_labels=True)

        if save_dir is not None:
            # save_dir='/Path/to/the/directory'
            # Node annotation
            node_names_df = pd.DataFrame({'ID':[i for i in range(len(node_names))],'name':node_names})
            node_names_df.to_csv(save_dir + '/node_name_df.csv')

            # Edge type annotation
            edge_df = pd.DataFrame({'Edge_Key':new_idx,'PN':pn_labels,'Source':source_labels,'Target':target_labels})
            edge_df.to_csv(save_dir+'/edge_type_df.csv')

            # Save networkx
            nx.write_gml(dag, save_dir+'/causualnex_dag.gml')

            print("Node Size: {}".format(len(dag.nodes())))
            print("Edge Size: {}".format(len(dag.edges())))

        self.dag = dag
