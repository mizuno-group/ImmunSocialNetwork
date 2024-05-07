# -*- coding: utf-8 -*-
"""
Created on 2024-04-24 (Wed) 22:33:42

@author: I.Azuma
"""
# %%
import copy
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from causalnex.structure.dynotears import from_pandas_dynamic

# %%
class DYNOTEARS_Analyzer():
    def __init__(self):
        self.time_series = None
        self.sm_l = None

    def set_data(self,time_series:list):
        self.time_series = time_series
        print(f"Sample: {len(self.time_series)}, Time Points: {self.time_series[0].shape[0]}, Variable Size: {self.time_series[0].shape[1]}")
    
    def create_skelton(self,do_plot=True,tabu_child_nodes=[],p=2,lambda_w=0.1,lambda_a=0.1):
        # DYNOTEARS
        self.sm = from_pandas_dynamic(time_series=self.time_series,p=p,lambda_w=lambda_w,lambda_a=lambda_a,tabu_child_nodes=tabu_child_nodes)

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
    
    def save_dag(self,save_dir='/Path/to/the/directory',weight_threshold=0.0,edge_limit=1000000):
        if self.sm_l is None:
            self.sm_l = self.sm.get_largest_subgraph()

        node_names = list(self.sm_l.nodes())
        dag = nx.DiGraph()

        new_idx = []
        pn_labels = []
        source_labels = []
        target_labels = []
        for (u,v,d) in self.sm_l.edges(data=True):
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
        node_names = [t.replace('lag','') for t in node_names]

        # Node annotation
        node_names_df = pd.DataFrame({'ID':[i for i in range(len(node_names))],'name':node_names})
        node_names_df['lag'] = [t.split('_')[-1] for t in node_names_df['name'].tolist()]
        node_names_df.to_csv(save_dir + '/node_name_df.csv')

        # Edge type annotation
        edge_df = pd.DataFrame({'Edge_Key':new_idx,'PN':pn_labels,'Source':source_labels,'Target':target_labels})
        edge_df.to_csv(save_dir+'/edge_type_df.csv')

        # Save networkx
        nx.write_gml(dag, save_dir+'/causualnex_dag.gml')

        print("Node Size: {}".format(len(dag.nodes())))
        print("Edge Size: {}".format(len(dag.edges())))
