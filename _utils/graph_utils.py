# -*- coding: utf-8 -*-
"""
Created on 2024-06-09 (Sun) 11:37:32

Graph handler

@author: I.Azuma
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch.nn import Linear, Parameter

from torch_geometric.utils import from_networkx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse

class DirectedMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, weight_scale=1.0, add_norm=True, add_bias=False):
        super().__init__(aggr='add')
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

        self.weight_scale = weight_scale
        self.add_norm = add_norm
        self.add_bias = add_bias
    
    def reset_parameters(self):
        self.bias.data.zero_()
                
    def forward(self, x, adj_matrix):
        # 1. Extract edge_index and weight adj
        edge_index, edge_weight = dense_to_sparse(adj_matrix)
        max_v = edge_weight.abs().max()
        edge_weight = self.weight_scale*edge_weight/max_v

        # 2. Add self-loops to the adjacency matrix
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=x.size(0))

        # 3. Normalize node features.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # 4. Start propagating messages
        out = self.propagate(edge_index, x=x, norm=norm)
        
        if self.add_bias:
            out = out + self.bias
            out = out.detach().numpy()
        else:
            pass

        return out

    def message(self, x_j, norm):
        if self.add_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j
    
    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out


def create_dag(sm,node_names:list,save_dir=None,weight_threshold=0.0,edge_limit=1000000,do_plot=True,do_abs=False):
    if do_plot:
        # Visualize
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw_circular(sm,
                        with_labels=True,
                        font_size=10,
                        node_size=1000,
                        arrowsize=10,
                        alpha=0.5,
                        ax=ax)
        plt.plot()
        plt.show()

        # Edge weight distribution
        fig,ax = plt.subplots()
        weight_list = [d['mean_effect'] for (u,v,d) in sm.edges(data=True)]
        plt.plot(sorted(weight_list))
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
        ax.set_axisbelow(True)
        ax.grid(color="#ababab",linewidth=0.5)
        plt.title("Edge Weight Distribution")
        plt.show()

    # DAG construction
    sm_l = sm.get_largest_subgraph()
    dag = nx.DiGraph()

    new_idx = []
    pn_labels = []
    source_labels = []
    target_labels = []
    sorted_edge_list = sorted(sm_l.edges(data=True),key=lambda x : abs(x[-1]['mean_effect']),reverse=True)  # sort by weight
    for (u,v,d) in sorted_edge_list:
        if abs(d['mean_effect']) < weight_threshold:
            continue

        if d['mean_effect'] > 0:
            pn_labels.append('positive')
            pn_color = '#E57373'  # red
        else:
            pn_labels.append('negative')
            pn_color = '#64B5F6'  # blue
        new_u = node_names.index(u)
        new_v = node_names.index(v)
        if do_abs:
            dag.add_edge(new_u, new_v, weight=abs(d['mean_effect'].astype(float)),color=pn_color)  # d['weight']
        else:
            dag.add_edge(new_u, new_v, weight=d['mean_effect'].astype(float),color=pn_color)
        new_idx.append('{} (interacts with) {}'.format(new_u,new_v))
        source_labels.append(new_u)
        target_labels.append(new_v)

        if len(new_idx) == int(edge_limit):
            print("Reached the upper size.")
            break

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

    # display
    id_list = [i for i in range(len(node_names))]
    name2id = dict(zip(node_names,id_list))
    id2name = dict(zip(id_list,node_names))
    graph_edge_weights = nx.get_edge_attributes(dag, 'weight')
    graph_edge_widths = [abs(weight)*5 for weight in graph_edge_weights.values()]
    labels = {k:id2name.get(k) for k in dag.nodes}
    edge_colors = [dag.edges[edge]['color'] for edge in dag.edges()]

    nx.draw(dag, edge_color=edge_colors, labels=labels,width=graph_edge_widths, arrows=True, with_labels=True)
    plt.show()

    return dag

def get_first_neighbors(graph, node):
    neighbors = set()
    # collect target
    neighbors.update(graph.successors(node))
    # collect source
    neighbors.update(graph.predecessors(node))
    return neighbors
