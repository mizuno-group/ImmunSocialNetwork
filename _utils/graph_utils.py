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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter

try:
    from torch_sparse import spmm  # (edge_index, edge_weight, m, n, mat)
except Exception as e:
    spmm = None

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, dense_to_sparse

import torch
from torch_geometric.utils import dense_to_sparse, add_self_loops, degree


class RandomWalkWithRestart(nn.Module):

    def __init__(
        self,
        alpha: float = 0.2,          # restart prob
        n_steps: int = 20,           # restart steps
        add_self_loops: bool = True, 
        eps: float = 1e-12,          
        clamp_negative: bool = True, 
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.n_steps = int(n_steps)
        self.add_self_loops = bool(add_self_loops)
        self.eps = float(eps)
        self.clamp_negative = bool(clamp_negative)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        """
        x:   [N, F] node features
        adj: [N, N] adjacency (dense)
        return: [N, F] updated features
        """
        assert adj.dim() == 2 and adj.size(0) == adj.size(1), "adj must be [N,N]"
        assert x.dim() == 2 and x.size(0) == adj.size(0), "x must be [N,F] with same N as adj"

        A = adj.to(dtype=x.dtype, device=x.device)

        # clamp
        if self.clamp_negative:
            A = A.clamp_min(0)

        if self.add_self_loops:
            A = A.clone()
            idx = torch.arange(A.size(0), device=A.device)
            A[idx, idx] = A[idx, idx] + 1.0

        # row-stochastic: P = D^{-1} A
        row_sum = A.sum(dim=1, keepdim=True).clamp_min(self.eps)
        P = A / row_sum

        H = x
        a = self.alpha
        for _ in range(self.n_steps):
            H = (1.0 - a) * (P @ H) + a * x

        return H

def build_norm_adj_from_dense(adj_matrix: torch.Tensor, weight_scale: float = 1.0, add_self_loop: bool = True):
    """
    adj_matrix: [N, N] dense (directed/weighted OK)
    return: edge_index [2, E], norm [E]  (norm = D^{-1/2} A D^{-1/2})
    """
    edge_index, edge_weight = dense_to_sparse(adj_matrix)

    # scale (max=0対策込み)
    max_v = edge_weight.abs().max()
    if max_v > 0:
        edge_weight = weight_scale * edge_weight / max_v
    else:
        edge_weight = edge_weight  # all zero case

    if add_self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=adj_matrix.size(0)
        )

    row, col = edge_index
    deg = degree(col, adj_matrix.size(0), dtype=edge_weight.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, norm


@torch.no_grad()
def message_passing_k_steps(
    x: torch.Tensor,
    adj_matrix: torch.Tensor,
    k: int = 3,
    weight_scale: float = 1.0,
    add_self_loop: bool = True,
    residual: bool = False,
    renorm_each_step: bool = False,
):
    """
    x: [N, F]
    adj_matrix: [N, N] dense
    k: message passing steps
    residual: x <- x + MP(x) if True
    renorm_each_step: recompute normalization at each step
    """
    assert k >= 1
    if not renorm_each_step:
        edge_index, norm = build_norm_adj_from_dense(adj_matrix, weight_scale, add_self_loop)

    h = x
    for _ in range(k):
        if renorm_each_step:
            edge_index, norm = build_norm_adj_from_dense(adj_matrix, weight_scale, add_self_loop)

        # message passing
        h_new = spmm(edge_index, norm, x.size(0), x.size(0), h)

        h = h + h_new if residual else h_new

    return h

class DirectedMessagePassing(MessagePassing):
    def __init__(self, out_channels, weight_scale=1.0, add_norm=True, add_bias=False):
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
        """
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
        """

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
    
    # display
    id_list = [i for i in range(len(node_names))]
    name2id = dict(zip(node_names,id_list))
    id2name = dict(zip(id_list,node_names))
    graph_edge_weights = nx.get_edge_attributes(dag, 'weight')
    graph_edge_widths = [abs(weight)*5 for weight in graph_edge_weights.values()]
    labels = {k:id2name.get(k) for k in dag.nodes}
    edge_colors = [dag.edges[edge]['color'] for edge in dag.edges()]

    if do_plot:
        print("Node Size: {}".format(len(dag.nodes())))
        print("Edge Size: {}".format(len(dag.edges())))

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
