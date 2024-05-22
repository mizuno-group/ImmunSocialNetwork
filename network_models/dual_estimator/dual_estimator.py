# Copyright 2019-2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tools to learn a ``StructureModel`` which describes the conditional dependencies between variables in a dataset.
"""

import logging
from copy import deepcopy
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.utils import check_array

#from network_models.dual_estimator.core import NotearsMLP
from network_models.dual_estimator.core_dev import Dual_NotearsMLP
from network_models.dual_estimator.dist_type import DistTypeContinuous, dist_type_aliases
from network_models.dual_estimator.structuremodel import StructureModel

__all__ = ["from_numpy", "from_pandas"]


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def from_numpy(
    Xd: np.ndarray,
    Xh: np.ndarray,
    binary_mask: bool = True,
    dist_type_schema: Dict[int, str] = None,
    ls_gamma: float = 0.5,
    lasso_beta: float = 0.0,
    ridge_beta: float = 0.0,
    use_bias: bool = False,
    hidden_layer_units: Iterable[int] = None,
    w_threshold: float = None,
    max_iter: int = 100,
    tabu_edges_h: List[Tuple[int, int]] = None,
    tabu_parent_nodes_h: List[int] = None,
    tabu_child_nodes_h: List[int] = None,
    tabu_edges_d: List[Tuple[int, int]] = None,
    tabu_parent_nodes_d: List[int] = None,
    tabu_child_nodes_d: List[int] = None,
    use_gpu: bool = True,
    **kwargs,
) -> StructureModel:
   
    # n examples, d properties
    if not Xd.size:
        raise ValueError("Input data Xd is empty, cannot learn any structure")
    if not Xh.size:
        raise ValueError("Input data Xh is empty, cannot learn any structure")
    logging.info("Learning structure using 'NOTEARS' optimisation.")

    # Check array for NaN or inf values
    check_array(Xd)
    check_array(Xh)

    if dist_type_schema is not None:

        # make sure that there is one provided key per column
        if set(range(Xd.shape[1])).symmetric_difference(set(dist_type_schema.keys())):
            raise ValueError(
                f"Difference indices and expected indices. Got {dist_type_schema} schema"
            )

    # if dist_type_schema is None, assume all columns are continuous, else init the alias mapped object
    dist_types_h = (
        [DistTypeContinuous(idx=idx) for idx in np.arange(Xh.shape[1])]
        if dist_type_schema is None
        else [
            dist_type_aliases[alias](idx=idx) for idx, alias in dist_type_schema.items()
        ]
    )
    dist_types_d = (
        [DistTypeContinuous(idx=idx) for idx in np.arange(Xd.shape[1])]
        if dist_type_schema is None
        else [
            dist_type_aliases[alias](idx=idx) for idx, alias in dist_type_schema.items()
        ]
    )

    # shape of X before preprocessing
    _, d_orig = Xd.shape
    # perform dist type pre-processing (i.e. column expansion)
    for dist_type in dist_types_h:
        Xh = dist_type.preprocess_X(Xh)
        tabu_edges_h = dist_type.preprocess_tabu_edges(tabu_edges_h)
        tabu_parent_nodes_h = dist_type.preprocess_tabu_nodes(tabu_parent_nodes_h)
        tabu_child_nodes_h = dist_type.preprocess_tabu_nodes(tabu_child_nodes_h)

    for dist_type in dist_types_d:
        Xd = dist_type.preprocess_X(Xd)
        tabu_edges = dist_type.preprocess_tabu_edges(tabu_edges_d)
        tabu_parent_nodes = dist_type.preprocess_tabu_nodes(tabu_parent_nodes_d)
        tabu_child_nodes = dist_type.preprocess_tabu_nodes(tabu_child_nodes_d)
    # shape of X after preprocessing
    _, d = Xd.shape

    # if None or empty, convert into a list with single item
    if hidden_layer_units is None:
        hidden_layer_units = [0]
    elif isinstance(hidden_layer_units, list) and not hidden_layer_units:
        hidden_layer_units = [0]

    # if no hidden layer units, still take 1 iteration step with bounds
    hidden_layer_bnds = hidden_layer_units[0] if hidden_layer_units[0] else 1

    # Flip i and j because Pytorch flattens the vector in another direction
    bnds_h = [
        (0, 0)
        if i == j
        else (0, 0)
        if tabu_edges_h is not None and (i, j) in tabu_edges_h
        else (0, 0)
        if tabu_parent_nodes_h is not None and i in tabu_parent_nodes_h
        else (0, 0)
        if tabu_child_nodes_h is not None and j in tabu_child_nodes_h
        else (None, None)
        for j in range(d)
        for _ in range(hidden_layer_bnds)
        for i in range(d)
    ]

    bnds_d = [
        (0, 0)
        if i == j
        else (0, 0)
        if tabu_edges_d is not None and (i, j) in tabu_edges_d
        else (0, 0)
        if tabu_parent_nodes_d is not None and i in tabu_parent_nodes_d
        else (0, 0)
        if tabu_child_nodes_d is not None and j in tabu_child_nodes_d
        else (None, None)
        for j in range(d)
        for _ in range(hidden_layer_bnds)
        for i in range(d)
    ]

    model = Dual_NotearsMLP(
        n_features=d,
        dist_types_h=dist_types_h,
        dist_types_d=dist_types_d,
        hidden_layer_units=hidden_layer_units,
        ls_gamma=ls_gamma,
        lasso_beta=lasso_beta,
        ridge_beta=ridge_beta,
        bounds_h=bnds_h,
        bounds_d=bnds_d,
        use_bias=use_bias,
        use_gpu=use_gpu,
        **kwargs,
    )

    model.fit(xd=Xd, xh=Xh, max_iter=max_iter, w_binary_mask=binary_mask)
    sm_d = StructureModel(model.adj_d)
    sm_h = StructureModel(model.adj_h)

    if w_threshold:
        sm_d.remove_edges_below_threshold(w_threshold)
        sm_h.remove_edges_below_threshold(w_threshold)

    # extract the mean effect and add as edge attribute
    mean_effect_d = model.adj_d_mean_effect
    for u, v, edge_dict in sm_d.edges.data(True):
        sm_d.add_edge(
            u,
            v,
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=mean_effect_d[u, v],
        )
    
    mean_effect_h = model.adj_h_mean_effect
    for u, v, edge_dict in sm_h.edges.data(True):
        sm_h.add_edge(
            u,
            v,
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=mean_effect_h[u, v],
        )

    # set bias as node attribute
    bias = model.disease_branch.bias
    for node in sm_d.nodes():
        value = None
        if bias is not None:
            value = bias[node]
        sm_d.nodes[node]["bias"] = value
    for node in sm_h.nodes():
        value = None
        if bias is not None:
            value = bias[node]
        sm_h.nodes[node]["bias"] = value

    # attach each dist_type object to corresponding node(s)
    for dist_type in dist_types_d:
        sm_d = dist_type.add_to_node(sm_d)
    for dist_type in dist_types_h:
        sm_h = dist_type.add_to_node(sm_h)

    # preserve the structure_learner as a graph attribute
    sm_d.graph["structure_learner"] = model
    sm_h.graph["structure_learner"] = model

    # collapse the adj down and store as graph attr
    adj_d = deepcopy(model.adj_d)
    sm_d.graph["adjacency_d"] = adj_d
    adj_h = deepcopy(model.adj_h)
    sm_h.graph["adjacency_h"] = adj_h

    for dist_type in dist_types_d:
        adjd = dist_type.collapse_adj(adj_d)
    sm_d.graph["graph_collapsed"] = StructureModel(adjd[:d_orig, :d_orig])

    for dist_type in dist_types_h:
        adjh = dist_type.collapse_adj(adj_h)
    sm_h.graph["graph_collapsed"] = StructureModel(adjh[:d_orig, :d_orig])

    return sm_d, sm_h


# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
def from_pandas(
    Xd: pd.DataFrame,  # disease
    Xh: pd.DataFrame,  # healthy
    binary_mask: bool = True,
    dist_type_schema: Dict[Union[str, int], str] = None,
    ls_gamma: float = 0.5,
    lasso_beta: float = 0.0,
    ridge_beta: float = 0.0,
    use_bias: bool = False,
    hidden_layer_units: Iterable[int] = None,
    max_iter: int = 100,
    w_threshold: float = None,
    tabu_edges_h: List[Tuple[str, str]] = None,
    tabu_parent_nodes_h: List[str] = None,
    tabu_child_nodes_h: List[str] = None,
    tabu_edges_d: List[Tuple[str, str]] = None,
    tabu_parent_nodes_d: List[str] = None,
    tabu_child_nodes_d: List[str] = None,
    use_gpu: bool = True,
    **kwargs,
) -> StructureModel:

    assert Xd.shape[1] == Xh.shape[1], "Xd and Xh should be the same columns size."

    data_d = deepcopy(Xd)
    data_h = deepcopy(Xh)

    # if dist_type_schema is not None, convert dist_type_schema from cols to idx
    dist_type_schema = (
        dist_type_schema
        if dist_type_schema is None
        else {Xd.columns.get_loc(col): alias for col, alias in dist_type_schema.items()}
    )
    non_numeric_cols = data_d.select_dtypes(exclude="number").columns

    if len(non_numeric_cols) > 0:
        raise ValueError(
            "All columns must have numeric data."
            f"Consider mapping the following columns to int {non_numeric_cols}"
        )

    col_idx = {c: i for i, c in enumerate(data_d.columns)}
    idx_col = {i: c for c, i in col_idx.items()}

    # healthy tabu settings
    if tabu_edges_h:
        tabu_edges_h = [(col_idx[u], col_idx[v]) for u, v in tabu_edges_h]
    if tabu_parent_nodes_h:
        tabu_parent_nodes_h = [col_idx[n] for n in tabu_parent_nodes_h]
    if tabu_child_nodes_h:
        tabu_child_nodes_h = [col_idx[n] for n in tabu_child_nodes_h]
    
    # disease tabu settings
    if tabu_edges_d:
        tabu_edges_d = [(col_idx[u], col_idx[v]) for u, v in tabu_edges_d]
    if tabu_parent_nodes_d:
        tabu_parent_nodes_d = [col_idx[n] for n in tabu_parent_nodes_d]
    if tabu_child_nodes_d:
        tabu_child_nodes_d = [col_idx[n] for n in tabu_child_nodes_d]

    g_d, g_h = from_numpy(
                Xd=data_d.values,
                Xh=data_h.values,
                binary_mask = binary_mask,
                dist_type_schema=dist_type_schema,
                ls_gamma=ls_gamma,
                lasso_beta=lasso_beta,
                ridge_beta=ridge_beta,
                use_bias=use_bias,
                hidden_layer_units=hidden_layer_units,
                w_threshold=w_threshold,
                max_iter=max_iter,
                tabu_edges_h=tabu_edges_h,
                tabu_parent_nodes_h=tabu_parent_nodes_h,
                tabu_child_nodes_h=tabu_child_nodes_h,
                tabu_edges_d=tabu_edges_d,
                tabu_parent_nodes_d=tabu_parent_nodes_d,
                tabu_child_nodes_d=tabu_child_nodes_d,
                use_gpu=use_gpu,
                **kwargs,
    )

    disease_sm = postprocessing(g_d, idx_col)
    healthy_sm = postprocessing(g_h, idx_col)


    """
    # set comprehension to ensure only unique dist types are extraced
    # NOTE: this prevents double-renaming caused by the same dist type used on expanded columns
    unique_dist_types = {node[1]["dist_type"] for node in g.nodes(data=True)}
    # use the dist types to update the idx_col mapping
    idx_col_expanded = deepcopy(idx_col)
    for dist_type in unique_dist_types:
        idx_col_expanded = dist_type.update_idx_col(idx_col_expanded)

    sm = StructureModel()
    # add expanded set of nodes
    sm.add_nodes_from(list(idx_col_expanded.values()))

    # recover the edge weights from g
    for u, v, edge_dict in g.edges.data(True):
        sm.add_edge(
            idx_col_expanded[u],
            idx_col_expanded[v],
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=edge_dict["mean_effect"],
        )

    # retrieve all graphs attrs
    for key, val in g.graph.items():
        sm.graph[key] = val

    # recover the node biases from g
    for node in g.nodes(data=True):
        node_name = idx_col_expanded[node[0]]
        sm.nodes[node_name]["bias"] = node[1]["bias"]

    # recover and preseve the node dist_types
    for node_data in g.nodes(data=True):
        node_name = idx_col_expanded[node_data[0]]
        sm.nodes[node_name]["dist_type"] = node_data[1]["dist_type"]

    # recover the collapsed model from g
    sm_collapsed = StructureModel()
    sm_collapsed.add_nodes_from(list(idx_col.values()))
    for u, v, edge_dict in g.graph["graph_collapsed"].edges.data(True):
        sm_collapsed.add_edge(
            idx_col[u],
            idx_col[v],
            origin="learned",
            weight=edge_dict["weight"],
        )
    sm.graph["graph_collapsed"] = sm_collapsed
    """

    return disease_sm, healthy_sm

def postprocessing(g, idx_col):
        # set comprehension to ensure only unique dist types are extraced
    # NOTE: this prevents double-renaming caused by the same dist type used on expanded columns
    unique_dist_types = {node[1]["dist_type"] for node in g.nodes(data=True)}
    # use the dist types to update the idx_col mapping
    idx_col_expanded = deepcopy(idx_col)
    for dist_type in unique_dist_types:
        idx_col_expanded = dist_type.update_idx_col(idx_col_expanded)

    sm = StructureModel()
    # add expanded set of nodes
    sm.add_nodes_from(list(idx_col_expanded.values()))

    # recover the edge weights from g
    for u, v, edge_dict in g.edges.data(True):
        sm.add_edge(
            idx_col_expanded[u],
            idx_col_expanded[v],
            origin="learned",
            weight=edge_dict["weight"],
            mean_effect=edge_dict["mean_effect"],
        )

    # retrieve all graphs attrs
    for key, val in g.graph.items():
        sm.graph[key] = val

    # recover the node biases from g
    for node in g.nodes(data=True):
        node_name = idx_col_expanded[node[0]]
        sm.nodes[node_name]["bias"] = node[1]["bias"]

    # recover and preseve the node dist_types
    for node_data in g.nodes(data=True):
        node_name = idx_col_expanded[node_data[0]]
        sm.nodes[node_name]["dist_type"] = node_data[1]["dist_type"]

    # recover the collapsed model from g
    sm_collapsed = StructureModel()
    sm_collapsed.add_nodes_from(list(idx_col.values()))
    for u, v, edge_dict in g.graph["graph_collapsed"].edges.data(True):
        sm_collapsed.add_edge(
            idx_col[u],
            idx_col[v],
            origin="learned",
            weight=edge_dict["weight"],
        )
    sm.graph["graph_collapsed"] = sm_collapsed

    return sm

def create_dag(sm,node_names:list,save_dir=None,weight_threshold=0.0,edge_limit=1000000,do_plot=True):
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
        weight_list = [d['weight'] for (u,v,d) in sm.edges(data=True)]
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
    sorted_edge_list = sorted(sm_l.edges(data=True),key=lambda x : x[-1]['weight'],reverse=True)  # sort by weight
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

    return dag

