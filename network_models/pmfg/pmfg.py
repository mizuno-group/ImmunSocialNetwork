# -*- coding: utf-8 -*-
"""
Created on 2023-11-28 (Tue) 15:03:39

Planar Maximally Filtered Graph implementation in python

@author: I.Azuma
"""
import numpy as np
import pandas as pd
import time
import networkx as nx
from networkx.algorithms.planarity import check_planarity
from tqdm import tqdm
import matplotlib.pyplot as plt
import planarity

class GraphHandler:
    def __init__(self):
        self.X = np.array([[],[]])
        self.labels = dict()
        self.n_node = None
        self.n_edge = None
        self.graph = None
        self.centrality = dict()
        self.params = dict()


    def set_data(self,df:pd.DataFrame):
        """
        set adjucency matrix dataframe
        
        Parameters
        ----------
        df: dataframe
            a dataframe of adjucency matrix
        
        """
        idx = list(df.index)
        self.n_node = len(idx)
        self.labels = dict(zip(list(range(self.n_node)),idx))
        self.X = df.values


    def set_graph(self,graph):
        """
        set networkx.Graph object
        
        Parameters
        ----------
        graph: networkx.Graph
        
        """
        self.graph = graph
        self.n_edge = graph.number_of_edges()
        self.n_node = graph.number_of_nodes()
        
            
    def adj2graph(self,X,update:bool=True):
        """
        convert adjacency matrix to graph object of networkx     
        
        Parameters
        ----------
        X: np.array
            adjacency matrix

        update: bool
            whether the stored graph is replaced by the generated one
            
        """
        n = X.shape[0]
        if self.X.shape[0]==0:
            self.X = X
        idx = list(range(n))
        edge = []
        ap = edge.append
        for i in range(n - 1):
            for j in range(i + 1,n):
                ap((idx[i],idx[j],X[i][j]))
        g = nx.Graph()
        g.add_weighted_edges_from(edge)            
        if update:
            self.set_graph(g)
        return g


    def edge2graph(self,edge:pd.DataFrame,update:bool=True):
        """
        construct a graph from edge dataframe

        Parameters
        ----------
        edge: pd.DataFrame
            should be the form as follows:
                1st column: source node (note, src and dest is exchangeable)
                2nd column: destination node
                3rd column: weight

        update: bool
            whether the stored graph is replaced by the generated one
        
        """
        x = edge.values
        x = [(v[0],v[1],v[2]) for v in x]
        g = nx.Graph()
        g.add_weighted_edges_from(x)
        if update:
            self.set_graph(g)
        return g


    def graph2edge(self,graph,sorting:bool=False):
        """
        convert a networkx.Graph object to a dataframe of edge
        
        Parameters
        ----------
        graph: networkx.Graph object

        sorting: bool
            whether the result is sorted or not
        
        """
        edge = []
        ap = edge.append
        for v in graph.edges(data=True):
            ap([v[0],v[1],v[2]['weight']])
        df = pd.DataFrame(edge,columns=['source','dest','weight'])
        if sorting:
            df = df.sort_values(by='weight',ascending=False)
        return df


    def calc_centrality(self,graph,centrality='betweenness'):
        """ 
        calculate centrality of the given graph
        
        Parameters
        ----------
        graph: networkx.Graph object

        centrality: str
            indicates centrality method
            'betweenness', 'pagerank', 'degree', or 'closeness'
        
        """
        if centrality=='betweenness':
            self.centrality = nx.betweenness_centrality(graph)
        elif centrality=='degree':
            self.centrality = nx.degree_centrality(graph)
        elif centrality=='closness':
            self.centrality = nx.closeness_centrality(graph)
        elif centrality=='pagerank':
            self.centrality = nx.pagerank(graph)
        else:
            raise KeyError('!! Wrong centrality: use betweenness, pagerank, degree, or closeness !!')
        self.params['centrality'] = centrality
        return self.centrality


class PMFG(GraphHandler):
    def __init__(self):
        super().__init__()


    def pmfg(self,graph=None,fdr:float=0.05,update:bool=True,boyer=True):
        """
        obtain PMFG from the given graph

        Parameters
        ----------
        graph: networkx.Graph
            a networkx.Graph object

        fdr: float
            indicates the threshold for termination based on false discovery rate
            should be not 0

        update: bool
            whether the stored graph is replaced by the generated one

        """
        sta = time.time()
        if graph is None:
            if self.graph is None:
                if self.X is None:
                    raise ValueError('!! Provide graph as an argument or adjucency matrix by set_data !!')
                else:
                    graph = self.adj2graph(self.X,update=False)
            else:
                graph = self.graph
        n_node = graph.number_of_nodes()
        n_fdr = int(1/fdr)
        self.params['fdr'] = fdr
        graph = self._sort_graph(graph)
        nmax = 3*(n_node - 2)
        g = nx.Graph()
        rejected = 0
        if boyer:
            print("Boyer's method")
            for e in tqdm(graph):
                g.add_edge(e['source'],e['dest'],weight=e['weight'])
                if planarity.is_planar(g):  # NOTE: 240731 update
                    n_edge = g.number_of_edges()
                    if n_edge==nmax:
                        print('--- terminated (#edge={} reached the max) ---'.format(n_edge))
                        break
                    else:
                        rejected = 0
                else:
                    g.remove_edge(e['source'],e['dest'])
                    rejected += 1         
                    if rejected > n_fdr*n_node: # equivalent to FDR
                        print('--- terminated (#rejected={} reached the FDR threshold) ---'.format(rejected))
                        break
        else:   
            print("Left-Right method")
            for e in tqdm(graph):
                g.add_edge(e['source'],e['dest'],weight=e['weight'])
                if check_planarity(g)[0]:
                    n_edge = g.number_of_edges()
                    if n_edge==nmax:
                        print('--- terminated (#edge={} reached the max) ---'.format(n_edge))
                        break
                    else:
                        rejected = 0
                else:
                    g.remove_edge(e['source'],e['dest'])
                    rejected += 1         
                    if rejected > n_fdr*n_node: # equivalent to FDR
                        print('--- terminated (#rejected={} reached the FDR threshold) ---'.format(rejected))
                        break
        end = time.time()
        h,mod = divmod(end - sta,3600)
        m,s = divmod(mod,60)
        print("{0} hr {1} min {2} sec".format(int(h),int(m),round(s,4)))
        if update:
            self.set_graph(g)
        return g


    def _sort_graph(self,graph):
        """
        sort the given graph object of networkx     
        
        Parameters
        ----------
        graph: networkx.Graph
            
        """    
        sorted_edge = []
        ap = sorted_edge.append
        for s,d,w in sorted(graph.edges(data=True),key=lambda x:-x[2]['weight']):
            ap({'source':s,'dest':d,'weight':w['weight']})
        return sorted_edge


class GraphViewer(GraphHandler):
    def __init__(self):
        super().__init__()
        self.pos = None
        self.plt_params = {'figsize':(12,8),'edge_color':'lightgrey','width':0.2,
                           'font_size':14,'node_color':'royalblue','node_size':300,
                           'alpha':0.8}


    def plot(self,graph=None,pos=None,labels:dict=None,n_label:int=5,output:str='',
             fix_size:bool=False,fix_color:bool=False,
             cmap:str='Blues',centrality:str='pagerank',size_params:dict=None,
             plt_params:dict=dict()):
        """
        plot the given graph
        relatively take a long time

        Parameters
        ----------
        graph: networkx.Graph object

        pos: dict
            indicates the positions of nodes calculated by calc_pos method

        labels: dict
            indicates the labels
            keys and values are node indices and corresponding labels, respectively
            like inverter

        n_label: int
            indicates the number of nodes to be labeled
            0 indicates all nodes are labeled

        output: str
            the path for image when saved

        centrality: str
            indicates centrality method
            'betweenness', 'pagerank', 'degree', or 'closeness'

        size_params: dict
            indicates the parameters for size preparation in prep_size method

        plt_params: dict
            indicates the parameters for plot in general
        
        """
        if graph is None:
            if self.graph is None:
                raise ValueError('!! Provide graph to be visualized !!')
        else:
            self.set_graph(graph)
        if pos is None:
            if self.pos is None:
                raise ValueError('!! No pos: provide or prepare pos by calc_pos before this process !!')
            else:
                pos = self.pos
        if len(self.centrality)==0:
            self.calc_centrality(self.graph,centrality)
        # size preparation
        if not fix_size:
            if size_params is not None:
                size = self._prep_size(**size_params)
            else:
                size = self._prep_size()
            size = list(size.values())
        # color preparation
        if not fix_color:
            cm = plt.get_cmap(cmap)
            node_color = list(self.centrality.values())
        # label preparation
        if labels is not None:
            if n_label==0:
                label_focus = labels
            else:
                sc = sorted(self.centrality.items(),key=lambda x: x[1],reverse=True)[:n_label]
                sc = [v[0] for v in sc]
                label_focus = {k:v for k,v in labels.items() if k in sc}
        # plot
        self.plt_params.update(plt_params)
        plt.figure(figsize=self.plt_params['figsize'])
        nx.draw_networkx_edges(self.graph,pos,edge_color=self.plt_params['edge_color'],
                               width=self.plt_params['width'])
        alpha = self.plt_params['alpha']
        if fix_size:
            if fix_color:
                nx.draw_networkx_nodes(self.graph,pos,node_size=self.plt_params['node_size'],
                                       node_color=self.plt_params['node_color'],alpha=alpha)
            else:
                nx.draw_networkx_nodes(self.graph,pos,node_size=self.plt_params['node_size'],
                                       node_color=node_color,cmap=cm,alpha=alpha)
        else:
            if fix_color:
                nx.draw_networkx_nodes(self.graph,pos,node_size=size,node_color=self.plt_params['node_color'],
                                       alpha=alpha)
            else:
                nx.draw_networkx_nodes(self.graph,pos,node_size=size,node_color=node_color,cmap=cm,alpha=alpha)
        if labels is not None:
            nx.draw_networkx_labels(self.graph,pos,label_focus,font_size=self.plt_params['font_size'])
        plt.axis('off')
        if len(output) > 0:
            plt.savefig(output)
        plt.show()
        

    def plot_cluster(self,graph=None,cluster:dict=None,method:str='pagerank',layout:str='spring'):
        """
        plot the given graph
        relatively take a long time

        Parameters
        ----------
        graph: networkx.Graph object

        layout: str
            determines the shape of network
            'spring' or 'kamada_kawai'

        cluster: dict
            indicate where each node belongs

        method: str
            indicates centrality method
            'betweenness', 'pagerank', 'degree', or 'closeness'
        
        """
        raise NotImplementedError


    def calc_pos(self,graph=None,layout:str='spring'):
        """
        set a graph object
        relatively take a long time
        
        Parameters
        ----------
        graph: networkx.Graph object

        layout: str
            determines the shape of network
            'spring' or 'kamada_kawai'
        
        """
        if graph is not None:
            self.graph = graph
        if layout=='spring':
            self.pos = nx.spring_layout(self.graph)
        elif layout=='kamada_kawai':
            self.pos = nx.kamada_kawai_layout(self.graph)
        else:
            raise KeyError('!! Wrong layout: use spring or kamada_kawai !!')


    def set_inverter(self,inverter:dict):
        """
        set the index inverter, from int indices to string one
        
        """
        self.idx_inverter = inverter


    def _prep_size(self,basesize:int=500,power:bool=False,power_val:float=1.5):
        """
        prepare node size according to centrality
        
        basesize: int
            indicate the base size of nodes

        power: bool
            whether node size is prepared by power scaling

        power_val: float
            the value of power for scaling
        
        """
        size = [basesize*v for v in self.centrality.values()]
        if power:
            size = [np.power(v,power_val) for v in size]
        return dict(zip(self.centrality.keys(),size))
