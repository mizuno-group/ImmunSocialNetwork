# -*- coding: utf-8 -*-
"""
Created on 2024-07-23 (Tue) 13:44:14

@author: I.Azuma
"""
# %%
import numpy as np
import networkx as nx
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn import metrics

# %%
# Step 1: Generate a random DAG
def generate_dag(n_nodes, p_edge, seed_start=0):
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    # Iterate over the upper triangle (excluding the diagonal)
    seed = seed_start
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # Sample an edge from the binomial distribution
            np.random.seed(seed)
            adjacency_matrix[i, j] = np.random.binomial(1, p_edge)
            seed += 1
    
    # Convert to networkx graph class
    G = nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
    return G

# Step 2: Sample edges and coefficients
def sample_edges_and_coefficients(G, n_nodes, biased=True, seed_start=0):
    edges = list(G.edges)
    gamma = np.zeros((n_nodes, n_nodes))
    seed = seed_start
    for i, j in edges:
        np.random.seed(seed)
        random_values = np.random.uniform(-1, 1)
        if biased:
            biased_values = np.sign(random_values) * np.log1p(np.abs(random_values)*10) / np.log1p(10)  # normalize
            gamma[i, j] = biased_values
        else:
            gamma[i, j] = random_values
        seed += 1
    return gamma

# Step 3: Generate data
def generate_data(G, gamma, n_nodes, n_samples, disp, nb_p, seed_start=0):

    data = np.zeros((n_samples, n_nodes))
    mu_sum = 0
    source_list = []
    seed = seed_start
    for i in range(n_nodes):
        parents = list(G.predecessors(i))
        if parents:
            pass
        else:
            source_list.append(i)
            # Introduce negative binomial noise for source nodes
            np.random.seed(seed)
            #data[:, i] = np.mean(np.random.negative_binomial(disp, disp / (gen_m + disp), size=n_samples))
            data[:, i] = np.random.negative_binomial(disp, nb_p, size=n_samples)

        seed += 1
    
    for i in range(n_nodes):
        parents = list(G.predecessors(i))
        if parents:
            mu = np.sum(data[:, parents] * gamma[parents, i], axis=1)
            mu = mu - (min(mu)-1e-5)  # Ensure positive value
            data[:, i] = np.random.poisson(mu)
        else:
            pass

        seed += 1

    return data, source_list


# Step 3: Generate data
def generate_data_legacy(G, gamma, n_nodes, n_samples, theta, seed_start=0):
    data = np.zeros((n_samples, n_nodes))
    mu_sum = 0
    source_list = []
    seed = seed_start
    for i in range(n_nodes):
        parents = list(G.predecessors(i))
        if parents:
            mu = np.sum(data[:, parents] * gamma[parents, i], axis=1)
            mu = mu - (min(mu)-0.01)  # Ensure positive value
        else:
            mu = 0
            source_list.append(i)
        seed += 1
        mu_sum += np.mean(mu)
        data[:, i] = np.random.poisson(mu)
    
    # Step 4: Introduce negative binomial noise
    mu_avg = mu_sum/n_nodes
    for i in source_list:
        np.random.seed(seed)
        data[:, i] = np.random.negative_binomial(theta, theta / (mu_avg + theta), size=n_samples)
        seed +=1

    return data

# %%
def main():
    # Parameters
    n_nodes = 20  # Number of genes/nodes
    p_edge = 0.3  # Probability of edge creation
    disp = 1  # Dispersion parameter for the negative binomial distribution
    nb_p = 0.1

    G = generate_dag(n_nodes, p_edge)  # OutEdgeView([(0, 8), (0, 10), (0, 20), ...

    # Sample edges and coefficients
    gamma = sample_edges_and_coefficients(G, n_nodes)

    # Generate data
    n_samples = 200
    data, source_list = generate_data(G, gamma, n_nodes, n_samples, disp, nb_p)

    sns.heatmap(data)
    plt.show()

    # Print the generated data shapes
    print("Data shape:", data.shape)
    print(data)

if __name__ == '__main__':
    main()
