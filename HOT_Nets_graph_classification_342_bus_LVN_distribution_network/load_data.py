import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import util

def read_graphfile():
    distribution_network = nx.read_gml('data/bus390Ex.gml')
    # ****************************************************
    labels = np.load('data/data_1/390_distribution_graphs_labels_1.npz', allow_pickle=True)['arr_0']
    with_without_unserved_samples_indices = np.load('data/data_1/390_with_without_unserved_samples_indices_1.npz', allow_pickle=True)['arr_0']
    degree_PIs = np.load('data/data_1/390_distribution_network_degree_PIs_1.npz', allow_pickle=True)['arr_0']
    betweenness_PIs = np.load('data/data_1/390_distribution_network_betweenness_PIs_1.npz', allow_pickle=True)['arr_0']
    closeness_PIs = np.load('data/data_1/390_distribution_network_closeness_PIs_1.npz', allow_pickle=True)['arr_0']
    # ****************************************************
    edges_failed_list = np.load('data/390_edges_failed.npz', allow_pickle = True)['arr_0']
    node_feature_matrix = np.load('data/390_node_feature_matrix.npz', allow_pickle = True)['arr_0']

    list_of_distribution_graphs = []
    count = 0
    for i in with_without_unserved_samples_indices:
        print(i)
        tmp = edges_failed_list[i]
        # initial network
        initial_graph = nx.read_gml('data/bus390Ex.gml')
        edge_feature_matrix = np.load('data/390_edge_feature_matrix_one_hot.npz', allow_pickle=True)['arr_0']
        #print("edge_feature_matrix is:", edge_feature_matrix.shape)

        deleted_edges_labels = []
        for j in range(len(tmp)):
            deleted_edges_labels.append(tmp[j]['label'])

        deleted_edges_indices = []
        for k in range(len(tmp)):
            tmp_index = int(np.where(edge_feature_matrix[:, 0] == deleted_edges_labels[k])[0][0])
            deleted_edges_indices.append(tmp_index)

        removed_edges = []
        for l in range(len(tmp)):
            tmp_removed_edge = list(distribution_network.edges())[deleted_edges_indices[l]]
            removed_edges.append(tmp_removed_edge)

        new_edge_feature_matrix = np.delete(edge_feature_matrix, deleted_edges_indices, 0)  # remove edges

        # remove edges from initial network
        initial_graph.remove_edges_from(removed_edges)

        # construct new graph
        tmp_G = nx.from_numpy_matrix(nx.to_numpy_matrix(initial_graph))
        tmp_G.graph['label'] = labels[count]  # label assignment
        tmp_G.graph['feat_dim'] = node_feature_matrix.shape[1]  # num_feature assignment; feat_dim will be used in train()
        for u in util.node_iter(tmp_G):
            util.node_dict(tmp_G)[u]['feat'] = node_feature_matrix[u, :]

        edge_counter = 0
        for u, v in tmp_G.edges():
            tmp_G[u][v]['edge_feat'] = new_edge_feature_matrix[edge_counter][2:]
            #print(new_edge_feature_matrix[edge_counter])
            edge_counter = edge_counter + 1

        list_of_distribution_graphs.append(tmp_G)
        count = count + 1


    print("*** Distribution network loading complete ***")
    #print(np.max(account))

    return list_of_distribution_graphs, list(degree_PIs), list(betweenness_PIs), list(closeness_PIs)
