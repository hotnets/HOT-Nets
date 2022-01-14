import networkx as nx
import numpy as np
import torch
import torch.utils.data

import util
#from util import create_edge_adj # we consider hodge 1-laplacian rather than edge adj
from compute_hodge_laplacian import compute_hodge_basis_matrices, compute_hodge_laplacian

class GraphSampler(torch.utils.data.Dataset):
    ''' Sample graphs and nodes in graph
    '''
    # max_num_edges needs to be tuned manually here
    def __init__(self, G_list, degree_PI_list, betweenness_PI_list, closeness_PI_list, features='default', edge_features = 'default', normalize=True, assign_feat='default', max_num_nodes=0, max_num_edges = 131):
        self.adj_all = [] # for node-level
        self.hodge_lap_all = [] # for edge-level
        self.len_all = []
        self.feature_all = []
        self.edge_feature_all = []
        self.label_all = []
        self.degree_PIs_all = []
        self.betweenness_PIs_all = []
        self.closeness_PIs_all = []
        
        self.assign_feat_all = []
        self.assign_edge_feat_all = []

        if max_num_nodes == 0:
            self.max_num_nodes = max([G.number_of_nodes() for G in G_list])
        else:
            self.max_num_nodes = max_num_nodes

        self.max_num_edges = max_num_edges

        #if features == 'default':
        self.feat_dim = util.node_dict(G_list[0])[0]['feat'].shape[0] # for node-level
        self.edge_feat_dim = list(G_list[0].edges(data = True))[0][2]['edge_feat'].shape[0] # for edge-level

        for G in G_list:

            adj = np.array(nx.to_numpy_matrix(G)) # adj in node-level
            if normalize:
                sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
                adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
            self.adj_all.append(adj)

            B1, B2 = compute_hodge_basis_matrices(G)
            hodge_lap = compute_hodge_laplacian(B1, B2) # hodge 1-laplacian
            self.hodge_lap_all.append(hodge_lap)

            self.len_all.append(G.number_of_nodes())
            self.label_all.append(G.graph['label'])
            # feat matrix: max_num_nodes x feat_dim
            if features == 'default':
                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(G.nodes()):
                    f[i,:] = util.node_dict(G)[u]['feat']
                self.feature_all.append(f)

            elif features == 'id':
                self.feature_all.append(np.identity(self.max_num_nodes))
            elif features == 'deg-num':
                degs = np.sum(np.array(adj), 1)
                degs = np.expand_dims(np.pad(degs, [0, self.max_num_nodes - G.number_of_nodes()], 0),
                                      axis=1)
                self.feature_all.append(degs)
            elif features == 'deg':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>max_deg] = max_deg
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                feat = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                f = np.zeros((self.max_num_nodes, self.feat_dim), dtype=float)
                for i,u in enumerate(util.node_iter(G)):
                    f[i,:] = util.node_dict(G)[u]['feat']

                feat = np.concatenate((feat, f), axis=1)

                self.feature_all.append(feat)
            elif features == 'struct':
                self.max_deg = 10
                degs = np.sum(np.array(adj), 1).astype(int)
                degs[degs>10] = 10
                feat = np.zeros((len(degs), self.max_deg + 1))
                feat[np.arange(len(degs)), degs] = 1
                degs = np.pad(feat, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                        'constant', constant_values=0)

                clusterings = np.array(list(nx.clustering(G).values()))
                clusterings = np.expand_dims(np.pad(clusterings, 
                                                    [0, self.max_num_nodes - G.number_of_nodes()],
                                                    'constant'),
                                             axis=1)
                g_feat = np.hstack([degs, clusterings])
                if 'feat' in util.node_dict(G)[0]:
                    node_feats = np.array([util.node_dict(G)[i]['feat'] for i in range(G.number_of_nodes())])
                    node_feats = np.pad(node_feats, ((0, self.max_num_nodes - G.number_of_nodes()), (0, 0)),
                                        'constant')
                    g_feat = np.hstack([g_feat, node_feats])

                self.feature_all.append(g_feat)

            # for edge features
            if edge_features == 'default':
                f_e = np.zeros((self.max_num_edges, self.edge_feat_dim), dtype=float)
                counter = 0
                for _, _, att in G.edges(data = True):
                    f_e[counter,:] = att['edge_feat']
                    counter = counter + 1
                self.edge_feature_all.append(f_e)

            if assign_feat == 'id':
                self.assign_feat_all.append(
                        np.hstack((np.identity(self.max_num_nodes), self.feature_all[-1])) )
            else:
                self.assign_feat_all.append(self.feature_all[-1])
                self.assign_edge_feat_all.append(self.edge_feature_all[-1])
            
        self.feat_dim = self.feature_all[0].shape[1]
        self.assign_feat_dim = self.assign_feat_all[0].shape[1]

        for ii in degree_PI_list:
            self.degree_PIs_all.append(ii)

        for jj in betweenness_PI_list:
            self.betweenness_PIs_all.append(jj)

        for kk in closeness_PI_list:
            self.closeness_PIs_all.append(kk)

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        hodge_lap = self.hodge_lap_all[idx]
        num_edges = hodge_lap.shape[0]
        hodge_lap_padded = np.zeros((self.max_num_edges, self.max_num_edges))
        hodge_lap_padded[:num_edges, :num_edges] = hodge_lap

        degree_PI = self.degree_PIs_all[idx]
        betweenness_PI = self.betweenness_PIs_all[idx]
        closeness_PI = self.closeness_PIs_all[idx]

        # use all nodes for aggregation (baseline)

        return {'adj':adj_padded,
                'hodge_lap': hodge_lap_padded, # edge_adj
                'feats':self.feature_all[idx].copy(),
                'edge_feats': self.edge_feature_all[idx].copy(),
                'degree_PI': degree_PI,
                'betweenness_PI': betweenness_PI,
                'closeness_PI': closeness_PI,
                'label':self.label_all[idx],
                'num_nodes': num_nodes,
                'assign_feats':self.assign_feat_all[idx].copy(),
                'assign_edge_feats': self.assign_edge_feat_all[idx].copy()}

