import networkx as nx
import numpy as np
import torch

import pickle
import random

from graph_sampler import GraphSampler

def prepare_val_data(graphs, degree_PIs, betweenness_PIs, closeness_PIs, args, val_idx, max_nodes=0):
    seed = 0
    np.random.seed(seed)
    np.random.shuffle(graphs)
    np.random.shuffle(degree_PIs)
    np.random.shuffle(betweenness_PIs)
    np.random.shuffle(closeness_PIs)

    val_size = len(graphs) // 10
    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx+1) * val_size :]
    val_graphs = graphs[val_idx*val_size: (val_idx+1)*val_size]

    # degree-based PI
    train_degree_PIs = degree_PIs[:val_idx * val_size]
    if val_idx < 9:
        train_degree_PIs = train_degree_PIs + degree_PIs[(val_idx + 1) * val_size:]
    val_degree_PIs = degree_PIs[val_idx * val_size: (val_idx + 1) * val_size]

    # betweenness-based PI
    train_betweenness_PIs = betweenness_PIs[:val_idx * val_size]
    if val_idx < 9:
        train_betweenness_PIs = train_betweenness_PIs + betweenness_PIs[(val_idx + 1) * val_size:]
    val_betweenness_PIs = betweenness_PIs[val_idx * val_size: (val_idx + 1) * val_size]

    # closeness-based PI
    train_closeness_PIs = closeness_PIs[:val_idx * val_size]
    if val_idx < 9:
        train_closeness_PIs = train_closeness_PIs + closeness_PIs[(val_idx + 1) * val_size:]
    val_closeness_PIs = closeness_PIs[val_idx * val_size: (val_idx + 1) * val_size]

    print('Num training graphs: ', len(train_graphs), 
          '; Num validation graphs: ', len(val_graphs))

    print('Num training degree_PIs: ', len(train_degree_PIs),
          '; Num validation degree_PIs: ', len(val_degree_PIs))

    print('Num training betweenness_PIs: ', len(train_betweenness_PIs),
          '; Num validation betweenness_PIs: ', len(val_betweenness_PIs))

    print('Num training closeness_PIs: ', len(train_closeness_PIs),
          '; Num validation closeness_PIs: ', len(val_closeness_PIs))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, train_degree_PIs, train_betweenness_PIs, train_closeness_PIs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, val_degree_PIs, val_betweenness_PIs, val_closeness_PIs, normalize=False, max_num_nodes=max_nodes,
            features=args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim

