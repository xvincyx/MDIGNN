
import torch
import csv, os
import numpy as np
import pickle as pk
import networkx as nx
import scipy.sparse as sp
from torch_geometric.utils import to_undirected
from utils.load_data import data_split, data_process

import pandas as pd
from utils.hermitian import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold

def load_syn(root, name=None):
    data = pk.load(open(root + '.pk', 'rb'))
    return [data]

def to_edge_dataset_sparse(q, edge_index, K, data_split, size, root='../dataset/', laplacian=True, norm=True,
                           max_eigen=2.0, gcn_appr=False):
    f_node, e_node = edge_index[0], edge_index[1]
    L = hermitian_decomp_sparse(f_node, e_node, size, q, norm=True, laplacian=laplacian, max_eigen=2.0,
                                gcn_appr=gcn_appr)
    multi_order_laplacian = cheb_poly_sparse(L, K)

    return multi_order_laplacian

def geometric_dataset_sparse(q, K, root='../dataset/',
                             load_only=False, save_pk=True, laplacian=True, gcn_appr=False, n_splits=5):
    node_feature_file = root + 'scaled_file.csv'
    edge_file = root + 'merged_graph.csv'
    label_file = root + 'label.csv'


    # Load node features
    node_features = pd.read_csv(node_feature_file, index_col=0)
    X = node_features.values.astype('float32')

    # Load edges
    edges = pd.read_csv(edge_file)
    nodes = node_features.index.tolist()
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    filtered_edges = edges[edges['source'].isin(node_to_index) & edges['target'].isin(node_to_index)]
    f_node = np.array([node_to_index[source] for source in filtered_edges['source']], dtype='int64')
    e_node = np.array([node_to_index[target] for target in filtered_edges['target']], dtype='int64')

    # Load labels
    labels = pd.read_csv(label_file)
    label_map = labels.set_index('Gene')['Label'].astype('int').to_dict()
    label = np.array([label_map.get(node, -1) for node in nodes], dtype='int')

    # Filter out nodes with label -1
    valid_indices = np.where(label != -1)[0]
    valid_X = X[valid_indices]
    valid_label = label[valid_indices]
    valid_node_names = [nodes[i] for i in valid_indices]

    pseudo_indices = np.where(label == -1)[0]
    pseudo_X = X[pseudo_indices]
    pseudo_label = label[pseudo_indices]
    pseudo_node_names = [nodes[i] for i in pseudo_indices]

    valid_node_names_df = pd.DataFrame({'Name': valid_node_names})
    name_df=pd.DataFrame({'Name': nodes})
    combined_df = pd.concat([name_df], axis=0, ignore_index=True)
    gene_names = combined_df['Name'].tolist()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    train_masks, val_masks = [], []

    for train_index, val_index in skf.split(valid_indices, label[valid_indices]):
        train_mask = np.zeros(label.shape, dtype='bool_')
        val_mask = np.zeros(label.shape, dtype='bool_')
        train_mask[valid_indices[train_index]] = True
        val_mask[valid_indices[val_index]] = True
        train_masks.append(train_mask)
        val_masks.append(val_mask)

    train_masks = np.stack(train_masks, axis=1)
    val_masks = np.stack(val_masks, axis=1)

    L = hermitian_decomp_sparse(f_node, e_node, X.shape[0], q, norm=True, laplacian=laplacian,
                                max_eigen=2.0, gcn_appr=gcn_appr, edge_weight=None)
    multi_order_laplacian = cheb_poly_sparse(L, K)
    return X, label, train_masks, val_masks, multi_order_laplacian, gene_names