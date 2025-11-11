import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch

def data_split(genes, disease, mapping):
    # Read the labeled genes
    label_gene = pd.read_csv(f'diseases/{disease}/{disease}_gene.csv', index_col=0)

    # Filter the labeled genes that are in the graph
    label_gene = label_gene.loc[np.intersect1d(label_gene.index, genes)]
    known_target_genes = label_gene.index.values

    # Find the genes that are not labeled and sample them
    zero_gene = np.setdiff1d(genes, label_gene.index)
    random.seed(1)
    sample_zero_gene = random.sample(list(zero_gene), len(label_gene.index))

    # Append sampled zero genes to the labeled genes
    for gene in sample_zero_gene:
        new_line = pd.DataFrame({'Label': 0}, index=[gene])
        label_gene = label_gene._append(new_line)

    # Split data into train+val and test
    trainval, test = train_test_split(label_gene, test_size=0.2, stratify=label_gene, random_state=0)

    # Split train+val into train and val
    train, val = train_test_split(trainval, test_size=0.25, stratify=trainval, random_state=0)

    # Get indices and labels for train, val, and test sets
    train_idx = [mapping[t] for t in train.index]
    train_y = torch.tensor(train.Label.astype(int), dtype=torch.float32)

    val_idx = [mapping[t] for t in val.index]
    val_y = torch.tensor(val.Label.astype(int), dtype=torch.float32)

    test_idx = [mapping[t] for t in test.index]
    test_y = torch.tensor(test.Label.astype(int), dtype=torch.float32)

    return (train_idx, train_y), (val_idx, val_y), (test_idx, test_y), known_target_genes

# Please note that you need to provide 'genes', 'disease', and 'mapping' when calling this function.


def data_process():
    databases=['RegNetwork','trrust','JASPAR','MOTIFMAP','EVEX','CHEA','string','CPDB','IREF']
    edges = pd.read_csv('./graphs/edges_kegg.csv')
    edges = edges.dropna()
    edges = edges.drop_duplicates()
    edges['kegg'] = np.ones(len(edges.index))
    for name in databases:
        graph = pd.read_csv('./graphs/edges_'+name+'.csv')
        graph = graph.dropna()
        graph = graph.drop_duplicates()  # remove the repeated edges
        graph[name] = np.ones(len(graph.index))
        edges = pd.merge(edges, graph, on=['source', 'target'], how='outer')
    edges.fillna(0, inplace=True)

    print("Number of edges before reprocessing:",len(edges.values))

    genes = np.union1d(edges.values[:,0],edges.values[:,1])
    print("Number of genes（nodes）in the graph:",len(genes))

    # expression data from GEO
    expression = pd.read_csv('diseases'+disease+disease+'_expression.csv',index_col=0)
    expression = expression.groupby('Gene').mean()
    expression = expression[expression.apply(np.sum, axis=1) != 0]  # drop the zero-expression data
    # print(expression.shape)

    X = np.zeros((len(genes), 0))
    X = pd.DataFrame(X, index=genes)
    columns = [f'expression_{i}' for i in range(expression.shape[1])]
    expression.columns = columns
    X = X.join(expression, how="left")
    X.dropna(inplace=True)  # graph overlap with expression data

    genes_drop = np.setdiff1d(genes, X.index.values)

    # edges overlap with expression
    genes_drop = pd.DataFrame(genes_drop, columns=['source'])
    genes_drop['label1'] = np.ones(len(genes_drop.index))
    edges = pd.merge(edges, genes_drop, on=['source'], how='left')
    genes_drop.columns = ['target', 'label2']
    edges = pd.merge(edges, genes_drop, on=['target'], how='left')
    edges.fillna(0, inplace=True)
    edges['label'] = edges['label1']+ edges['label2']
    edges.drop(columns=['label1', 'label2'], inplace=True)
    edges = edges[edges['label'] == 0]
    edges = edges.values[:, :-1]
    print("Number of edges after overlapping with expression:", len(edges))

    genes=np.union1d(edges[:,0],edges[:,1])
    print("Number of genes（nodes）in the graph after overlapping with expression data:",len(genes))
    X = X.loc[genes]

    N = len(X)
    mapping = dict(zip(genes, range(N)))
    # print("Number of edges before removing self loops",len(edges))

    # Remove self loops
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask]
    print("Number of edges after removing self loops",len(edges))

    edge_index = edges[:, :2]
    edge_feature = edges[:, 2:]
    edge_feature = edge_feature.astype(float)
    edge_index = np.vectorize(mapping.__getitem__)(edge_index)

    degrees = np.zeros((N, 1))
    nodes, counts = np.unique(edge_index, return_counts=True)
    degrees[nodes, 0] = counts
    X = X.values

    X = np.concatenate([X, degrees.reshape((-1, 1))], 1)
    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    print("Shape of node features", X.shape)

    # Torch -------------------------------------------------
    edge_index = torch.from_numpy(edge_index.T)
    edge_index = edge_index.to(torch.long).contiguous()
    X = torch.from_numpy(X).to(torch.float32)
    edge_feature = torch.from_numpy(edge_feature).to(torch.float32)

    edge_new1 = torch.empty(edge_index.shape)
    edge_new1 = edge_new1.to(torch.long)
    edge_new1[[0, 1], :] = edge_index[[1, 0], :]

    edge_new2 = torch.cat((edge_index,edge_new1), dim=1)
    edge_new2 = torch.unique(edge_new2, dim=1)

    return genes, mapping, X, edge_index, edge_new1, edge_new2, edge_feature