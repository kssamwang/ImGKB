import numpy as np
import torch
from math import ceil
from scipy.sparse import csr_matrix,lil_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import  roc_auc_score, classification_report


def load_data(dataset):
    graph_indicator = np.loadtxt("datasets/%s/%s_graph_indicator.txt"%(dataset, dataset), dtype=np.int64)
    # the value in the i-th line is the graph_id of the node with node_id i
    edges = np.loadtxt("datasets/%s/%s_A.txt"%(dataset, dataset), dtype=np.int64, delimiter="," )
    # each line correspond to (row, col) resp. (node_id, node_id)
    edges -= 1
    _, graph_size = np.unique(graph_indicator, return_counts=True)
    # _:graph_idx, graph_size:the number of node of a graph
    A = csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(graph_indicator.size, graph_indicator.size))#.todense()
    X = np.loadtxt("datasets/%s/%s_node_labels.txt" % (dataset, dataset), dtype=np.int64).reshape(-1, 1)
    # the value in the i-th line corresponds to the node with node_id i
    enc = OneHotEncoder(sparse_output=False)
    X = enc.fit_transform(X)
    adj = []
    features = []
    start_idx = 0
    for i in range(graph_size.size):
        adj.append(A[start_idx:start_idx + graph_size[i], start_idx:start_idx + graph_size[i]])
        features.append(X[start_idx:start_idx + graph_size[i], :])
        start_idx += graph_size[i]
    class_labels = np.loadtxt("datasets/%s/%s_graph_labels.txt" % (dataset, dataset), dtype=np.int64)
    class_labels = np.where(class_labels==-1, 0, class_labels)
    return adj, features, class_labels



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_batches(adj, features, y, batch_size, graph_pooling_type, device,  shuffle=False):
    N = len(y)
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    n_batches = ceil(N/batch_size)

    adj_lst = list()
    features_lst = list()
    graph_indicator_lst = list()
    y_lst = list()
    graph_pool_lst=list()
    nu=0 ## compute null number
    for i in range(0, N, batch_size):
        n_graphs = min(i+batch_size, N) - i
        n_nodes = sum([adj[index[j]].shape[0] for j in range(i, min(i+batch_size, N))])

        adj_batch = lil_matrix((n_nodes, n_nodes))
        features_batch = np.zeros((n_nodes, features[0].shape[1]))
        graph_indicator_batch = np.zeros(n_nodes)
        y_batch = np.zeros(n_graphs)
        graph_pool_batch = lil_matrix((n_graphs, n_nodes))

        idx = 0
        for j in range(i, min(i+batch_size, N)):
            n = adj[index[j]].shape[0]
            adj_batch[idx:idx+n, idx:idx+n] = adj[index[j]]    
            features_batch[idx:idx+n,:] = features[index[j]]
            graph_indicator_batch[idx:idx+n] = j-i
            y_batch[j-i] = y[index[j]]
            # print(y_batch)
            if graph_pooling_type == "average":
                graph_pool_batch[j-i, idx:idx+n] = 1./n
            else:
                graph_pool_batch[j-i, idx:idx+n] = 1

            idx += n
        if sum(y_batch)==0 or sum(y_batch)==n_graphs:
            nu+=1
            pass
        else:
            adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
            features_lst.append(torch.FloatTensor(features_batch).to(device))
            graph_indicator_lst.append(torch.LongTensor(graph_indicator_batch).to(device))
            y_lst.append(torch.LongTensor(y_batch).to(device))
            graph_pool_lst.append(sparse_mx_to_torch_sparse_tensor(graph_pool_batch).to(device))

    return adj_lst, features_lst, graph_pool_lst, graph_indicator_lst, y_lst, n_batches-nu


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_metrics(logits, labels):
    auc_ = roc_auc_score(labels.detach().cpu().numpy(), logits.detach().cpu().numpy()[:, 1])
    target_names = ['C0', 'C1']
    DICT = classification_report(labels.detach().cpu().numpy(), logits.detach().cpu().numpy().argmax(axis=1),
                                 target_names=target_names, output_dict=True)
    macro_recall = DICT['macro avg']['recall']
    macro_f1 = DICT['macro avg']['f1-score']
    return auc_, macro_recall, macro_f1
