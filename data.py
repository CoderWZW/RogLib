from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
from scipy.sparse.csgraph import connected_components

class Data(object):
    def __init__(self, config):
        print("data initialization")
        self.model_name = config['model_name']
        self.data_path = config['data_path']
        print(self.model_name)


    def load(self):
        print("load data")
        if self.model_name == 'nettack':
            self._A_obs, self._X_obs, self._z_obs = load_npz(self.data_path)


    def process(self):
        print("data process")
        if self.model_name == 'nettack':
            
            self._A_obs = self._A_obs + self._A_obs.T
            self._A_obs[self._A_obs > 1] = 1
            lcc = largest_connected_components(self._A_obs)

            self._A_obs = self._A_obs[lcc][:,lcc]
            
            #print(self._A_obs)

            assert np.abs(self._A_obs - self._A_obs.T).sum() == 0, "Input graph is not symmetric"
            assert self._A_obs.max() == 1 and len(np.unique(self._A_obs[self._A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
            assert self._A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

            self._X_obs = self._X_obs[lcc].astype('float32')
            self._z_obs = self._z_obs[lcc]
            '''
                N：the number of nodes
                K: the number of classes
                An: normalization A
                sizes: [16, K]
                degrees: the degree of each nodes
            '''
            self._N = self._A_obs.shape[0]
            self._K = self._z_obs.max()+1
            self._Z_obs = np.eye(self._K)[self._z_obs]
            self._An = preprocess_graph(self._A_obs)
            self.sizes = [16, self._K]
            self.degrees = self._A_obs.sum(0).A1
            print(self._An, self._X_obs, self._Z_obs)
            print(type(self._An), type(self._X_obs), type(self._Z_obs))

    
    def train_val_test_split(self):
        seed = 15
        unlabeled_share = 0.8
        val_share = 0.1
        train_share = 1 - unlabeled_share - val_share
        np.random.seed(seed)

        self.split_train, self.split_val, self.split_unlabeled = train_val_test_split_tabular(np.arange(self._N),
                                                                            train_size=train_share,
                                                                            val_size=val_share,
                                                                            test_size=unlabeled_share,
                                                                            stratify=self._z_obs)
        #print(self.split_train, self.split_val, self.split_unlabeled)

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name,allow_pickle=True) as loader:
        loader = dict(loader,allow_pickle=True)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels

def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):

    """
    Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized