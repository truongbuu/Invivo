"""For extracting and transforming the data"""

import deepchem as dc
from itertools import islice
from deepchem.feat import graph_features
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol
import numpy as np
import rdkit
from deepchem.data.datasets import NumpyDataset

list_tasks = ['target1', 'target2','target3','target4','target5','target6',
            'target7','target8','target9','target10','target11','target12']


def load_csv_dataset(dataset_file):
    """Read the dataset and return the train, validation and test data"""
    dataset =  dc.utils.save.load_from_disk(dataset_file)
    featurizer1 = dc.feat.ConvMolFeaturizer()
    featurizer2 = dc.feat.CircularFingerprint(size=256)

    loader_graph = dc.data.CSVLoader(
        tasks=list_tasks, smiles_field="smiles",
        featurizer=featurizer1)
    dataset_graph = loader_graph.featurize(dataset_file)

    loader_circular = dc.data.CSVLoader(
        tasks=list_tasks, smiles_field="smiles",
        featurizer=featurizer2)
    dataset_circular = loader_circular.featurize(dataset_file)

    new_feat = np.concatenate((dataset_graph.X.reshape(-1,1), dataset_circular.X), axis = 1)
    new_dataset = NumpyDataset(new_feat, dataset_graph.y, dataset_graph.w, dataset_graph.ids)

    splitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, \
               test_dataset = splitter.train_valid_test_split(new_dataset, seed = 0)

    return train_dataset, valid_dataset, test_dataset


def reshape_y_pred(y_true, y_pred):
    """
    TensorGraph.Predict returns a list of arrays, one for each output
    We also have to remove the padding on the last batch
    Metrics taks results of shape (samples, n_task, prob_of_class)
    """
    n_samples = len(y_true)
    retval = np.stack(y_pred, axis=1)
    return retval[:n_samples]
