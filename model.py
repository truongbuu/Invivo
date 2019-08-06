"""Build a Graph Conv Model for Mutilabel Classification"""

import numpy as np
import tensorflow as tf
import random

import deepchem as dc
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Feature
from deepchem.models.tensorgraph.layers import Dense, GraphConv, BatchNorm
from deepchem.models.tensorgraph.layers import GraphPool, GraphGather
from deepchem.models.tensorgraph.layers import Dense, SoftMax, SoftMaxCrossEntropy, WeightedError, Stack
from deepchem.models.tensorgraph.layers import Label, Weights
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol

from utils.data_utils import load_csv_dataset, reshape_y_pred
from utils.model_utils import graph_conv_net



class Model(object):
    def __init__(self, data_files, prior = True, batch_size = 100, num_task = 12):
        """Graph Conv Model for Mutlitask Classification.

           Agrs:
           - data_file: path to the dataset.
           - prior: if this value is True, we build a general-purpose model
           (encode the labels as the feature). If it is False, then build the
           traditional model)
           - batch_size: training batch size
           - num_task: number of tasks in the dataset.
        """
        self.batch_size = batch_size
        self.num_task = num_task
        self.prior = prior

        """Load dataset"""
        self.train_data = data_files[0]
        self.vali_data  = data_files[1]
        self.test_data  = data_files[2]# = load_csv_dataset(data_file)
        self.scaled_w = 1/self.train_data.y.mean(axis = 0)

        self.metric = dc.metrics.Metric(
            dc.metrics.roc_auc_score, np.mean, verbose = False, mode="classification")
        if prior == True:
            self.tg,  self.atom_features, self.circular_feat,self.degree_slice, self.membership, self.deg_adjs,\
                    self.labels, self.weights = graph_conv_net(self.batch_size\
                                                    , self.prior, self.num_task)
        else:
            self.tg,  self.atom_features, self.circular_feat, self.degree_slice, self.membership, self.deg_adjs,\
                    self.labels, self.weights= graph_conv_net(self.batch_size\
                                                    , self.prior, self.num_task)


    def data_generator(self, dataset, prior_label , task = None, num_prior = 0\
                                   , epochs=1, pad_batches=True):
        """Data generator for training and evaluation"""
        for epoch in range(epochs):
            for ind, (X_b, y_b, w_b, ids_b) in enumerate(
                dataset.iterbatches(
                    self.batch_size, pad_batches=pad_batches, deterministic=True)):
              d = {}
              for index, label in enumerate(self.labels):
                d[label] = to_one_hot(y_b[:, index])
              #if epochs < 12:
              w_b = w_b*(0.1) + (w_b*y_b*self.scaled_w)/10.0
              multiConvMol = ConvMol.agglomerate_mols(X_b[:,0])
              circular_feat = X_b[:,1:]
              d[self.circular_feat] = circular_feat
              """Encode labels into the atom_features"""
              if prior_label:
                  prior = []
                  if task is None:
                      for e in range(self.batch_size):
                          arr = np.zeros(self.num_task * 2)
                          if random.random() < 0.5:
                              index = random.sample(range(self.num_task), \
                                           random.randint(0, self.num_task -1))
                          else:
                              index = []
                          if len(index) != 0:
                              for sth in index:
                                  if w_b[e,sth] != 0:
                                      if y_b[e,sth] == 1:
                                          arr[2*sth+1] = 1
                                      else:
                                          arr[2*sth] = 1
                                      w_b[e,sth] = w_b[e,sth]*0.001
                          prior.append(arr)
                  else:
                      for e in range(self.batch_size):
                          arr = np.zeros(self.num_task * 2)
                          list_t = list(range(self.num_task))
                          list_t.pop(task)
                          index = random.sample(list_t, num_prior)
                          if len(index) != 0:
                              for sth in index:
                                  if w_b[e,sth] != 0:
                                      if y_b[e,sth] == 1:
                                          arr[2*sth+1] = 1
                                      else:
                                          arr[2*sth] = 1
                                      w_b[e,sth] = w_b[e,sth]*0.001
                          prior.append(arr)
                          w_b[e,task] = 1.0
                          arr[2*task] = 0.
                          arr[2*task+1] = 0.

                  prior = np.array(prior)
                  atom_feat = multiConvMol.get_atom_features()
                  member = multiConvMol.membership

                  new_atom_feats = []
                  for i in range(atom_feat.shape[0]):
                      new_atom_feat = np.concatenate((atom_feat[i],\
                                                        prior[member[i]]))
                      new_atom_feats.append(new_atom_feat)
                  new_atom_feats = np.array(new_atom_feats)
                  d[self.atom_features] = new_atom_feats
              else:
                  d[self.atom_features] = multiConvMol.get_atom_features()

              d[self.weights] = w_b
              d[self.degree_slice] = multiConvMol.deg_slice
              d[self.membership] = multiConvMol.membership
              for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                  d[self.deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
              yield d

    def train(self, epochs = 10):
        """Train the model

           Args:
            epochs: number of training epochs.

           Returns:
             - train_loss: an array contains training loss at different epochs.
             - eval_score: an array consists of ROC-AUC score (average,per class).
        """
        train_loss = []
        eval_score = []
        for epoch in range(epochs):
            tl = self.tg.fit_generator(self.data_generator(self.train_data, prior_label = self.prior,epochs=1))
            es = self.compute_score(data = 'vali')
            train_loss.append(tl)
            eval_score.append(es)
            print ('Epoch %d - training loss %f '% (epoch, tl))
            print ('Validation ROC-AUC Score: ', es[0])
        return train_loss, eval_score

    def compute_score(self, num_prior = 0, data = 'test'):
        """Compute the ROC-AUC Score

           Args:
            num_prior: set the number of prior labels for general purpose model.
            data: a string that indicates which set of the data we evaluate on.

           Returns:
             - An average ROC-AUC score for all the classes.
             - An array consists of ROC-AUC score per class.
        """
        if data == 'test':
            eval_data = self.test_data
        elif data == 'train':
            eval_data = self.train_data
        elif data == 'vali':
            eval_data = self.vali_data
        else:
            print ('Data must take either one of these string values: train, test, vali')
            return None

        if self.prior == False:
            valid_predictions = self.tg.predict_on_generator(self.data_generator(eval_data, prior_label = self.prior ))
            valid_predictions = reshape_y_pred(eval_data.y, valid_predictions)
            valid_scores = self.metric.compute_metric(eval_data.y, valid_predictions, eval_data.w, per_task_metrics= True)
            return valid_scores
        else:
            metric_tasks = []
            for task in range(self.num_task):
                valid_predictions = self.tg.predict_on_generator(self.data_generator(eval_data, \
                                                        prior_label = self.prior, task = task, num_prior= num_prior))
                valid_predictions = reshape_y_pred(eval_data.y, valid_predictions)
                metric_tasks.append(self.metric.compute_metric(eval_data.y, valid_predictions, eval_data.w, per_task_metrics= True)[1][task])
            return sum(metric_tasks)/self.num_task, metric_tasks
