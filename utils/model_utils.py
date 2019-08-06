"""Create a tensorgraph"""

import tensorflow as tf

import deepchem as dc
from deepchem.models.tensorgraph.tensor_graph import TensorGraph
from deepchem.models.tensorgraph.layers import Feature, Add, Dropout, Flatten
from deepchem.models.tensorgraph.layers import Dense, GraphConv, BatchNorm
from deepchem.models.tensorgraph.layers import GraphPool, GraphGather, Reshape
from deepchem.models.tensorgraph.layers import Dense, SoftMax, SoftMaxCrossEntropy, WeightedError, Stack
from deepchem.models.tensorgraph.layers import Label, Weights, Concat
from deepchem.metrics import to_one_hot
from deepchem.feat.mol_graphs import ConvMol


def graph_conv_net(batch_size, prior, num_task):
    """
    Build a tensorgraph for multilabel classification task

    Return: features and labels layers
    """
    tg = TensorGraph(use_queue=False)
    if prior == True:
        add_on = num_task
    else:
        add_on = 0
    atom_features = Feature(shape=(None, 75 + 2*add_on))
    circular_features = Feature(shape=(batch_size, 256), dtype=tf.float32)

    degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
    membership = Feature(shape=(None,), dtype=tf.int32)
    deg_adjs = []
    for i in range(0, 10 + 1):
        deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
        deg_adjs.append(deg_adj)

    gc1 = GraphConv(
        64 + add_on,
        activation_fn=tf.nn.elu,
        in_layers=[atom_features, degree_slice, membership] + deg_adjs)
    batch_norm1 = BatchNorm(in_layers=[gc1])
    gp1 = GraphPool(in_layers=[batch_norm1, degree_slice, membership] + deg_adjs)


    gc2 = GraphConv(
        64 + add_on,
        activation_fn=tf.nn.elu,
        in_layers=[gc1, degree_slice, membership] + deg_adjs)
    batch_norm2 = BatchNorm(in_layers=[gc2])
    gp2 = GraphPool(in_layers=[batch_norm2, degree_slice, membership] + deg_adjs)

    add = Concat(in_layers = [gp1, gp2])
    add = Dropout(0.5, in_layers =[add])
    dense = Dense(out_channels=128, activation_fn=tf.nn.elu, in_layers=[add])
    batch_norm3 = BatchNorm(in_layers=[dense])
    readout = GraphGather(
        batch_size=batch_size,
        activation_fn= tf.nn.tanh,
        in_layers=[batch_norm3, degree_slice, membership] + deg_adjs)
    batch_norm4 = BatchNorm(in_layers=[readout])

    dense1 = Dense(out_channels=128, activation_fn=tf.nn.elu, in_layers=[circular_features])
    dense1 = BatchNorm(in_layers=[dense1])
    dense1 = Dropout(0.5, in_layers =[dense1])
    dense1 = Dense(out_channels=128, activation_fn=tf.nn.elu, in_layers=[circular_features])
    dense1 = BatchNorm(in_layers=[dense1])
    dense1 = Dropout(0.5, in_layers =[dense1])
    merge_feat = Concat(in_layers = [dense1, batch_norm4])
    merge = Dense(out_channels=256, activation_fn=tf.nn.elu, in_layers=[merge_feat])
    costs = []
    labels = []
    for task in range(num_task):
        classification = Dense(
                out_channels=2, activation_fn=None,in_layers=[merge])
        softmax = SoftMax(in_layers=[classification])
        tg.add_output(softmax)
        label = Label(shape=(None, 2))
        labels.append(label)
        cost = SoftMaxCrossEntropy(in_layers=[label, classification])
        costs.append(cost)
    all_cost = Stack(in_layers=costs, axis=1)
    weights = Weights(shape=(None, num_task))
    loss = WeightedError(in_layers=[all_cost, weights])
    tg.set_loss(loss)
    #if prior == True:
    #    return tg, atom_features,circular_features, degree_slice, membership, deg_adjs, labels, weights#, prior_layer
    return tg, atom_features, circular_features ,degree_slice, membership, deg_adjs, labels, weights
