import model
import numpy as np
from utils.data_utils import load_csv_dataset
import matplotlib.pyplot as plt
from utils.data_utils import load_csv_dataset, reshape_y_pred
import deepchem as dc

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("train_path", type=str, help="training_path")
parser.add_argument("model_type", type=str, help="model A or B")


def main():
    args = parser.parse_args()

    """Processing the data"""
    data_file  = args.train_path
    model_type = args.model_type
    datasets = load_csv_dataset(data_file)

    """Training and Evaluating the Model"""
    if model_type == 'B':
        modelB1 = model.Model(datasets, prior = False)
        mB1_train,mB1_vali = modelB1.train(epochs = 35)
        scoreB1 = modelB1.compute_score(data='test')
        print ('ROC-AUC score for model B1: ', scoreB1[0])
        plt.figure(figsize=(20,5))
        plt.rcParams.update({'font.size': 15})
        plt.bar(np.arange(1,13,1),scoreB1[1],color='pink',edgecolor='black')
        plt.xticks(np.arange(1, 13, 1))
        plt.xlabel('Target')
        plt.ylabel('ROC-AUC score')
        plt.ylim(ymin=0.6)
        plt.show()

    elif model_type == 'A':
        modelA = model.Model(datasets)
        mA_train,mA_vali =  modelA.train(epochs = 45)

        test_score_total = []
        plt.figure(figsize=(10,10))
        for j in range(30):
            test_score = []
            for i in range(12):
                test_score.append(modelA.compute_score(num_prior= i, data='test')[0])
            test_score_total.append(test_score)
        test_score_total = np.array(test_score_total)
        plt.figure(figsize=(10,5))
        plt.plot(test_score_total.mean(axis = 0))
        m = test_score_total.mean(axis = 0)
        std = test_score_total.std(axis = 0)
        plt.fill_between(np.arange(0, 12, 1.0), (m + 1.0*std).flatten()\
                         , (m - 1.0*std).flatten(),alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.grid(True)
        plt.ylabel('Average AUC-ROC score')
        plt.xlabel('# of labels included')
        plt.xticks(np.arange(0, 12, 1.0))
        plt.show()
    else:
        print ('Invalid model type! It is either A or B')

if __name__ == '__main__':
    main()
