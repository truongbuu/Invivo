# Invivo

## Prerequisites:
```
_ Python 3.5.
_ deepchem 
_ tensorflow 1.12
_ numpy, matplotlib, seaborn, pandas
```

## Instructions:
To train and validate the model : 
```
python3 train_eval.py dataset_path model\_type 
```
where dataset_path is the path to the data file and model_type is either A or B. 
Model A encodes other labels information while model B only relies on the SMILES representation.

To recreate the figures in this report, please take a look at the ```Demo.ipynb``` notebook file. 
Different runs produces different results due to different train-validation-test splits. Please see the report for more details.
