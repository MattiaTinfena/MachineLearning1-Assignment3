from sklearn.datasets import load_wine
import numpy as np
import pandas as pd

def loadData(data, targets, tsLength):

    # Normalize the entire dataset
    data_min = np.min(data, axis=0)  # Feature-wise min
    data_max = np.max(data, axis=0)  # Feature-wise max

    data = (data - data_min) / (data_max - data_min)  # Normalize the entire dataset

    # Split the dataset into training and test sets
    numTrainSamples = round(len(data) * (1 - tsLength))
    trainingSetIndices = np.random.permutation(len(data))[:numTrainSamples]
    testSetIndices = np.setdiff1d(np.arange(len(data)), trainingSetIndices)  # Indices not in training set

    # Create training and test sets
    trainingSet = data.iloc[trainingSetIndices].values
    trainingSetTargets = targets[trainingSetIndices]

    testSet = data.iloc[testSetIndices].values
    testSetTargets = targets[testSetIndices]

    return (trainingSet, trainingSetTargets), (testSet, testSetTargets)
