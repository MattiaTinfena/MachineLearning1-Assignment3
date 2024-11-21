import numpy as np

def knn(trainingSet, trainingTargets, testSet, k, testTargets = None):

    if len(locals()) < 4:
        raise ValueError("At least 4 arguments are required: trainingSet, trainingTargets, testSet, and k")
    if(trainingSet.shape[1] != testSet.shape[1]):
        raise ValueError("The number of features in the training and test sets must be the same")
    if k < 1 or k > len(trainingSet):
        raise ValueError("k must be a positive integer less than the number of training samples")
    
    predictions = []  # To store the predicted labels for each test sample
    for i in range(len(testSet)):  # Iterate through each test sample
        distances = []
        for j in range(len(trainingSet)):  # Compute distance to all training samples
            distances.append([np.linalg.norm(trainingSet[j] - testSet[i]), j])
        
        # Sort distances and get indices of the k nearest neighbors
        sortedDistances = sorted(distances, key=lambda x: x[0])
        kNearest = sortedDistances[:k]
        # Retrieve the targets of the k nearest neighbors
        kNearestTargets = [trainingTargets[x[1]] for x in kNearest]
        # Find the most frequent target among the k nearest neighbors
        mostFrequent = max(set(kNearestTargets), key=kNearestTargets.count)
        
        # Append the prediction for the current test sample
        predictions.append(mostFrequent)

    if testTargets is not None:
        errorRate = 0
        for a in range(len(predictions)):
            if predictions[a] != testTargets[a]:
                errorRate += 1
        errorRate /= len(predictions)
        return predictions, errorRate

    return predictions
