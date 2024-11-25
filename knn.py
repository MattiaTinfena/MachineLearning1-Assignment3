import numpy as np

def knn(trainingSet, trainingTargets, testSet, k, testTargets = None):

    if len(locals()) < 4:
        raise ValueError("At least 4 arguments are required: trainingSet, trainingTargets, testSet, and k")
    if(trainingSet.shape[1] != testSet.shape[1]):
        raise ValueError("The number of features in the training and test sets must be the same")
    if k < 1 or k > len(trainingSet):
        raise ValueError("k must be a positive integer less than the number of training samples")
    
    predictions = []
    for i in range(len(testSet)):
        distances = []
        for j in range(len(trainingSet)):
            distances.append([np.linalg.norm(trainingSet[j] - testSet[i]), j])
        
        sortedDistances = sorted(distances, key=lambda x: x[0])
        kNearest = sortedDistances[:k]
        kNearestTargets = [trainingTargets[x[1]] for x in kNearest]
        mostFrequent = max(set(kNearestTargets), key=kNearestTargets.count)
        
        predictions.append(mostFrequent)

    if testTargets is not None:
        errorRate = 0
        for a in range(len(predictions)):
            if predictions[a] != testTargets[a]:
                errorRate += 1
        errorRate /= len(predictions)
        return predictions, errorRate

    return predictions
