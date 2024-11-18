from loadData import *
from functions import *


k = [1,2,3,4,5,10,15,20,30,40,50]


# Load data
(trainingSet, trainingTargets), (testSet, testTargets) = loadData(0.3) #30% of the data is used for testing


for i in range(len(k)):
    predictions, errorRate = (knn(trainingSet, trainingTargets, testSet, k[i], testTargets))
    print("errorRate: ", round(errorRate,3), "k: ", k[i])