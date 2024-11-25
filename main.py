import matplotlib.pyplot as plt
import numpy as np
from loadData import *
from knn import *
from functions import *

#############################
# TASK 1 - OBTAIN A DATASET #
#############################

wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)  # Convert to DataFrame
targets = wine.target  # Get target values

(trainingSet, trainingTargets), (testSet, testTargets) = loadData(data, targets, 0.3)  # 30% of the data is used for testing

###################################
# TASK 2 - BUILD A KNN CLASSIFIER #
###################################

# List of the values of k
kval = [1, 2, 3, 5, 10, 15, 20, 25, 50]


for k in kval:
    predictions, errorRate = knn(trainingSet, trainingTargets, testSet, k, testTargets)

    print(f"K = {k}")
    if k%3 == 0:
        print("WARNING: k mutiple of the class number, possible ties")
    print(f"Error rate: {round(errorRate,3)}\n")

####################################
# TASK 3 - TEST THE KNN CLASSIFIER #
####################################

reps = 10

classes = sorted(set(targets))  # Get the unique classes in the target values

accuracies = [[[[] for r in range(reps)] for k in kval ] for c in classes]
errorsRate = [[[[] for r in range(reps)] for k in kval ] for c in classes]
recalls = [[[[] for r in range(reps)] for k in kval ] for c in classes]
precisions = [[[[] for r in range(reps)] for k in kval ] for c in classes]
f1s = [[[[] for r in range(reps)] for k in kval ] for c in classes]

for r in range(reps):

    #Compute a new testSet and trainingSet for each repetition
    (trainingSet, trainingTargets), (testSet, testTargets) = loadData(data, targets, 0.3)
    
    confusionMatrices = [[] for c in classes]

    for c in classes:

        trainingBin = []
        testBin = []

        for i in range(len(trainingTargets)):
            if trainingTargets[i] == c:
                trainingBin.append(1) 
            else:
                trainingBin.append(0) 

        for i in range(len(testTargets)):
            if testTargets[i] == c:
                testBin.append(1) 
            else:
                testBin.append(0) 

        for k in range(len(kval)):
            confusionMatrix = np.zeros((2, 2), dtype=int)

            predictions, errorRate = knn(trainingSet, trainingBin, testSet, kval[k], testBin)

            for j in range(len(predictions)):
                true_label = testBin[j]
                predicted_label = predictions[j]
                confusionMatrix[true_label, predicted_label] += 1
        
            confusionMatrices[c].append(confusionMatrix)
            
            TP = confusionMatrix[1, 1]
            FP = confusionMatrix[0, 1]
            TN = confusionMatrix[0, 0]
            FN = confusionMatrix[1, 0]

            precision = TP/(TP+FP)
            recall  = TP/(TP+FN)
            f1score = 2*TP/(2*TP+FP+FN)
            accuracy = (TP+TN)/(TP+TN+FP+FN)

            errorsRate[c][k][r].append(errorRate)
            accuracies[c][k][r].append(accuracy)
            recalls[c][k][r].append(recall)
            precisions[c][k][r].append(precision)
            f1s[c][k][r].append(f1score)
    
    #Plot confusion matrices
    if r == 9:
        for c in classes:
            plotConfMatr(confusionMatrices[c], c, kval)

#Plot metrics

avgER = [[0 for k in range(len(kval))] for c in range(len(classes))]
avgAcc = [[0 for k in range(len(kval))] for c in range(len(classes))]
avgRec = [[0 for k in range(len(kval))] for c in range(len(classes))]
avgPrec = [[0 for k in range(len(kval))] for c in range(len(classes))]
avgF1 = [[0 for k in range(len(kval))] for c in range(len(classes))]

medER = [[0 for k in range(len(kval))] for c in range(len(classes))]
medAcc = [[0 for k in range(len(kval))] for c in range(len(classes))]
medRec = [[0 for k in range(len(kval))] for c in range(len(classes))]
medPrec = [[0 for k in range(len(kval))] for c in range(len(classes))]
medF1 = [[0 for k in range(len(kval))] for c in range(len(classes))]

stdER = [[0 for k in range(len(kval))] for c in range(len(classes))]
stdAcc = [[0 for k in range(len(kval))] for c in range(len(classes))]
stdRec = [[0 for k in range(len(kval))] for c in range(len(classes))]
stdPrec = [[0 for k in range(len(kval))] for c in range(len(classes))]
stdF1 = [[0 for k in range(len(kval))] for c in range(len(classes))]

percER = [[0 for k in range(len(kval))] for c in range(len(classes))]
percAcc = [[0 for k in range(len(kval))] for c in range(len(classes))]
percRec = [[0 for k in range(len(kval))] for c in range(len(classes))]
percPrec = [[0 for k in range(len(kval))] for c in range(len(classes))]
percF1 = [[0 for k in range(len(kval))] for c in range(len(classes))]


for c_idx, c in enumerate(classes):
    for k_idx in range(len(kval)):
        avgER[c_idx][k_idx] = computeAvg(errorsRate[c][k_idx])
        avgAcc[c_idx][k_idx] = computeAvg(accuracies[c][k_idx])
        avgRec[c_idx][k_idx] = computeAvg(recalls[c][k_idx])
        avgPrec[c_idx][k_idx] = computeAvg(precisions[c][k_idx])
        avgF1[c_idx][k_idx] = computeAvg(f1s[c][k_idx])  

        medER[c_idx][k_idx] = computeMed(errorsRate[c][k_idx])
        medAcc[c_idx][k_idx] = computeMed(accuracies[c][k_idx])
        medRec[c_idx][k_idx] = computeMed(recalls[c][k_idx])
        medPrec[c_idx][k_idx] = computeMed(precisions[c][k_idx])
        medF1[c_idx][k_idx] = computeMed(f1s[c][k_idx])

        stdER[c_idx][k_idx] = computeStd(errorsRate[c][k_idx])
        stdAcc[c_idx][k_idx] = computeStd(accuracies[c][k_idx])
        stdRec[c_idx][k_idx] = computeStd(recalls[c][k_idx])
        stdPrec[c_idx][k_idx] = computeStd(precisions[c][k_idx])
        stdF1[c_idx][k_idx] = computeStd(f1s[c][k_idx])

        percER[c_idx][k_idx] = computePerc(errorsRate[c][k_idx], 25, 75)
        percAcc[c_idx][k_idx] = computePerc(accuracies[c][k_idx], 25, 75)
        percRec[c_idx][k_idx] = computePerc(recalls[c][k_idx], 25, 75)
        percPrec[c_idx][k_idx] = computePerc(precisions[c][k_idx], 25, 75)
        percF1[c_idx][k_idx] = computePerc(f1s[c][k_idx], 25, 75)

        
avgData = [avgER, avgAcc, avgRec, avgPrec, avgF1]
avgTitles = ['Average error rate', 'Average accuracy', 'Average recall', 'Average precision', 'Average F1 score']
plotTables(avgData, avgTitles, kval)

medData = [medER, medAcc, medRec, medPrec, medF1]
medTitles = ['Median error rate', 'Median accuracy', 'Median recall', 'Median precision', 'Median F1 score']
plotTables(medData, medTitles, kval)

stdData = [stdER, stdAcc, stdRec, stdPrec, stdF1]
stdTitles = ['Std.dev error rate', 'Std.dev accuracy', 'Std.dev recall', 'Std.dev precision', 'Std.dev F1 score']
plotTables(stdData, stdTitles, kval)

percData = [percER, percAcc, percRec, percPrec, percF1]
percTitles = ['Percentile error rate', 'Percentile accuracy', 'Percentile recall', 'Percentile precision', 'Percentile F1 score']
plotTables(percData, percTitles, kval, "Percentile interval: 25-75")