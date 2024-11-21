import matplotlib.pyplot as plt
import numpy as np
from loadData import *
from knn import *
from plotConfusionMatrices import *


# Load wine dataset
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)  # Convert to DataFrame
targets = wine.target  # Get target values

classes = set(targets)  # Set delle classi presenti nei target

# Lista dei valori di k
kval = [1, 2, 3, 5, 10, 15, 20, 30, 50]

reps = 10 

accuracies = [[[[] for r in range(reps)] for k in kval ] for c in classes]
errorsRate = [[[[] for r in range(reps)] for k in kval ] for c in classes]
recalls = [[[[] for r in range(reps)] for k in kval ] for c in classes]
precisions = [[[[] for r in range(reps)] for k in kval ] for c in classes]
f1s = [[[[] for r in range(reps)] for k in kval ] for c in classes]

for r in range(reps):
    #Compute a new testSet and trainingSet for each repetition
    (trainingSet, trainingTargets), (testSet, testTargets) = loadData(data, targets, 0.3)  # 30% of the data is used for testing
    
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

        # Calcolo delle confusion matrices per ogni k
        for k in range(len(kval)):
            confusionMatrix = np.zeros((2, 2), dtype=int)

            # Esegui il kNN e ottieni le predizioni
            predictions, errorRate = knn(trainingSet, trainingBin, testSet, kval[k], testBin)

            for j in range(len(predictions)):
                true_label = testBin[j]
                predicted_label = predictions[j]
                confusionMatrix[true_label, predicted_label] += 1
        
            confusionMatrices[c].append(confusionMatrix)
            
            # Calcolo delle metriche
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

    
    #PLOT CONFUSION MATRICES
    if r == 9:
        for c in classes:
            plotConfMatr(confusionMatrices[c], c, kval)

#PLOT METRICS
avgER = [[[] for k in kval] for c in classes]
avgAcc = [[[] for k in kval] for c in classes]
avgRec = [[[] for k in kval] for c in classes]
avgPrec = [[[] for k in kval] for c in classes]
avgF1 = [[[] for k in kval] for c in classes]

for c in classes:
    for k in range(len(kval)):
        avgER[c][k] = computeAvg(errorsRate[c][k])
        avgAcc[c][k] = computeAvg(accuracies[c][k])
        avgRec[c][k] = computeAvg(recalls[c][k])
        avgPrec[c][k] = computeAvg(precisions[c][k])
        avgF1[c][k] = computeAvg(f1s[c][k])                        
        if kval[k] % 2 == 0:
            print("WARNING: k:", kval[k], "divisibile by 2")
            print()
        print("Class: ", c , "k: ", kval[k])
        print("Average errorRate: ", round(avgER[c][k],3))
        print("Average Precision: ", round(avgPrec[c][k], 3))
        print("Average Recall: ", round(avgRec[c][k], 3))
        print("Average F1 score: ", round(avgF1[c][k], 3))
        print("Average Accuracy: ", round(avgAcc[c][k], 3))
        print()