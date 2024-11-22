import matplotlib.pyplot as plt
import numpy as np
from loadData import *
from knn import *
from functions import *


# Load wine dataset
wine = load_wine()
data = pd.DataFrame(wine.data, columns=wine.feature_names)  # Convert to DataFrame
targets = wine.target  # Get target values

classes = sorted(set(targets))  # Set delle classi presenti nei target

# Lista dei valori di k
kval = [1, 2, 3, 5, 10, 15, 20, 30, 50]

reps = 3 

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

avgER = [[0 for k in range(len(kval))] for c in range(len(classes))]
avgAcc = [[0 for k in range(len(kval))] for c in range(len(classes))]
avgRec = [[0 for k in range(len(kval))] for c in range(len(classes))]
avgPrec = [[0 for k in range(len(kval))] for c in range(len(classes))]
avgF1 = [[0 for k in range(len(kval))] for c in range(len(classes))]


for c_idx, c in enumerate(classes):  # Usa enumerate per iterare con gli indici
    for k_idx in range(len(kval)):  # Itera sugli indici di kval
        avgER[c_idx][k_idx] = computeAvg(errorsRate[c][k_idx])
        avgAcc[c_idx][k_idx] = computeAvg(accuracies[c][k_idx])
        avgRec[c_idx][k_idx] = computeAvg(recalls[c][k_idx])
        avgPrec[c_idx][k_idx] = computeAvg(precisions[c][k_idx])
        avgF1[c_idx][k_idx] = computeAvg(f1s[c][k_idx])  

data = [avgER, avgAcc, avgRec, avgPrec, avgF1]
titles = ['Average error rate', 'Average accuracy', 'Average recall', 'Average precision', 'Average F1 score']
plotTables(data, titles, kval)