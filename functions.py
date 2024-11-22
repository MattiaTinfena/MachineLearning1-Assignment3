import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def plot_cm(matrix, ax, k):

    conf_matrix = matrix

    labels = ["True Pos", "False Pos"]
    ticks = ["False Neg", "True Neg"]

    # Plot the confusion matrix on the provided ax
    custom_cmap = LinearSegmentedColormap.from_list("custom_blues", ["#FFFFFF", "#66B2FF", "#457b9d"])
    cax = ax.matshow(conf_matrix, cmap = custom_cmap)

    # Add values inside the matrix with reduced font size
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf_matrix[i, j]), va = 'center', ha = 'center', color = "black", fontsize = 8)

    # Customize the axes with reduced font size
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation = 0, ha = "center", fontsize = 8)
    ax.set_yticklabels(ticks, rotation = 90, ha = "center", fontsize = 8)
    ax.tick_params(axis = 'y', pad = 15)
    ax.set_title(f"k = {k}", fontsize = 10, pad = 3)

    
    # Add a colorbar to the parent figure
    if hasattr(ax, 'cax'):
        ax.figure.colorbar(cax, ax = ax, fraction = 0.046, pad = 0.04)



def plotConfMatr(matrices, cl, k):
    # Create a 3x3 grid of subplots
    
    fig, axes = plt.subplots(3, 3, figsize = (10, 10))  # Reduced figure size for compactness
    fig.suptitle(f"Confusion Matrices of class {cl}", fontsize = 14, y = 0.98)

    # Plot each confusion matrix in the corresponding subplot
    for i, ax in enumerate(axes.flat):
        if i < len(matrices):
            plot_cm(matrices[i], ax, k[i])
        else:
            # Hide unused subplots
            ax.axis('off')

    # Adjust spacing to bring plots closer together
    plt.subplots_adjust(wspace = 0.08, hspace = 0.3)  # Reduce horizontal and vertical spacing
    
    # Show the figure
    plt.show()   


def computeAvg(metric):
    # Controlla se ci sono sottoliste
    if any(isinstance(i, list) for i in metric):
        # Appiattisce la lista se necessario
        metric = [item for sublist in metric for item in sublist]
    
    # Calcola la media
    if len(metric) == 0:
        raise ValueError("Metric list is empty, cannot compute average.")
    
    return sum(metric) / len(metric)

def plotTables(data, titles, kval):
    
    transposed_d0 = list(map(list, zip(*data[0])))  
    transposed_d1 = list(map(list, zip(*data[1])))  
    transposed_d2 = list(map(list, zip(*data[2])))  
    transposed_d3 = list(map(list, zip(*data[3])))  
    transposed_d4 = list(map(list, zip(*data[4])))  

    # Dati per ogni tabella
    data = [transposed_d0, transposed_d1, transposed_d2, transposed_d3, transposed_d4]

    # Etichette aggiornate
    columns = ["Class 0", "Class 1", "Class 2"]  # Ora diventano colonne
    rows = [f"K= {k}" for k in kval]  # Ora diventano righe

    # Crea un layout con 2 righe e 3 colonne (5 tabelle in totale)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))  # 2 righe e 3 colonne

    # Aggiungi ciascuna tabella nella rispettiva sotto-figura
    for i, ax in enumerate(axes.flat):  # axes.flat rende l'iterazione su tutte le sotto-figure
        if i < 5:  # Solo 5 tabelle, quindi limitiamo l'iterazione
            ax.axis('tight')
            ax.axis('off')

            # Crea la tabella nella i-esima sotto-figura
            table = ax.table(
                cellText=[[round(value, 3) for value in row] for row in data[i]],
                colLabels=columns,
                rowLabels=rows,
                loc='center',
                cellLoc='center'
            )
            
            # Imposta larghezza e altezza per ogni cella
            for cell in table.get_celld().values():
                cell.set_width(0.1)  # Imposta larghezza per ogni cella
                cell.set_height(0.09)

            # Aggiungi un titolo a ciascuna tabella (personalizzabile per ogni tabella)
            titles = [titles[0], titles[1], titles[2], titles[3], titles[4]]
            ax.set_title(f"{titles[i]}", fontsize=14)
        else:
            # Se ci sono meno di 6 tabelle, disabilitiamo le celle extra
            ax.axis('off')

    # Mostrare il plot con tutte le tabelle
    plt.tight_layout()  # Ottimizza la disposizione
    plt.show()


