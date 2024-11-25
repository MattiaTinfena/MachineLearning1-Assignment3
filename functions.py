import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plotCm(matrix, ax, k):

    conf_matrix = matrix

    labels = ["Predicted Neg", "Predicted Pos"]
    ticks = ["Actual Neg", "Actual Pos"]

    custom_cmap = LinearSegmentedColormap.from_list("custom_blues", ["#FFFFFF", "#66B2FF", "#457b9d"])
    cax = ax.matshow(conf_matrix, cmap = custom_cmap)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(conf_matrix[i, j]), va = 'center', ha = 'center', color = "black", fontsize = 8)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation = 0, ha = "center", fontsize = 8)
    ax.set_yticklabels(ticks, rotation = 90, ha = "center", fontsize = 8)
    ax.tick_params(axis = 'y', pad = 15)
    if k%2 == 0:
        ax.set_title(f"k = {k}\n(k multiple of 2, possible tides)", fontsize = 10, pad = 3, color = "red")
    else:
        ax.set_title(f"k = {k}", fontsize = 10, pad = 3)
    
    if hasattr(ax, 'cax'):
        ax.figure.colorbar(cax, ax = ax, fraction = 0.046, pad = 0.04)



def plotConfMatr(matrices, cl, k):
    
    # Create a 3x3 grid of subplots
    
    fig, axes = plt.subplots(3, 3, figsize = (10, 10))
    fig.suptitle(f"Confusion Matrices of class {cl}", fontsize = 14, y = 0.98)

    for i, ax in enumerate(axes.flat):
        if i < len(matrices):
            plotCm(matrices[i], ax, k[i])
        else:
            ax.axis('off')

    plt.subplots_adjust(wspace = 0.08, hspace = 0.44)

    plt.show()   


def computeAvg(metric):
    #Remove sublists if present
    if any(isinstance(i, list) for i in metric):
        metric = [item for sublist in metric for item in sublist]
    
    if len(metric) == 0:
        raise ValueError("Metric list is empty, cannot compute average.")
    
    return round((sum(metric) / len(metric)),3)

def computeMed(metric):
    if any(isinstance(i, list) for i in metric):
        metric = [item for sublist in metric for item in sublist]
    ordered = sorted(metric)
    if(len(ordered) % 2 == 0):
        median = [0, 0]
        median[0] = ordered[int(len(ordered)/2)]
        median[1] = ordered[int((len(ordered)/2)) - 1]
        return round(((median[0] + median[1])/2),3)
    else:
        return round(ordered[len(ordered)//2],3)
    
def computeStd(metric):
    if any(isinstance(i, list) for i in metric):
        metric = [item for sublist in metric for item in sublist]
    
    if len(metric) == 0:
        raise ValueError("Metric list is empty, cannot compute standard deviation.")
    
    std = np.std(metric)
    return round(std,3)

def computePerc(metric, perc1, perc2):
    if any(isinstance(i, list) for i in metric):
        metric = [item for sublist in metric for item in sublist]

        percentile25 = np.percentile(metric, perc1)
        percentile75 = np.percentile(metric, perc2)
        return f"[{round(percentile25,3)}, {round(percentile75,3)}]"


def plotTables(data, titles, kval,note=None):

    transposed_d0 = list(map(list, zip(*data[0])))  
    transposed_d1 = list(map(list, zip(*data[1])))  
    transposed_d2 = list(map(list, zip(*data[2])))  
    transposed_d3 = list(map(list, zip(*data[3])))  
    transposed_d4 = list(map(list, zip(*data[4])))  

    data = [transposed_d0, transposed_d1, transposed_d2, transposed_d3, transposed_d4]

    columns = ["Class 0", "Class 1", "Class 2"]
    
    rows = [f"K= {k}" for k in kval]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    for i, ax in enumerate(axes.flat):
        if i < 5:
            ax.axis('tight')
            ax.axis('off')

            table = ax.table(
                cellText=[[value for value in row] for row in data[i]],
                colLabels=columns,
                rowLabels=rows,
                loc='center',
                cellLoc='center'
            )

            for cell in table.get_celld().values():
                cell.set_width(0.25)
                cell.set_height(0.09)
            
            
            for (a, b), cell in table.get_celld().items():
                if kval[a - 1]%2 == 0 and a > 1:
                    cell.set_text_props(color="red")

            ax.set_title(f"{titles[i]}", fontsize=14)
        else:
            ax.axis('off')

    plt.tight_layout()

    axes[1, 2].text(0.5, 0.6, "WARNING: K multiple of 2, possible tides", wrap=True, horizontalalignment='center', fontsize=12, verticalalignment='center', color='red')

    if note is not None:
        axes[1, 2].axis('off')
        axes[1, 2].text(0.5, 0.5, f"NOTE: {note}", wrap=True, horizontalalignment='center', fontsize=12, verticalalignment='center')
    plt.show()
