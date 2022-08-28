import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def showLossCurves(epochs, trainLosses, valLosses):
    figure(figsize=(8, 6))
    plt.plot(epochs, trainLosses, label='Train loss')
    plt.plot(epochs, valLosses, label='Val loss')
    plt.legend()
    plt.grid()
    plt.show()

def printAccuracy(dataLoader, dataName):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataLoader:
            images, labels = data

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} {dataName} samples: {100 * correct // total} %')

def showConfusionMatrix(dataLoader, classes):
    confusionMatrix = __createConfusionMatrix(dataLoader, classes)

    figure, ax = plt.subplots(figsize=(12, 8))
    image, cbar = __heatmap(confusionMatrix, classes, classes, ax=ax, cmap="magma_r", cbarlabel="Confusion matrix")
    __annotate_heatmap(image, valfmt="{x:.2f}")

    figure.tight_layout()
    plt.show()

def __heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(labels=row_labels)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def __annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def __createConfusionMatrix(dataLoader, classes):
    confusionMatrix = np.zeros((len(classes), len(classes)))

    with torch.no_grad():
        for data in dataLoader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                confusionMatrix[label][prediction] += 1

    for i in range(len(classes)):
        total = sum(confusionMatrix[i])
        for j in range(len(classes)):
            confusionMatrix[i][j] /= total

    return confusionMatrix
