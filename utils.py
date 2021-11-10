import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd

def standardize_dataset(dataset):
    return (dataset - np.mean(dataset)) / np.std(dataset)

def prepare_dataset(X, y, modeling_problem):
    X = standardize_dataset(X)
    if modeling_problem == 'multiclass':
        y = np.array(pd.get_dummies(y))
    else:
        y = np.expand_dims(y, axis=1)
    return X, y

def prepare_dataset_cv(X, y, modeling_problem):
    X = standardize_dataset(X)
    y_original = y
    if modeling_problem == 'multiclass':
        y = np.array(pd.get_dummies(y))
    else:
        y = np.expand_dims(y, axis=1)
    return X, y, y_original


def save_predictions(model, X, y, name):    
    X_val = standardize_dataset(X)
    preds = []
    for sample in range(X_val.shape[0]):
        preds.append(model.predict(X_val[sample]))

    fig = plt.figure()
    scatter = plt.scatter(X[:, 0], X[:, 1], c=np.array(preds).ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    fig.savefig(f'png/{name}-predictions.png', dpi=fig.dpi)
    print(f'Figure saved as png/{name}-predictions.png')

    fig = plt.figure()
    scatter = plt.scatter(X[:, 0], X[:, 1], c=np.array(y).ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    fig.savefig(f'png/{name}-target.png', dpi=fig.dpi)
    print(f'Figure saved as png/{name}-target.png')

def save_accuracies(model, modeling_problem, name):
    fig = plt.figure()
    plt.plot(model.train_accuracy_graph)
    plt.plot(model.val_accuracy_graph)
    plt.title('Model Accuracy')
    plt.ylabel('%',rotation=0)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc = 'lower right')
    fig.savefig(f'png/{name}-{modeling_problem}-acc.png', dpi=fig.dpi)
    print(f'Figure saved as png/{name}-{modeling_problem}-acc.png')

def save_multiple_accuracies(model, model2, name, name2):
    fig = plt.figure()
    plt.plot(model.val_accuracy)
    plt.plot(model2.val_accuracy)
    plt.title('Model Accuracy')
    plt.ylabel('%',rotation=0)
    plt.xlabel('Epoch')
    plt.legend([name, name2], loc = 'lower right')
    fig.savefig(f'png/{name}-{name2}-acc.png', dpi=fig.dpi)
    print(f'Figure saved as png/{name}-vs-{name2}-acc.png')

def save_losses(model, type, name):
    fig = plt.figure()
    plt.plot(model.train_loss_graph)
    plt.plot(model.val_loss_graph)
    plt.plot(model.learning_rates)
    plt.title('Model Loss')
    plt.ylabel('Loss',rotation=0)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val', 'LR'], loc = 'upper right')
    fig.savefig(f'png/{name}-{type}-loss.png', dpi=fig.dpi)
    print(f'Figure saved as png/{name}-{type}-loss.png')

def save_cv_scores(cv_scores, k_split, name):
    fig = plt.figure()
    x_ticks = list(range(1, 6))
    bar = plt.bar(x_ticks, cv_scores, color = 'green', width = 0.4)
    ax = plt.gca()
    ax.set_ylim([min(cv_scores)-1, 100])
    plt.title(f'{k_split}-Fold CV')
    plt.ylabel('%',rotation=0)
    plt.xlabel('Run #')
    plt.legend(bar, ['Mean: {:.2f}%'.format(np.mean(cv_scores)), 'Std.dev: {:.2f}'.format(np.std(cv_scores))], loc = 'upper right')
    fig.savefig(f'png/{name}-cv.png', dpi=fig.dpi)
    print(f'Figure saved as png/{name}-cv.png')

def save_confusion_matrix(model, X, y, name):
    X_val = standardize_dataset(X)
    y_val = y
    preds = []

    for sample in range(X_val.shape[0]):
        preds.append(model.predict(X_val[sample]))

    conf_matrix = confusion_matrix(y_val, preds)
    fig = plt.figure(figsize=(3, 2), dpi=150)
    display = ConfusionMatrixDisplay(conf_matrix).plot(cmap='plasma', colorbar=False)
    ax = plt.gca()
    ax.grid(False)
    display.figure_.savefig(f'png/{name}-confusion-matrix.png', dpi=fig.dpi, bbox_inches="tight")
    print(f'Figure saved as png/{name}-confusion-matrix.png')
