## NumPy Neural Network

This is a Neural Network written in Python by using the NumPy library.

The model implements backpropagation and Stochastic Gradient Descent (SGD) to learn simple datasets.

Important dependencies are NumPy, Scikit-Learn, Pandas and Matplotlib. The exact versions used in this repo are found in requirements.txt, however, this is an auto-generated file from conda on platform osx-arm64 (Mac M1).

The screenshots below are the results from a neural network with 64 nodes in one hidden layer.

### Breast Cancer dataset

|                   Accuracies                   |                     Losses                      |
| :--------------------------------------------: | :---------------------------------------------: |
| ![image](png/breast_cancer-multiclass-acc.png) | ![image](png/breast_cancer-multiclass-loss.png) |

|                 Predictions                 |                 Target                 |
| :-----------------------------------------: | :------------------------------------: |
| ![image](png/breast_cancer-predictions.png) | ![image](png/breast_cancer-target.png) |

|      5-fold Cross Validation       |                 Confusion Matrix                 |
| :--------------------------------: | :----------------------------------------------: |
| ![image](png/breast_cancer-cv.png) | ![image](png/breast_cancer-confusion-matrix.png) |

### Iris dataset

|              Accuracies               |                 Losses                 |
| :-----------------------------------: | :------------------------------------: |
| ![image](png/iris-multiclass-acc.png) | ![image](png/iris-multiclass-loss.png) |

|            Predictions             |            Target             |
| :--------------------------------: | :---------------------------: |
| ![image](png/iris-predictions.png) | ![image](png/iris-target.png) |

|  5-fold Cross Validation  |            Confusion Matrix             |
| :-----------------------: | :-------------------------------------: |
| ![image](png/iris-cv.png) | ![image](png/iris-confusion-matrix.png) |

### Moons dataset

|               Accuracies               |                 Losses                  |
| :------------------------------------: | :-------------------------------------: |
| ![image](png/moons-multiclass-acc.png) | ![image](png/moons-multiclass-loss.png) |

|             Predictions             |             Target             |
| :---------------------------------: | :----------------------------: |
| ![image](png/moons-predictions.png) | ![image](png/moons-target.png) |

|  5-fold Cross Validation   |             Confusion Matrix             |
| :------------------------: | :--------------------------------------: |
| ![image](png/moons-cv.png) | ![image](png/moons-confusion-matrix.png) |

### Wine dataset

|              Accuracies               |                 Losses                 |
| :-----------------------------------: | :------------------------------------: |
| ![image](png/wine-multiclass-acc.png) | ![image](png/wine-multiclass-loss.png) |

|            Predictions             |            Target             |
| :--------------------------------: | :---------------------------: |
| ![image](png/wine-predictions.png) | ![image](png/wine-target.png) |

|  5-fold Cross Validation  |            Confusion Matrix             |
| :-----------------------: | :-------------------------------------: |
| ![image](png/wine-cv.png) | ![image](png/wine-confusion-matrix.png) |

### Digits dataset

|               Accuracies                |                  Losses                  |
| :-------------------------------------: | :--------------------------------------: |
| ![image](png/digits-multiclass-acc.png) | ![image](png/digits-multiclass-loss.png) |

|   5-fold Cross Validation   |             Confusion Matrix              |
| :-------------------------: | :---------------------------------------: |
| ![image](png/digits-cv.png) | ![image](png/digits-confusion-matrix.png) |
