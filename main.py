import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold 

from numpy_net import NeuralNetwork
from utils import prepare_dataset, prepare_dataset_cv, save_accuracies, save_confusion_matrix, save_cv_scores, save_losses, save_predictions

seed = 4
np.random.seed(seed)    # Seed for reproducibility

def train(X_data, y_data, nn_architecture, modeling_problem, lr, epochs=1000):
    X, y = prepare_dataset(X_data, y_data, modeling_problem)
    model = NeuralNetwork(nn_architecture, modeling_problem, seed)
    model = model.fit(np.array(X), np.array(y), epochs, batch_size=1, learning_rate=lr, lr_decay=True, use_val=True, use_logging=True)
    save_accuracies(model, modeling_problem, name)
    save_losses(model, modeling_problem, name)
    save_predictions(model, X_data, y_data, name)
    save_confusion_matrix(model, X_data, y_data, name)
    return model

def train_cross_validation(X_data, y_data, nn_architecture, modeling_problem, lr, epochs=1000):
    X, y, y_original = prepare_dataset_cv(X_data, y_data, modeling_problem)
    cv_scores = []
    k_split = 5
    print(f'Starting {k_split}-fold cross validation')
    kfold = StratifiedKFold(n_splits=k_split, shuffle=True,random_state=seed) 
    for train, val in kfold.split(X, y_original):
        model = NeuralNetwork(nn_architecture, modeling_problem, seed)
        model = model.fit(np.array(X[train]), np.array(y[train]), epochs, batch_size=1, learning_rate=lr, lr_decay=True, use_val=False, use_logging=False)
        correct_preds = 0
        for val_sample, val_target in zip(X[val], y[val]):
            if modeling_problem == 'multiclass':
                if int(model.predict(val_sample)) == int(np.argmax(val_target)):
                    correct_preds += 1
            elif modeling_problem == 'logistic':
                if int(model.predict(val_sample)) == int(val_target):
                    correct_preds += 1
            else:
                if int(model.predict(val_sample)) == int(val_target):
                    correct_preds += 1
        cv_scores.append((float(correct_preds)/len(X[val])) * 100)
        print("{:.2f}% (+/- {:.2f})".format(np.mean(cv_scores), np.std(cv_scores)))
    save_cv_scores(cv_scores, k_split, name)
    return cv_scores

X_data, y_data, name = load_breast_cancer().data, load_breast_cancer().target, 'breast_cancer'    # Binary problem
#(X_data, y_data), name = make_moons(n_samples = 1000, noise=0.2, random_state=100), 'moons'  # Binary problem
#X_data, y_data, name = load_iris().data, load_iris().target, 'iris' # 3 classes
#X_data, y_data, name = load_wine().data, load_wine().target, 'wine' # 3 classes
#X_data, y_data, name = load_digits().data, load_digits().target, 'digits' # 10 classes

neural_network_architecture = [     # Neural Network architecture. Each item in the list represents number of nodes in each layer
    X_data.shape[1],                # Number of input nodes == number of features in our data
    64,                             # Number of nodes in hidden layer
    2                               # Number of output nodes. Regression: 1, Classification: Binary = 1, Multiclass = Number of classes
]

epochs = 850                        # breast cancer
#epochs = 850                        # moons
#epochs = 350                        # iris
#epochs = 2000                       # wine
#epochs = 200                        # digits
learning_rate = 0.005               # breast cancer multiclass
#learning_rate = 0.007               # moons multiclass
#learning_rate = 0.01                # iris multiclass
#learning_rate = 0.01                # wine multiclass
#learning_rate = 0.001               # digits multiclass

modeling_problem = 'multiclass'
#modeling_problem = 'logistic'
#modeling_problem = 'regression'

model = train(X_data, y_data, neural_network_architecture, modeling_problem, learning_rate, epochs)
cv_scores = train_cross_validation(X_data, y_data, neural_network_architecture, modeling_problem, learning_rate, epochs)