import numpy as np

class NeuralNetwork:
    
    @staticmethod
    def sigmoid(y):
        return 1 / (1 + np.exp(-y))

    @staticmethod
    def derivative_sigmoid(y):
        return y * (1 - y)

    @staticmethod
    def relu(y):
        return np.maximum(0, y)

    @staticmethod
    def derivative_relu(y):
        y[y <= 0] = 0
        y[y > 0] = 1
        return y

    @staticmethod
    def softmax(y):
        exps = np.exp(y - np.max(y))
        return exps / np.sum(exps)

    @staticmethod
    def cross_entropy_loss(predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, max(predictions))
        return np.where(targets==1, -np.log(predictions), 0).sum()

    @staticmethod
    def derivative_cross_entropy(predictions, targets):
        output = predictions.copy()
        output[targets == 1] = output[targets == 1] - 1
        return output

    @staticmethod
    def standardize_dataset(dataset):
        return (dataset - np.mean(dataset)) / np.std(dataset)

    def __init__(self, architecture, modeling_problem, seed):

        np.random.seed(seed)
        self.learning_rate = None
        self.output = None
        self.architecture = architecture
        self.activations = []
        self.modeling_problem = modeling_problem

        # Ensure that we get correct number of output nodes
        if self.modeling_problem != 'multiclass':
            self.architecture[-1] = 1
        
        # He weight initialization
        self.weights = []
        for layer in range(1, len(architecture)):
            self.weights.append(np.random.randn(architecture[layer-1], architecture[layer]) * np.sqrt(2. / architecture[layer-1]))
            # Activation for hidden layer
            self.activations.append('relu')

        # Bias initialization with value zero
        self.biases = []
        for layer in range(1, len(architecture)):
            self.biases.append(np.zeros((1, architecture[layer])))

        # Change last activation to reflect modeling problem
        if self.modeling_problem == 'logistic':
            self.activations[-1] = 'sigmoid'
        elif self.modeling_problem == 'multiclass':
            self.activations[-1] = 'softmax'
        else:
            self.activations[-1] = 'relu'

        self.length = len(self.weights)

    def activation(self, activation_type, y):
        if activation_type == 'sigmoid':
            return self.sigmoid(y)
        if activation_type == 'softmax':
            return self.softmax(y)
        if activation_type == 'relu':
            return self.relu(y)

    def derivative_activation(self, activation_type, y):
        if activation_type == 'sigmoid':
            return self.derivative_sigmoid(y)
        if activation_type == 'relu':
            return self.derivative_relu(y)

    def feed_forward_pass(self, input_data):
        self.layers = [input_data]  # Input layer is the same as the input data
        for layer in range(self.length):
            self.layers.append(self.activation(self.activations[layer],
                                                np.dot(self.layers[layer]
                                                , self.weights[layer])
                                                + self.biases[layer]))

        return self.layers[self.length].flatten()     # Last layer is the output

    def backward_pass_minibatch(self, errors, layers):
        # Gradient array for weights
        dw = []
        for layer in range(1, len(self.architecture)):
            dw.append(np.zeros((self.architecture[layer-1], self.architecture[layer])))

        # Gradient array for biases        
        db = []
        for layer in range(1, len(self.architecture)):
            db.append(np.zeros((1, self.architecture[layer])))

        for i in range(len(errors)):
            if self.modeling_problem == 'multiclass':
                error = [errors[i]]     # Error from output layer
            else:
                error = [np.array(errors[i]).reshape((1))]

            for backward in range(self.length, 0, -1):
                # Calculate error gradient
                err_delta = error * self.derivative_activation('relu', layers[i][backward])

                # Add computed gradients for weights and bias
                dw[backward - 1] += (np.dot(layers[i][backward - 1].T, err_delta))
                db[backward - 1] += np.sum(err_delta, axis=0, keepdims=True)

                # Calculate error for next iteration
                error = np.dot(err_delta, self.weights[backward - 1].T)

        for i in range(len(self.weights)):
            # Update weights and biases
            self.weights[i] -= self.learning_rate * (dw[i] / len(errors))
            self.biases[i] -= self.learning_rate * (db[i] / len(errors))

    def iterate_minibatches(self, X_data, y_data, batch_size, shuffle_between_epochs=False):
        assert X_data.shape[0] == y_data.shape[0]
        if shuffle_between_epochs:
            indices = np.arange(X_data.shape[0])
            np.random.shuffle(indices)
        for start in range(0, X_data.shape[0], batch_size):
            end = min(start + batch_size, X_data.shape[0])
            if shuffle_between_epochs:
                extract = indices[start:end]
            else:
                extract = slice(start, end)
            yield np.array(X_data[extract], dtype=np.float64), np.array(y_data[extract], dtype=np.float64)

    def generate_train_val(self, X_data, y_data, val_ratio):
        assert X_data.shape[0] == y_data.shape[0]
        indices = np.arange(X_data.shape[0])
        np.random.shuffle(indices)
        extract = indices[0:-int(len(X_data)*val_ratio)]
        extract_val = indices[-int(len(X_data)*val_ratio):]
        X_train = np.array(X_data[extract], dtype=np.float64)
        y_train = np.array(y_data[extract], dtype=np.float64)
        X_val = np.array(X_data[extract_val], dtype=np.float64)
        y_val = np.array(y_data[extract_val], dtype=np.float64)
        return X_train, y_train, X_val, y_val


    def fit(self, X_train, y_train, epochs=1000, batch_size=1, learning_rate=0.1, lr_decay=False, use_val=False, use_logging=False):

        X_train = self.standardize_dataset(X_train)
        self.train_loss, self.val_loss = [], []
        self.train_loss_graph, self.val_loss_graph = [], []
        self.train_accuracy, self.val_accuracy = [], []
        self.train_accuracy_graph, self.val_accuracy_graph = [], []
        self.learning_rates = []

        if use_val:
            X_train, y_train, X_val, y_val = self.generate_train_val(X_train, y_train, val_ratio=0.6)

        self.batch_size = batch_size
        self.use_val = use_val
        self.learning_rate = learning_rate * batch_size

        for epoch in range(epochs):
            self.correct_preds = 0
            train_loss_epoch = []
            minibatch_iteration = 0
            for X, y in self.iterate_minibatches(X_train, y_train, batch_size, shuffle_between_epochs=True):
                if (len(X) == batch_size):
                    errors = []
                    layers = []
                    for x_sample, y_sample in zip(X, y):
                        x_sample = x_sample.reshape(1, x_sample.shape[0])
                        self.output = self.feed_forward_pass(x_sample)
                        errors = self.calculate_training_error(errors, y_sample)
                        layers.append(self.layers)
                        minibatch_iteration += 1
                        
                    self.backward_pass_minibatch(errors, layers)
                    train_loss_epoch = self.calculate_training_loss(train_loss_epoch, y_sample)

            if use_val:
                val_loss_epoch, val_acc_epoch = self.calculate_validation_metrics(X_val, y_val)
                self.val_loss.append(val_loss_epoch)
                self.val_loss_graph.append(np.mean(self.val_loss))
                self.val_accuracy.append(val_acc_epoch)
                self.val_accuracy_graph.append(np.mean(self.val_accuracy))

            self.train_loss.append(np.mean(train_loss_epoch))
            self.train_loss_graph.append(np.mean(self.train_loss))
            self.train_accuracy.append(float(self.correct_preds/minibatch_iteration)*100)
            self.train_accuracy_graph.append(np.mean(self.train_accuracy))

            if use_logging:
                self.log_metrics(epoch)

            if lr_decay:
                if epoch == int(epochs*0.7):
                    self.learning_rate *= 0.2
            self.learning_rates.append(self.learning_rate)

        return self

    def predict(self, x_values):
        # A prediction is just a normal feed forward pass
        if self.modeling_problem == 'multiclass':
            return int(np.argmax(self.feed_forward_pass(np.array(x_values))))
        elif self.modeling_problem == 'logistic':
            return int(np.round(self.feed_forward_pass(np.array(x_values))))
        else:
            return float(self.feed_forward_pass(np.array(x_values)))

    def log_metrics(self, epoch):
        if epoch % 10 == 0 and epoch != 0:
            if self.use_val:
                print("Epoch: {}, Learning Rate: {:.4f}, Train Accuracy: {:.2f}%, Train Loss: {:.5f}, Val Accuracy: {:.2f}%, Val Loss: {:.5f}".format(
                    epoch, self.learning_rate, self.train_accuracy[-1], np.mean(self.train_loss), self.val_accuracy[-1], np.mean(self.val_loss)))
            else:
                print("Epoch: {}, Learning Rate: {:.4f}, Train Accuracy: {:.2f}%, Train Loss: {:.5f}".format(
                    epoch, self.learning_rate, self.train_accuracy[-1], self.train_loss[-1]))

    def calculate_error(self, y_sample):
        if self.modeling_problem == 'multiclass' or self.modeling_problem == 'logistic':
            error = self.derivative_cross_entropy(self.output, y_sample)
        else:
            error = self.output - y_sample

        return error

    def calculate_training_loss(self, train_loss_epoch, y_sample):
        if self.modeling_problem == 'multiclass' or self.modeling_problem == 'logistic':
            loss = self.cross_entropy_loss(self.output, y_sample)
            if int(np.argmax(self.output)) == int(np.argmax(y_sample)):
                self.correct_preds += 1
        else:
            loss = np.square(np.subtract(y_sample, self.output)).mean()
            if int(np.round(self.output)) == int(y_sample):
                self.correct_preds += 1

        train_loss_epoch.append(loss)

        return train_loss_epoch


    def calculate_training_error(self, errors, y_sample):
        error = self.calculate_error(y_sample)
        errors.append(error)

        return errors


    def calculate_validation_metrics(self, X_val, y_val):
        val_loss_epoch = []
        correct_preds = 0
        for x_sample, y_sample in zip(X_val, y_val):
            if self.modeling_problem == 'multiclass':
                val_loss_epoch.append(self.cross_entropy_loss(self.feed_forward_pass(x_sample), y_sample))
                if int(np.argmax(self.feed_forward_pass(x_sample))) == int(np.argmax(y_sample)):
                    correct_preds += 1
            elif self.modeling_problem == 'logistic':
                val_loss_epoch.append(self.cross_entropy_loss(self.feed_forward_pass(x_sample), y_sample))
                if int(np.round(self.feed_forward_pass(x_sample))) == int(y_sample):
                    correct_preds += 1
            else:
                val_loss_epoch.append(abs(float(y_sample) - float(self.feed_forward_pass(x_sample))))
                if int(np.round(self.feed_forward_pass(x_sample))) == int(y_sample):
                    correct_preds += 1

        val_loss = np.mean(val_loss_epoch)
        val_accuracy = float(correct_preds/X_val.shape[0])*100
        
        return val_loss, val_accuracy