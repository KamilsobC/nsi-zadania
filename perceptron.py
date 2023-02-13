class Perceptron(object):
    """ Perceptron for demonstrating a binary classifier """

    def __init__(self, learn_rate = 0.01, iterations = 100):
        self.learn_rate = learn_rate
        self.iterations = iterations


    def fit(self, X, y, biased_X = False):
        """ Fit training data to our model """
        X = self._add_bias(X)
        self._initialise_weights(X)
        
        self.errors = []

        for cycle in range(self.iterations):
            trg_error = 0
            for x_i, output in zip(X, y):
                output_pred = self.predict(x_i, biased_X=True)
                trg_update = self.learn_rate * (output - output_pred)
                self.weights += trg_update * x_i
                trg_error += int(trg_update != 0.0)
            self.errors.append(trg_error)
        return self


    def _net_input(self, X):
        """ Net input function (weighted sum) """
        return np.dot(X, self.weights)


    def predict(self, X, biased_X=False):
        """ Make predictions for the given data, X, using unit step function """
        if not biased_X:
            X = self._add_bias(X)
        return np.where(self._net_input(X) >= 0.0, 1, 0)


    def _add_bias(self, X):
        """ Add a bias column of 1's to our data, X """
        bias = np.ones((X.shape[0], 1))
        biased_X = np.hstack((bias, X))
        return biased_X


    def _initialise_weights(self, X):
        """ Initialise weigths - normal distribution sample with standard dev 0.01 """
        random_gen = np.random.RandomState(1)
        self.weights = random_gen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        return self

    
# create a perceptron classifier and train on our data
classifier = Perceptron(learn_rate = 0.1, iterations = 50)
classifier.fit(X, y)

# plot our misclassification error after each iteration of training
plt.plot(range(1, len(classifier.errors) + 1), classifier.errors, marker = 'x')
plt.title("Visualisation of errors")
plt.xlabel('Epochs')
plt.ylabel('Errors')
plt.show()