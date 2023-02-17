import numpy as np

class Perceptron:

    def __init__(self,name,weights=None,bias=None,errors_history=None,learning_rate=0.001) -> None:
        self.name = name
        self.weights = weights
        self.bias = bias
        self.errors_history = errors_history
        self.learning_rate = learning_rate  
    def __str__(self):
        return self.name
    def forwardprop(self,data,label,weights=None,bias=None):
        if weights is None:
            weights = self.weights
        if bias is None:
            bias = self.bias

        label_pred = self.predict(data, weights, bias)
        loss = (label_pred - label)**2   
        d_loss = 2*(label_pred - label)
        return label_pred, loss, d_loss

    def backprop(self,data, d_loss):
        deltas = list()
        for value in data:
            deltas.append(d_loss*value)
        return deltas
    
    def activation_function(self,prediction):
        if prediction >= 0:
            return 1
        return 0
        
    def predict(self, data, weights=None, bias=None,return_sum=False):
        if weights is None:
            weights = self.weights
        if bias is None:
            bias = self.bias

        sum = np.dot(weights, data) + bias
        
        prediction = self.activation_function(sum)
        if return_sum:
            return sum
        return prediction
    
    def train(self,data,labels,log=False):
        lr = self.learning_rate
        epoch = 0
        error = 999

        weights = np.random.rand(data.shape[1])
        bias = np.random.rand()

        errors = list()
        epochs = list()
            
        while (epoch <= 1000) and (error > 0.0001):
            
            loss_ = 0
            for i in range(data.shape[0]):

                label_pred, loss, d_loss = self.forwardprop(data[i], labels[i], weights, bias)

                partial_derivates = self.backprop(data[i], d_loss)
                
                weights = weights - (lr * np.array(partial_derivates))

            for index, feature_value_test in enumerate(data):
                label_pred, loss, d_loss = self.forwardprop(feature_value_test, labels[index], weights, bias)
                loss_ += loss

            errors.append(loss_/len(data))
            epochs.append(epoch)
            error = errors[-1]
            epoch += 1
            if log:
                print('Epoch {}. loss: {}'.format(epoch, errors[-1]))
            self.weights=weights
            self.bias = bias
            self.errors_history = errors
        return weights, bias, errors

    