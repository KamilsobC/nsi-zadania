import numpy as np

class Adaline:

    def __init__(self,name,weights=None,bias=None,errors_history=None,learning_rate=0.001,adaline = False,log = True,epochs=1000) -> None:
        self.adaline = adaline
        self.name = name
        self.weights = weights
        self.bias = bias
        self.errors_history = errors_history
        self.learning_rate = learning_rate 
        self.log = log
        self.epochs = epochs
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
        
    def identity_function(self,prediction):
        return prediction
    
    def predict(self, data, weights=None, bias=None,return_sum=False):
        if weights is None:
            weights = self.weights
        if bias is None:
            bias = self.bias

        sum = np.dot(weights, data) + bias
        
        # prediction = self.activation_function(sum)
        prediction = self.identity_function(sum)

        if return_sum:
            return sum
        return prediction
    
    def train(self,data,labels):
        lr = self.learning_rate
        epoch = 0
        error = 999

        weights = np.random.rand(data.shape[1])
        bias = np.random.rand()

        errors = list()
        epochs = list()
            
        while (epoch <= self.epochs) and (error > 0.0001):
            
            loss_ = 0
            for i in range(data.shape[0]):
                
                sum = np.dot(weights, data[i]) + bias
                # label_pred = self.activation_function(sum)
                label_pred = self.identity_function(sum)
                loss = (label_pred - labels[i])**2   
                d_loss = 2*(label_pred - labels[i])
                deltas = [ d_loss*value for value in data[i]]
                weights = weights - (lr * np.array(deltas))
                bias = lr * (- d_loss)

            for index, feature_value_test in enumerate(data):
                label_pred, loss, d_loss = self.forwardprop(feature_value_test, labels[index], weights, bias)
                loss_ += loss

            errors.append(loss_/len(data))
            epochs.append(epoch)
            error = errors[-1]
            epoch += 1
            if self.log:
                print('Epoch {}. loss: {}'.format(epoch, errors[-1]))
            self.weights=weights
            self.bias = bias
            self.errors_history = errors
        return weights, bias, errors

    