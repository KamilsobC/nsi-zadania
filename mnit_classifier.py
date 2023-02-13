import numpy as np
import json
from PIL import Image

class NN():
    def __init__(self,batch_size=32,hidden_neurons=750,epochs=1000,data_amount=60000,learning_rate=0.0001,path = 'training/params.json'):
        
        np.random.seed(100)
        self.data_amount = data_amount
        self.test_amount = int(data_amount/4) 
        self.batch_size = batch_size
        self.hidden_neurons = hidden_neurons
        self.desired_epochs = epochs
        self.path = path
        self.learning_rate = learning_rate

    def load_data(self):
        with open('training/data.json') as f:
            print('Loading data from json...')
            raw_data = json.load(f)

        self.training_set = np.array(raw_data['images'])[:self.data_amount]
        self.labels = np.array(raw_data['labels'])[:self.data_amount]
        self.test_set = np.array(raw_data['images_test'])[:self.test_amount]
        self.test_labels = np.array(raw_data['labels_test'])[:self.test_amount]

    def _normalize_data_set(self,data):
        data = (data.astype(np.float32) - 127.5) / 127.5
        data =  data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        return data
        
    def _prepare_data_set(self,data,labels):
        keys = np.array(range(data.shape[0]))
        np.random.shuffle(keys)
        data =data[keys]
        labels = labels[keys]
        proc_labels = np.zeros((len(data), 10))
        for i in range(len(data)):
            proc_labels[i][self.labels[i]] = 1
        return data,proc_labels

    def data_preprocessing(self):

        self.training_set = self._normalize_data_set(self.training_set)
        self.test_set = self._normalize_data_set(self.test_set)
        train_data,train_labels = self._prepare_data_set(self.training_set,self.labels)
        self.train_data = train_data
        self.processed_labels = train_labels

    def chunking(self):
        self.steps = self.training_set.shape[0] // self.batch_size

        if self.steps * self.batch_size < self.training_set.shape[0]:
            self.steps += 1

    def load_weights(self,load_from_path=False):
        if load_from_path:
            parameters = None
            with open(self.path) as f:
                parameters = json.load(f)
            self.weights_hidden1 = np.array(parameters['weights_hidden1'])
            self.weights_output = np.array(parameters['weights_output'])
            self.bias_h1 = np.array(parameters['bias_h1'])
            self.bias_o = np.array(parameters['bias_o'])
        else:
            attributes = self.training_set.shape[1]
            output_labels = len(self.processed_labels[0])
            hidden_nodes1 = self.hidden_neurons
            self.weights_hidden1 = np.random.rand(attributes, hidden_nodes1) * 0.01
            self.bias_h1 = np.random.randn(hidden_nodes1)
            self.weights_output = np.random.rand(hidden_nodes1, output_labels) * 0.01
            self.bias_o = np.random.randn(output_labels)

    def save_weights(self):
        data = {
            'weights_hidden1': self.weights_hidden1.tolist(),
            'bias_h1': self.bias_h1.tolist(),
            'weights_output': self.weights_output.tolist(),
            'bias_o': self.bias_o.tolist(),
        }

        with open(self.path, 'w') as json_file:
            print('Saving weights')
            json.dump(data, json_file)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x) *(1-self.sigmoid (x))

    def softmax(self, x):
        try:
            x.shape[1]
            index = 1
        except IndexError:
            index = 0
        expx = np.exp(x)
        return expx / expx.sum(axis=index, keepdims=True)

    def training(self):
        self.load_data()
        self.data_preprocessing()
        self.chunking()
        self.load_weights()
        error_cost = 0

        for i in range(self.desired_epochs):

            for step in range(self.steps):
            
                x_batch = self.training_set[step * self.batch_size:(step + 1) * self.batch_size]
                y_batch = self.processed_labels[step * self.batch_size:(step + 1) * self.batch_size]


                X1 = np.dot(x_batch, self.weights_hidden1) + self.bias_h1
                prediction_h1 = self.sigmoid(X1)

                y = np.dot(prediction_h1, self.weights_output) + self.bias_o
                prediction_o = self.softmax(y)
                
                ######### Back Propagation

                pred_h1 = prediction_h1
                error_cost_o = prediction_o - y_batch
                der_cost_o = np.dot(pred_h1.T, error_cost_o)
                dcost_bo = error_cost_o


                taining_data = x_batch

                weight_o = self.weights_output
                error_cost_h1 = np.dot(error_cost_o, weight_o.T)
                derivative_h1 = self.sigmoid_der(X1)
                pred_h2 = prediction_h1
                der_cost_h1 = np.dot(taining_data.T, derivative_h1 * error_cost_h1)

                dcost_bh1 = error_cost_h1 * derivative_h1

                #hidden
                self.weights_hidden1 -= self.learning_rate * der_cost_h1
                self.bias_h1 -= self.learning_rate * dcost_bh1.sum(axis=0)
                
                #output
                self.weights_output -= self.learning_rate * der_cost_o
                self.bias_o -= self.learning_rate * dcost_bo.sum(axis=0)
                
                loss = np.sum(-y_batch * np.log(prediction_o))
                error_cost = loss
            print(error_cost)

            print('Iterations: ' + str(i))
        self.save_weights()


    def forward(self,batch):
        X1 = np.dot(batch, self.weights_hidden1) + self.bias_h1
        prediction_h1 = self.sigmoid(X1)

        y = np.dot(prediction_h1, self.weights_output) + self.bias_o
        prediction_o = self.softmax(y)
        return prediction_o,prediction_h1

    def predict(self, sample):
        output, _ = self.forward(sample)
        label = np.argmax(output)
        
        return label
        
    def testing(self):
        self.load_weights(load_from_path=True)
        self.load_data()
        self.data_preprocessing()

        error = 0
        correct = 0
        count = 0

        for i in range(len(self.test_set)):
            result = self.predict(self.test_set[i])
            label = self.test_labels[i]
            
            
            if (result == label):
                print('SUCCESS,Result:',result,'Label',label)

                correct += 1
            else:
                print('ERROR,Result:',result,'Label',label)

                error += 1
            
            count += 1
            print(count,str(correct/count))
        # error_result = (error / len(self.test_set)) * 100


if __name__ == "__main__":
    epochs=100
    batch_size = 32
    amount_of_hidden_layer_neurons = 350
    data_amount = 10000
    path = 'training/params3.json'

    net = NN(data_amount = data_amount,hidden_neurons=amount_of_hidden_layer_neurons,batch_size=batch_size,epochs=epochs,path=path)
    # net.load_data()
    # net.data_preprocessing()
    # net.chunking()
    # net.load_weights()
    net.training()
    net.testing()