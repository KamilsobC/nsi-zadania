import numpy as np
import json
from PIL import Image

class NN():
    def __init__(self,batch_size=32,hidden_neurons=10,epochs=1000,data_amount=60000,learning_rate=0.0001,path = 'training/params.json'):
        
        np.random.seed(100)
        self.data_amount = data_amount
        self.test_amount = int(data_amount/4) 
        self.batch_size = batch_size
        self.steps = data_amount // self.batch_size

        self.desired_epochs = epochs
        self.path = path
        self.learning_rate = learning_rate
        self.hidden_neurons = hidden_neurons

    def load_data(self,train=True):
        with open('training/data.json') as f:
            print('Loading data from json...')
            raw_data = json.load(f)
        if train:
            self.training_set = np.array(raw_data['images'])[:self.data_amount]
            self.labels = np.array(raw_data['labels'])[:self.data_amount]
        else:
            self.test_set = np.array(raw_data['images_test'])[:self.test_amount]
            self.test_labels = np.array(raw_data['labels_test'])[:self.test_amount]

    def _normalize_data_set(self,data):
        data = (data.astype(np.float32) - 127.5) / 127.5
        data =  data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        return data
    

    def data_preprocessing(self,train=True):
        if train:
            self.training_set = self._normalize_data_set(self.training_set)
            
            #Shuffle
            keys = np.array(range(self.training_set.shape[0]))
            np.random.shuffle(keys)

            self.training_set = self.training_set[keys]
            self.labels = self.labels[keys]

            self.processed_labels = np.zeros((len(self.training_set), 10))
            for i in range(len(self.training_set)):
                self.processed_labels[i][self.labels[i]] = 1
        else:
            self.test_set = self._normalize_data_set(self.test_set)

    def load_weights(self,load_from_path=False):
        if load_from_path:
            parameters = None
            with open(self.path) as f:
                parameters = json.load(f)
            self.weights_hidden1 = np.array(parameters['weights_hidden1'])
            self.bias_h1 = np.array(parameters['bias_h1'])
        else:
            attributes = self.training_set.shape[1]
            output_labels = len(self.processed_labels[0])
            self.weights_hidden1 = np.random.rand(attributes, output_labels) * 0.01
            self.bias_h1 = np.random.randn(output_labels)

    def save_weights(self):
        data = {
            'weights_hidden1': self.weights_hidden1.tolist(),
            'bias_h1': self.bias_h1.tolist(),
        }

        with open(self.path, 'w') as json_file:
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
        self.load_weights()

        error_cost = 0

        for i in range(self.desired_epochs):

            for step in range(self.steps):
            
                x_batch = self.training_set[step * self.batch_size:(step + 1) * self.batch_size]
                y_batch = self.processed_labels[step * self.batch_size:(step + 1) * self.batch_size]

                prediction_o,prediction_h1,X1,_ = self.forward(x_batch)

                self.backprop(prediction_o,prediction_h1,x_batch,y_batch,X1)
                
                loss = np.sum(-y_batch * np.log(prediction_h1))
                error_cost = loss

            print(error_cost)

            print('Iterations: ' + str(i))
        self.save_weights()


    def forward(self,batch):
        X1 = np.dot(batch, self.weights_hidden1) + self.bias_h1
        prediction_h1 = self.sigmoid(X1)

        # y =np.dot(prediction_h1, self.weights_output) + self.bias_o
        # prediction_o = self.softmax(y)
        prediction_o = None
        y = None
        return prediction_o,prediction_h1,X1,y
    
    def backprop(self,prediction_o,prediction_h1,x_batch,y_batch,X1):
        
        error_cost_o = prediction_h1 - y_batch
        error_cost_h1 = np.dot(x_batch, self.weights_hidden1)

        dcost_bh1 = error_cost_h1 * self.sigmoid_der(X1)
        der_cost_h1 = np.dot(x_batch.T, self.sigmoid_der(X1) * error_cost_h1)

        #hidden
        self.weights_hidden1 -= self.learning_rate * der_cost_h1
        self.bias_h1 -= self.learning_rate * dcost_bh1.sum(axis=0)


    def predict(self, sample):
        output, prediction_h1,_,_ = self.forward(sample)
        label = np.argmax(prediction_h1)
        
        return label
        
    def testing(self):
        self.load_weights(load_from_path=True)
        self.load_data(train=False)
        self.data_preprocessing(train=False)
        error,correct,count = 0,0,0

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


if __name__ == "__main__":
    epochs=1250
    batch_size = 32
    amount_of_hidden_layer_neurons = 10
    data_amount = 10000
    path = 'training/params_adaline10neurons.json'
    size_img = (28,28)

    net = NN(data_amount = data_amount,hidden_neurons=amount_of_hidden_layer_neurons,batch_size=batch_size,epochs=epochs,path=path)
    # net.load_data()
    # net.data_preprocessing()
    # net.chunking()
    # net.load_weights()
    net.training()
    # net.testing()