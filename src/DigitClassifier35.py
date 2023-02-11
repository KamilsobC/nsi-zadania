from src.Perceptron import Perceptron
from src.utils import *


class DigitClassifier35:

    def __init__(self,path_to_data='result.csv'):
        self.data = self.load_data(path_to_data)
        self.perceptrons = self.load_perceptrons() 
        if None in self.perceptrons:
            self.train_and_save(self.data)
            self.perceptrons = self.load_perceptrons()
        
    def train_and_save(self,data):
        for i in range(10):
            perceptron = Perceptron(str(i))
            data_for_number = prepare_data_for_perceptron(data,int(perceptron.name))
            x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
            perceptron.train(x_train, y_train)   
            save_pickle(perceptron,'saved_data/perc'+perceptron.name+'.pickle')
    
    def calculate_accuracy(x_test, y_test,perceptron):
        tp, tn, fp, fn = 0, 0, 0, 0

        for sample, label in zip(x_test, y_test):

            prediction = perceptron.predict(sample)

            if prediction == label:
                if prediction == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if prediction == 1:
                    fp += 1
                else:
                    fn += 1

        accuracy = (tp + tn)/(tp + tn + fp + fn)
        print(tp,tn,fp,fn)
        return accuracy

    def load_data(self,path_to_data='result.csv'):
        return read_csv_to_pd(path_to_data)
    
    def load_perceptrons(self):
        perceptrons = []

        for i in range(10):
            perceptron =load_pickle('saved_data/perc'+ str(i) +'.pickle')
            perceptrons.append(perceptron)
        
        return perceptrons

    def classify(self):
        print('tej')

if __name__ == "__main__":
    dc = DigitClassifier35()
    dc.classify()