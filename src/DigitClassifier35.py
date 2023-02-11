from src.Perceptron import Perceptron
from src.utils import *


class DigitClassifier35:

    def __init__(self,path_to_data='result.csv'):
        self.perceptrons = []
        pass

        
    def train_and_save(self,data):
        for i in range(10):
            perceptron = Perceptron(str(i))
            data_for_number = prepare_data_for_perceptron(data,int(perceptron.name))
            x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
            perceptron.train(x_train, y_train)   
            save_pickle(perceptron,'saved_data/perc'+str(perceptron.name)+'.pickle')
    
    def calculate_accuracy(self,x_test, y_test,perceptron):
        tp, tn, fp, fn = 0, 0, 0, 0

        for sample, label in zip(x_test, y_test):

            prediction = perceptron.predict(sample)
            if prediction ==1:
                if int(perceptron.name) == label:
                    tp+=1
                else:
                    fp+=1
            else: 
                if int(perceptron.name) != label:
                    tn+=1
                else:
                    fn+=1
            
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        print("tp","tn","fp","fn")
        print(tp,tn,fp,fn)
        print(accuracy)
        print()

        return accuracy

    def load_data(self,path_to_data='result.csv'):
        return read_csv_to_pd(path_to_data)
    
    def load_perceptrons(self):
        perceptrons = []

        for i in range(10):
            perceptron =load_pickle('saved_data/perc'+ str(i) +'.pickle')
            perceptrons.append(perceptron)
        
        self.perceptrons=perceptrons

    def classify(self,data,label):
        results = []
        for perceptron in self.perceptrons:
            prediction = perceptron.predict(data)
            if prediction == 1:
                results.append(perceptron.name)
        return results

if __name__ == "__main__":
    dc = DigitClassifier35()
    