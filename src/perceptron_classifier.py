from perceptron import Perceptron
from utils import *

def train_and_save():
    for i in range(10):
        print(i)
        perceptron = Perceptron(str(i))
        print(perceptron.name)
        data = read_csv_to_pd('result.csv')
        data_for_number = prepare_data_for_perceptron(data,int(perceptron.name))
        x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
        weights, bias, errors = perceptron.train(x_train, y_train)   
        acc = calculate_accuracy(x_test, y_test, perceptron)
        print('Accuracy: ', acc)
        save_pickle(perceptron,'saved_data/perc'+perceptron.name+'.pickle')


def load_perceptrons():
    perceptrons = []

    for i in range(10):
        print(i)
        perceptron =load_pickle('saved_data/perc'+ str(i) +'.pickle')
        perceptrons.append(perceptron)
    
    for item in perceptrons:
        print(item)
        print(item.weights)


if __name__ == "__main__":
    load_perceptrons()