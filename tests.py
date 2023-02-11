from src.Perceptron import Perceptron
from src.utils import *
from src.DigitClassifier35 import DigitClassifier35 
def test_perceptron():
    perceptron = Perceptron("0")
    create_csv_mnist_dataset()
    data = read_csv_to_pd('result.csv')
    desired_detections=[0,1,2,3,4,5,6,7,8,9]
    data_for_number = prepare_data_for_perceptron(data,0)
    x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
    weights, bias, errors = perceptron.train(x_train, y_train)   


def test_classifier():
    dc = DigitClassifier35()
    print('hej')
    
if __name__ == "__main__":
    test_perceptron()
    test_classifier()