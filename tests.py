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
    eg_input =  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    eg_label = [0]
    test_data=list_to_numpy(eg_input)
    test_label=list_to_numpy(eg_label)
    x = perceptron.predict(test_data)
    print(x)

def test_classifier():
    dc = DigitClassifier35()
    data = read_csv_to_pd('result.csv')
    for i in range(10):
        number = i 
        data_for_number = prepare_data_for_perceptron(data,number)
        x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
        perceptron = Perceptron(number)
        perceptron.train(x_train,y_train)
        print(i)
        dc.calculate_accuracy(x_test,y_test,perceptron)
        dc.perceptrons.append(perceptron)
    
    data_x,data_y = prepare_data(data)
    for perceptron in dc.perceptrons:
        print(perceptron.name)
        tp, tn, fp, fn =  0, 0, 0, 0
        for sample, label in zip(data_x, data_y):

            prediction = perceptron.predict(sample)
            
            if perceptron.name == label:
                if prediction == 1:
                    tp+=1
                else:
                    fp+=1
            else:
                if prediction == 0:
                    tn+=1
                else:
                    fn+=1

        accuracy = (tp + tn)/(tp + tn + fp + fn)
        print(tp,tn,fp,fn)
        print(accuracy)


    print('test')
    # data = read_csv_to_pd('result.csv')
    # data_for_number = prepare_data_for_perceptron(data,number)
    # x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
    # dc.calculate_accuracy(x_test,y_test,perceptron)
    # print(dc.perceptrons[0])
    # dc.calculate_accuracy(x_test,y_test,dc.perceptrons[0])
    
if __name__ == "__main__":
    # test_perceptron()
    test_classifier()