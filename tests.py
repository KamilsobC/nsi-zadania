from src.Perceptron import Perceptron
from src.Adaline import Adaline

from src.utils import *
from src.DigitClassifier35 import DigitClassifier35 
from src.DigitClassifier35_adaline import DigitClassifier35Adaline

def test_perceptron():
    perceptron = Perceptron("0")
    create_csv_mnist_dataset()
    data = read_csv_to_pd('result.csv')
    desired_detections=[0,1,2,3,4,5,6,7,8,9]
    data_for_number = prepare_data_for_perceptron(data,0)
    x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
    weights, bias, errors = perceptron.train(x_train, y_train)   
    tp, tn, fp, fn = 0, 0, 0, 0
    for sample, label in zip(x_test, y_test):

        prediction = perceptron.predict(sample)
        if prediction ==1 and label ==1:
            tp+=1
        if prediction ==1 and label ==0:
            fn+=1
        if prediction ==0 and label ==0:
            tn+=1
        if prediction ==0 and label ==1:
            fn+=1
         
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    print("tp","tn","fp","fn")
    print(tp,tn,fp,fn)
    print(accuracy)
    for sample, label in zip(x_train, y_train):

        prediction = perceptron.predict(sample)
        if prediction ==1 and label ==1:
            tp+=1
        if prediction ==1 and label ==0:
            fn+=1
        if prediction ==0 and label ==0:
            tn+=1
        if prediction ==0 and label ==1:
            fn+=1
        
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    print("tp","tn","fp","fn")
    print(tp,tn,fp,fn)
    print(accuracy)

def test_adaline():
    perceptron = Adaline("0",epochs = 3000)
    create_csv_mnist_dataset()
    data = read_csv_to_pd('result.csv')
    desired_detections=[0,1,2,3,4,5,6,7,8,9]
    data_for_number = prepare_data_for_perceptron(data,0)
    x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
    weights, bias, errors = perceptron.train(x_train, y_train)
    cnt = 0
    max_error = 0.8
    
    # expx = np.exp(x)
    #     return expx / expx.sum(axis=index, keepdims=True)
    
    for data,labels in zip(x_test,y_test):
        test_data = list_to_numpy(data)
        test_label = list_to_numpy(labels)
        x = perceptron.predict(test_data)
        if x>max_error and test_label==1:
            cnt+=1
            print('good tp',x,test_label)
        if x>max_error and test_label==0:
            print('bad tf',x,test_label)
    
        if x<max_error and test_label==0:
            cnt+=1
            print('good tn',x,test_label)
        if x<max_error and test_label==1:
            print('bad fn',x,test_label)
    print(cnt/len(x_test))

def test_classifier():
    dc = DigitClassifier35()
    data = read_csv_to_pd('result.csv')
    dc.train_and_save(data)
    data_x,data_y = prepare_data(data)
    dc.load_perceptrons()
    for per in dc.perceptrons:
        print(per.name)
        dc.calculate_accuracy(data_x,data_y,per)

    dc.test_digit_classifier(data_x,data_y)
    # print('test')
    # data = read_csv_to_pd('result.csv')
    # data_for_number = prepare_data_for_perceptron(data,number)
    # x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
    # dc.calculate_accuracy(x_test,y_test,perceptron)
    # print(dc.perceptrons[0])
    # dc.calculate_accuracy(x_test,y_test,dc.perceptrons[0])

def test_classifier_adaline():
    dc = DigitClassifier35Adaline()
    data = read_csv_to_pd('result.csv')
    dc.train_and_save(data)
    data_x,data_y = prepare_data(data)
    dc.load_perceptrons()
    for per in dc.perceptrons:
        print(per.name)
        dc.calculate_accuracy(data_x,data_y,per)
       
   

    dc.test_digit_classifier(data_x,data_y)
    # print('test')
    # data = read_csv_to_pd('result.csv')
    # data_for_number = prepare_data_for_perceptron(data,number)
    # x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
    # dc.calculate_accuracy(x_test,y_test,perceptron)
    # print(dc.perceptrons[0])
    # dc.calculate_accuracy(x_test,y_test,dc.perceptrons[0])

if __name__ == "__main__":
    # test_adaline()
    test_classifier()
    # test_perceptron()
    # test_classifier_adaline()
