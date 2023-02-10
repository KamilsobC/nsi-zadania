import pandas as pd
from PIL import Image
import os 
import numpy as np

def bmp_to_array(path,number):
    im = Image.open(path)
    p = np.array(im)
    x = p.ravel()
    y = x[::3]
    z = y<1
    zz = z.astype(int)*255
    zz = np.insert(zz, 0, number)
    return zz

def create_csv_mnist_dataset():
    # columns = ['label','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35']
    csv_array=[]
    csv_array.append(np.zeros(36))
    path_to_numbers = 'numbers'
    numbers_folders = os.listdir(path_to_numbers)

    for folder in numbers_folders:
        numbers_bmps = os.listdir(path_to_numbers+'/'+folder)
        for number_bmp in numbers_bmps:
            if '.bmp' in number_bmp:
                path=path_to_numbers+'/'+folder+'/'+number_bmp
                csv_array.append(bmp_to_array(path,int(folder)).astype(int))
    arr = np.array(csv_array)
    with open("result.csv", "wb") as f:
        np.savetxt(f, arr.astype(int), fmt='%s', delimiter=",")

def create_train_test_sets(d):
    d = d.sample(frac=1)
    percent_25=(int(round(len(d.index))/4))
    test = d.head(percent_25)
    train = d.head(-percent_25)
    x_train = train.drop('label', axis=1).to_numpy()/255
    x_test = test.drop('label', axis=1).to_numpy()/255
    y_train = train['label'].to_numpy()
    y_test = test['label'].to_numpy()
    return x_train,x_test,y_train,y_test

def read_csv_to_pd(path):
    columns = ['label','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35']
    data = pd.read_csv(path)
    data.columns = columns
    return data

def prepare_data_for_perceptron(data,number):
    data_desired = data[data['label'] == number].copy() 
    data_desired.loc[data_desired['label'] >=0, 'label'] = 1
    data_rest = data[data['label'] != number].copy()
    data_rest.loc[data_rest['label'] >=0, 'label'] = 0
    return pd.concat([data_desired, data_rest])

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