from src.Perceptron import Perceptron
from src.Adaline import Adaline
from src.utils import *


class DigitClassifier35Adaline:

    def __init__(self,path_to_data='result.csv'):
        self.perceptrons = []
        pass

        
    def train_and_save(self,data):
        for i in range(10):
            perceptron = Adaline(str(i))
            data_for_number = prepare_data_for_perceptron(data,int(perceptron.name))
            x_train,x_test,y_train,y_test =  create_train_test_sets(data_for_number)
            perceptron.train(x_train, y_train)   
            save_pickle(perceptron,'saved_data/adaline'+str(perceptron.name)+'.pickle')
    
    def calculate_accuracy(self,x_test, y_test,perceptron):
        best_fit=0.8
        tp, tn, fp, fn = 0, 0, 0, 0
        for sample, label in zip(x_test, y_test):
            prediction = perceptron.predict(sample)
            if prediction>best_fit:
                if str(perceptron.name) == str(label):
                    # print('good_tp',per.name,label)
                    tp+=1
                else:
                    # print('BAD_fp',per.name,label,prediction)
                    fp+=1
            if prediction<best_fit:
                if str(perceptron.name) != str(label):
                    # print('good_tn',per.name,label)
                    tn+=1
                else:
                    # print('BAD_fn',per.name,label,prediction)
                    fn+=1
        accuracy = (tp + tn)/(tp + tn + fp + fn)
        print(accuracy)

        return accuracy
    
    def test_digit_classifier(self,x_test, y_test):
        tp,tf=0,0
        cnt=[0]*10
        cnt_false=[0]*10
        for sample, label in zip(x_test, y_test):
            results = self.classify(sample,return_all=True)
            if label == int(results[0][0]):
                tp+=1
                # print("label",label,"sucess:",results[0][0])
                cnt[label]+=1
            else:
                tf+=1
                # print("label",label,"failure:",results[0][0])
                cnt_false[label]+=1
        
        for index,(good,bad) in enumerate(zip(cnt,cnt_false)):
            print(index,"adaline: +",good,"-",bad,good/(good+bad))
        print(tp/(tp+tf))

    def load_data(self,path_to_data='result.csv'):
        return read_csv_to_pd(path_to_data)
    
    def load_perceptrons(self):
        perceptrons = []

        for i in range(10):
            perceptron =load_pickle('saved_data/adaline'+ str(i) +'.pickle')
            perceptrons.append(perceptron)
        
        self.perceptrons=perceptrons

    def classify(self,data,return_all=False):
        results = []
        for perceptron in self.perceptrons:
            prediction = perceptron.predict(data)                
            results.append((perceptron.name,prediction))
        result = sorted(results,key=lambda res:res[1])
        result.reverse()
        if return_all:
            return result    
        return result[0][0]

if __name__ == "__main__":
    dc = DigitClassifier35()
    