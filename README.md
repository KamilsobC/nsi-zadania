# nsi-zadania


## zad1 -  Perceptron Classifier z UI

 zad1.py - UI do klasyfikatora  
 ui.py - wyexportowane z Qtdesigner ui  
 src/Perceptron - klasa Perceptronu  
 src/DigitClassifier35.py - klasa klasyfikatora cyfr od 0 do 9  
 saved_data/percy[0-9].pickle - zapisane wytrenowane perceptrony   

 result.csv - dane cyfr od 0 do 9, 1 kolumna to label, reszta dane obrazu od 0 do 255 

 w tests.py  
 if __name__ == "__main__":  
       test_classifier()  
       test_perceptron()  
    
## zad3 -  Adaline Classifier z UI

 zad2.py - UI do klasyfikatora  
 ui.py - wyexportowane z Qtdesigner ui  
 src/Adaline - klasa Adaline'u  
 src/DigitClassifier35_adaline.py - klasa klasyfikatora cyfr od 0 do 9  
 saved_data/adaline[0-9].pickle - zapisane wytrenowane perceptrony   

 result.csv - dane cyfr od 0 do 9, 1 kolumna to label, reszta dane obrazu od 0 do 255  

 w tests.py
 if __name__ == "__main__":
     test_adaline()  
     test_classifier_adaline()  
     
## zad 7 - NN Mnist classifier z brackpropagation

zad7.py - klasa z Siecią Neuronową (784,N-ukrytych neuronów, 10)  
zad7.ipynb - możliwość sprawdzenia sieci na poszczególnym samplu  

data/data.json - dane treningowe  
data/params392.json - parametry dla sieci NN z 392 hidden neuronami jak w zad7.py  
