from PyQt5 import QtWidgets
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtCore import QModelIndex
from ui import Ui_Dialog
import sys
from src.utils import *
# from src.DigitClassifier35 import DigitClassifier35
from src.DigitClassifier35_adaline import DigitClassifier35Adaline
import csv

class ApplicationWindow(QtWidgets.QDialog):
    def __init__(self,config):
        super(ApplicationWindow, self).__init__()
        self.path = config.path
        self.total = config.total
        self.data = None
        self.perceptrons = None
        self.current_number_pixels = [0]*config.total
        self.current_number_label = None
        self.label_to_save = None
        
        self.classifier = DigitClassifier35Adaline()
        self.classifier.load_perceptrons()
        
        data = read_csv_to_pd(self.path)
        data_x,data_y = prepare_data(data)
        for per in self.classifier.perceptrons:
            print(per.name)
            self.classifier.calculate_accuracy(data_x,data_y,per)
        self.classifier.test_digit_classifier(data_x,data_y)
        
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.init_buttons()        
        self.init_list()

        self.ui.predictButton.clicked.connect(self._on_clicked_pushButton)
        self.ui.listView.clicked[QModelIndex].connect(self._on_clicked_ListViewItem)
        self.ui.pushButton.clicked.connect(self._on_clicked_clear)
        self.ui.save_as_button.clicked.connect(self._on_clicked_save)
        self.ui.number_label_text.textChanged.connect(self._on_edit_new_label)

    def init_list(self):
        self.list_item_model = QStandardItemModel()
        self.ui.listView.setModel(self.list_item_model)
        self.load_datalist()
    
    def load_datalist(self):
        self.load_numbers()
        for index,item in enumerate(self.data):
            self.list_item_model.appendRow(QStandardItem(str(index) + '_' + str(item[0])))

    
    def init_buttons(self):
        self.buttons=[]
        self.buttons.append(getattr(self.ui,'checkBox'))
        for item in range(2,36):
            res= 'checkBox_'+str(item)
            button =  getattr(self.ui,res)
            self.buttons.append(button)
        for button in self.buttons:
            button.stateChanged.connect(self._on_clicked_matrix)

    def _on_edit_new_label(self):
        new_label = self.ui.number_label_text.text()
        self.label_to_save =   new_label
        self.current_number_label = new_label
    

    def _on_clicked_save(self):
        if self.label_to_save is None or len(self.label_to_save)!=1 or not self.label_to_save.isdigit():
            self.ui.number_label_text.setText('NO LABEL')
            return 

        with open(self.path,'a') as csv_data:
            writer = csv.writer(csv_data)
            writer.writerow([int(self.label_to_save)] + self.current_number_pixels)
        self.init_list()
        self.clear()
    
    
    def _on_clicked_clear(self):
        self.ui.predict_label.setText("")
        self.current_number_pixels = [0]*35
        self.current_number_label = None
        self.clear()

    def _on_clicked_pushButton(self):
        self.ui.predict_label.setText("")
        data = list_to_numpy(self.current_number_pixels,True)
        result = self.classifier.classify(data)
        self.ui.predict_label.setText(str(result))
    
    def _on_clicked_matrix(self,data):
        btnz = lambda x: 255 if x else 0
        list_of_clicked_btnz = [btnz(x.isChecked()) for x in self.buttons]
        self.current_number_pixels=  list_of_clicked_btnz
        data = list_to_numpy(self.current_number_pixels,True)
        result = self.classifier.classify(data)
        self.ui.predict_label.setText(str(result))

    def _on_clicked_ListViewItem(self):
        row_index = self.ui.listView.currentIndex().row()
        self.load_number_from_data(row_index)

    def load_numbers(self):
        self.data = read_csv_to_pd('result.csv',True)

    def load_number_from_data(self,i):
        number = self.data[i]
        self.clear()
        self.current_number_pixels = number[1:]
        self.current_number_label = number[0]
        self.draw_number_from_data()

    def draw_number_from_data(self):
        self.ui.currentNumberLabel.setText(str(self.current_number_label))
        for index,px in enumerate(self.current_number_pixels):
            if px == 255:
                self.buttons[index].setChecked(True)

    def clear(self):
        for index,data in enumerate(self.buttons):
            self.buttons[index].setChecked(True)
            self.buttons[index].setChecked(False)
        self.current_number_pixels=[0]*self.total
        z = lambda x: True if x>0 else False
        rez = len(list(filter(z,self.current_number_pixels)))
        btnz = lambda x: True if x else False
        list_of_clicked_btnz = [btnz(x.isChecked()) for x in self.buttons]
        cnt=0
        for btn in list_of_clicked_btnz:
            if btn:
                cnt+=1
        print(rez,cnt)

    def load_perceptrons():
        pass
    
def main():
    from config import Config
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow(Config)
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    
    main()
