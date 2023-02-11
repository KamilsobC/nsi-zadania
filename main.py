from PyQt5 import QtWidgets
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtCore import QModelIndex
from ui import Ui_Dialog
import sys
from src.utils import *
from src.DigitClassifier35 import DigitClassifier35
class ApplicationWindow(QtWidgets.QDialog):
    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.data = None
        self.perceptrons = None
        self.current_number_pixels = None
        self.current_number_label = None
        self.classifier = DigitClassifier35()
        self.classifier.load_perceptrons()

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.init_buttons()        
        self.init_list()

        self.ui.predictButton.clicked.connect(self._on_clicked_pushButton)
        self.ui.listView.clicked[QModelIndex].connect(self._on_clicked_ListViewItem)



    def init_list(self):
        self.list_item_model = QStandardItemModel()
        self.ui.listView.setModel(self.list_item_model)
        self.load_numbers()
        for index,item in enumerate(self.data):
            self.list_item_model.appendRow(QStandardItem(str(index) + '_' + str(item[0])))
    

    def init_buttons(self):
        self.buttons=[]
        self.buttons.append(getattr(self.ui,'checkBox'))
        for item in range(2,36):
            res= 'checkBox_'+str(item)
            self.buttons.append(getattr(self.ui,res))


    def _on_clicked_pushButton(self):
        data = list_to_numpy(self.current_number_pixels,True)
        result = self.classifier.classify(data,self.current_number_label)
        self.ui.predict_label.setText(str(result))

    
    def _on_clicked_ListViewItem(self):
        row_index = self.ui.listView.currentIndex().row()
        self.load_number_from_data(row_index)

    def load_numbers(self):
        self.data = read_csv_to_pd('result.csv',True)

    def load_number_from_data(self,i):
        number = self.data[i]
        self.current_number_pixels = number[1:]
        self.current_number_label = number[0]
        self.draw_number_from_data()

    def draw_number_from_data(self):
        self.clear()
        self.ui.currentNumberLabel.setText(str(self.current_number_label))
        for index,px in enumerate(self.current_number_pixels):
            if px == 255:
                self.buttons[index].setChecked(True)

    def clear(self):
        for button in self.buttons:
            button.setChecked(False)

    def load_perceptrons():
        pass
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
