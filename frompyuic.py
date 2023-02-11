from PyQt5 import QtWidgets
from ui import Ui_Dialog
import sys

class ApplicationWindow(QtWidgets.QDialog):
    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.buttons=[]
        self.buttons.append(getattr(self.ui,'checkBox'))
        
        for item in range(2,36):
            res= 'checkBox_'+str(item)
            self.buttons.append(getattr(self.ui,res))
        self.ui.pushButton.clicked.connect(self.render_number)
        
    def render_number(self):
        print('test')
        # for button in self.buttons:
            # button.setChecked(True)
            # button.
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
