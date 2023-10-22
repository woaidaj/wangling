from CallMainWindow import *

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    para = ParameterWindow()
    fI = FocusedImageWindow()
    win.show()
    exit(app.exec_())
