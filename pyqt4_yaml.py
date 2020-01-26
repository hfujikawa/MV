# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 19:18:34 2020
https://stackoverflow.com/questions/26005148/how-to-edit-a-yaml-file-with-pyqt4-gui
@author: hfuji
"""

import sys, yaml
from PyQt4 import QtGui, QtCore

class Example(QtGui.QWidget):

    def __init__(self):
        super(Example, self).__init__()
        self.verticalLayout = QtGui.QVBoxLayout()
        self.plainTextEdit = QtGui.QPlainTextEdit()
        self.verticalLayout.addWidget(self.plainTextEdit)
        self.pushButton = QtGui.QPushButton("Load Yaml")
        self.verticalLayout.addWidget(self.pushButton)
        self.setLayout(self.verticalLayout)
        self.pushButton.clicked.connect(self.loadYaml)

    def loadYaml(self):
        fileName = str(QtGui.QFileDialog.getOpenFileName(self, "Open File","/home/some/folder","Yaml(*.yaml);;AllFiles(*.*)"))
        f = open(fileName)
        getData = yaml.safe_load(f)
        prettyData = yaml.dump(getData, default_flow_style=False)
        self.plainTextEdit.appendPlainText(str(prettyData))

def main():
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
