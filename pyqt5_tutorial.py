# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 09:02:43 2020
https://www.sejuku.net/blog/75467
@author: hfuji
"""

import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sip
 
class MainWindow(QWidget):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
 
        # self.test = QCheckBox('test', self)
        # 一つ目のチェックボックス
        self.upper = QCheckBox('Upper', self)
        self.upper.move(100, 30)
     
        # 二つ目のチェックボックス
        self.lower = QCheckBox('Lower', self)
        self.lower.move(180, 30)
     
        # グループ化
        self.group = QButtonGroup()
        self.group.addButton(self.upper,1)
        self.group.addButton(self.lower,2)
        
        self.setGeometry(300, 50, 400, 350)
        self.setWindowTitle('QCheckBox')
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())