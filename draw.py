# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QPainter, QColor

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(30,30,600,400)
        self.pos1 = [0,0]
        self.pos2 = [0,0]
        self.show()

    def paintEvent(self, event):
        width = self.pos2[0]-self.pos1[0]
        height = self.pos2[1] - self.pos1[1]     

        qp = QPainter()
        qp.begin(self)
        # qp.setPen(QColor(168, 34, 3))
        qp.drawRect(self.pos1[0], self.pos1[1], width, height)
        qp.fillRect(self.pos1[0], self.pos1[1], width, height, QColor(168, 34, 3))     
        qp.end()

    def mousePressEvent(self, event):
        self.pos1[0], self.pos1[1] = event.pos().x(), event.pos().y()
        print(event)
        print("clicked")

    def mouseReleaseEvent(self, event):
        self.pos2[0], self.pos2[1] = event.pos().x(), event.pos().y()
        print("released")
        self.update()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyWidget()
    window.show()
    app.aboutToQuit.connect(app.deleteLater)
    sys.exit(app.exec_())