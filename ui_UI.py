# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\toluc\OneDrive\Documentos\Jorge\INCO\10mo\SEM Inteligencia Artificial  2 - Julio\Practica_4_Red_Unicapa\UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1115, 692)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.btnClasificar = QtWidgets.QPushButton(self.centralwidget)
        self.btnClasificar.setGeometry(QtCore.QRect(780, 480, 281, 71))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.btnClasificar.setFont(font)
        self.btnClasificar.setObjectName("btnClasificar")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 691, 571))
        self.label.setTabletTracking(False)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("c:\\Users\\toluc\\OneDrive\\Documentos\\Jorge\\INCO\\10mo\\SEM Inteligencia Artificial  2 - Julio\\Practica_4_Red_Unicapa\\../prueba.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.lblNombre = QtWidgets.QLabel(self.centralwidget)
        self.lblNombre.setGeometry(QtCore.QRect(780, 580, 281, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblNombre.setFont(font)
        self.lblNombre.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lblNombre.setAlignment(QtCore.Qt.AlignCenter)
        self.lblNombre.setObjectName("lblNombre")
        self.txtArchivo = QtWidgets.QLineEdit(self.centralwidget)
        self.txtArchivo.setGeometry(QtCore.QRect(780, 400, 281, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.txtArchivo.setFont(font)
        self.txtArchivo.setAlignment(QtCore.Qt.AlignCenter)
        self.txtArchivo.setObjectName("txtArchivo")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1115, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Red Neuronal Unicapa"))
        self.btnClasificar.setText(_translate("MainWindow", "Clasificar"))
        self.lblNombre.setText(_translate("MainWindow", "Herrada Serrano Jorge Luis"))
        self.txtArchivo.setText(_translate("MainWindow", "Sin archivo seleccionado"))
