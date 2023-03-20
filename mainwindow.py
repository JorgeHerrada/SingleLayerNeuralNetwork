from PyQt5.QtWidgets import QMainWindow
from ui_mainwindow import Ui_MainWindow     # importamos la clase que define la UI
from PyQt5.QtCore import pyqtSlot
from perceptron import Perceptron
from PyQt5 import QtGui
import numpy as np

class MainWindow(QMainWindow):
    entradas = []
    salidas = []

    def __init__(self):
        super(MainWindow,self).__init__() # inicializa desde clase padre

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.btnAgregar.clicked.connect(self.click_agregar)
        self.ui.btnClasificar.clicked.connect(self.click_clasificar)

        # Creamos neurona, entrada y learning rate
        self.neuron = Perceptron()


    @pyqtSlot()
    def click_agregar(self):
        try:
            # Primer punto? limpiamos plot
            if len(self.entradas) == 0:
                self.neuron.clear()

            # hay texto en las cajas?
            if self.ui.txtX1.text() != "" and self.ui.txtX2.text() != "":
                # guardamos guardamos entradas (x1,x2) y salidas (y)
                self.entradas.append([float(self.ui.txtX1.text()),float(self.ui.txtX2.text())])
                print("entradas: ",self.entradas)

                # Ploteamos punto agregado en coordenada y color correspondiente
                if self.ui.checkBox.checkState():
                    # agregamos 1 a la lista de salidas deseadas 
                    self.salidas.append(1)
                    
                    # PLOTTEAR
                    self.neuron.graficador.setPunto(self.entradas[-1][0],self.entradas[-1][1],1)
                else:
                    # agregamos 0 a la lista de salidas deseadas 
                    self.salidas.append(0)

                    # PLOTTEAR
                    self.neuron.graficador.setPunto(self.entradas[-1][0],self.entradas[-1][1],0)
                # print("Salidas: ", self.salidas)
            else:
                print("Entrada invalida")
            
            # actualizar UI
            # print("apunto de guardarActualizar")
            self.neuron.guardarActualizar(self.ui)

            # limpiar 
            self.ui.txtX1.setText("")
            self.ui.txtX2.setText("")

        except ValueError:
            print("¡Error en la entrada! - ",ValueError)

    @pyqtSlot()
    def click_clasificar(self):
        # # Creamos neurona, entrada y learning rate
        # neuron = Perceptron(2, 0.1) 

        # Creamos matriz de entradas de 2 filas y n columnas
        # n es el numero puntos agregados
        X = np.zeros(shape=(2,len(self.salidas)))
        # print("X: ", X)
        for i in range(len(self.entradas[0])):
            for j in range(len(self.salidas)):
                X[i,j] = self.entradas[j][i]
        print("X: ", X)

        # Matriz de salidas deseadas (1 por cada par de entradas)
        Y = np.array(self.salidas)
        print("Y: ", Y)

        # neurona aprende e imprime resultados
        print("Pre entrenamiento: ",self.neuron.predict(X))
        self.neuron.fit(X, Y, self.ui)
        print("Post entrenamiento: ",self.neuron.predict(X))

        # limpiamos
        self.entradas = []
        self.salidas = []
        

        
 