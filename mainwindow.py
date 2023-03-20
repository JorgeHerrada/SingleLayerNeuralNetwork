from PyQt5.QtWidgets import QMainWindow
from ui_mainwindow import Ui_MainWindow     # importamos la clase que define la UI
from PyQt5.QtCore import pyqtSlot
from perceptron import Perceptron
from PyQt5 import QtGui
import numpy as np
import csv

class MainWindow(QMainWindow):
    entradas = []
    salidas = []

    def __init__(self):
        super(MainWindow,self).__init__() # inicializa desde clase padre

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # self.ui.btnAgregar.clicked.connect(self.click_agregar)
        self.ui.btnClasificar.clicked.connect(self.click_clasificar)

        # Creamos neurona, entrada y learning rate
        self.neuron = Perceptron()


    def agregar(self):
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

        # validamos que existe el archivo
        print("Archivo seleccionado: {}".format(self.ui.txtArchivo.text()))
        # leemos y definimos X y Y desde archivos
        X,Y = self.leerEntradas()

        # # Creamos neurona, entrada y learning rate
        # neuron = Perceptron(2, 0.1) 

        # neurona aprende e imprime resultados
        print("Pre entrenamiento: ",self.neuron.predict(X))
        self.neuron.fit(X, Y, self.ui)
        print("Post entrenamiento: ",self.neuron.predict(X))

        # limpiamos
        self.entradas = []
        self.salidas = []
        

    def leerEntradas(self):
        
        with open("entradas.csv", 'r') as file:
            lines = file.readlines()
        
        lines = self.limpiarSaltos(lines) # limpiar saltos de linea
        print(lines)

        # Creamos matriz de entradas de n filas y m columnas 
        # n es el numero de combinaciones 
        # m es el numero de entradas
        X = np.zeros(shape=(len(lines),lines[0].count(",")+1))

        for i,line in enumerate(lines):
            for j,column in enumerate(line.split(",")):
                X[i,j] = float(column)

        print("X: ", X.transpose())

        with open("salidas.csv", 'r') as file:
            lines = file.readlines()

        lines = self.limpiarSaltos(lines)
        # print(lines[0].split(","))
        print(lines)
        Y = np.zeros(shape=(len(lines),lines[0].count(",")+1))

        for i,line in enumerate(lines):
            for j,column in enumerate(line.split(",")):
                # print("estados: ",i,j)
                Y[i,j] = float(column)
        
        Y = Y.transpose()

        # Matriz de salidas deseadas (1 por cada par de entradas)
        print("Y: ", Y)


        return X.transpose(),Y[0]
 

    # Recibe una lista de lineas,
    # Retorna la lista de lineas sin los saltos de linea finales
    def limpiarSaltos(self, lineas):
        for i in range(len(lineas)):
            # si es igual a ascii de salto de linea, elimina
            if lineas[i][-1] == chr(10):
                lineas[i] = lineas[i][:-1]

        return lineas