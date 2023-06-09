from PyQt5.QtWidgets import QMainWindow
from ui_mainwindow import Ui_MainWindow     # importamos la clase que define la UI
from PyQt5.QtCore import pyqtSlot
from red import Red
from PyQt5 import QtGui
import numpy as np

class MainWindow(QMainWindow):
    entradas = []
    salidas = []

    def __init__(self):
        super(MainWindow,self).__init__() # inicializa desde clase padre

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # self.ui.btnAgregar.clicked.connect(self.click_agregar)
        self.ui.btnClasificar.clicked.connect(self.click_clasificar)

        # Creamos red sin definir neuronas 
        self.red = Red()


    @pyqtSlot()
    def click_clasificar(self):
        #limpiamos pesos, bias y plots
        self.red.clear()

        try:
            # validamos que existe el archivo
            print("Archivo entrada: {}".format(self.ui.txtEntrada.text()))
            print("Archivo salida: {}".format(self.ui.txtSalida.text()))
            # print("Archivo seleccionado: entradas1.csv")
            # leemos y definimos X y Y desde archivos
            X,Y = self.leerEntradasSalidas(self.ui.txtEntrada.text(),self.ui.txtSalida.text())
            # X,Y = self.leerEntradasSalidas("entradas1.csv") # PRUEBAS
        except:
            print("ERROR con los archivos. Revisa que los archivos existen.")
            return

        # red aprende e imprime resultados
        # print("Pre entrenamiento: ",self.red.predict(X))
        self.red.fit(X, Y, self.ui)
        print("Post entrenamiento: \n",self.red.predict(X))
        print("Pesos W: \n",self.red.w)
        print("Y esperada: \n",Y)

        # plottear linea
        for i in range(len(self.red.w)):
            self.red.graficador.drawDivision([self.red.punto(self.red.w[i,0], self.red.w[i,1], -self.red.b[0,i], -5),
                                        self.red.punto(self.red.w[i,0], self.red.w[i,1], -self.red.b[0,i], 5)],)


        # limpiamos
        # self.entradas = []
        # self.salidas = []
        

    def leerEntradasSalidas(self,entrada,salida):
        
        with open(entrada, 'r') as file:
            lines = file.readlines()
        
        lines = self.limpiarSaltos(lines) # limpiar saltos de linea
        # print("Lineas X: ",lines)

        # Creamos matriz de entradas de n filas y m columnas 
        # n es el numero de combinaciones 
        # m es el numero de entradas
        X = np.zeros(shape=(len(lines),lines[0].count(",")+1))

        for i,line in enumerate(lines):
            for j,column in enumerate(line.split(",")):
                X[i,j] = float(column)

        print("X: \n", X.transpose())

        # nombreArchivoSalidas = "salidas1.csv"

        with open(salida, 'r') as file:
            lines = file.readlines()

        lines = self.limpiarSaltos(lines)
        # print(lines[0].split(","))
        # print(lines)
        Y = np.zeros(shape=(len(lines),lines[0].count(",")+1))

        for i,line in enumerate(lines):
            for j,column in enumerate(line.split(",")):
                # print("estados: ",i,j)
                Y[i,j] = float(column)
        
        Y = Y.transpose()

        # Matriz de salidas deseadas (1 por cada par de entradas)
        print("Y: \n", Y)

        return X.transpose(),Y

    # Recibe una lista de lineas,
    # Retorna la lista de lineas sin los saltos de linea finales
    def limpiarSaltos(self, lineas):
        for i in range(len(lineas)):
            # si es igual a ascii de salto de linea, elimina
            if lineas[i][-1] == chr(10):
                lineas[i] = lineas[i][:-1]

        return lineas