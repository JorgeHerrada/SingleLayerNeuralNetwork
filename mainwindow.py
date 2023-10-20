from PyQt5.QtWidgets import QMainWindow
# importamos la clase que define la UI
from ui_mainwindow import Ui_MainWindow
from PyQt5.QtCore import pyqtSlot
from red import Red
from PyQt5 import QtGui
import numpy as np


class MainWindow(QMainWindow):
    entradas = []
    salidas = []
    testDataFile = 'testData.csv'

    def __init__(self):
        super(MainWindow, self).__init__()  # inicializa desde clase padre

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # self.ui.btnAgregar.clicked.connect(self.click_agregar)
        self.ui.btnClasificar.clicked.connect(self.click_clasificar)

        # Creamos red sin definir neuronas
        self.red = Red()

    @pyqtSlot()
    def click_clasificar(self):
        # limpiamos pesos, bias y plots
        self.red.clear()

        try:
            # validamos que existe el archivo
            print("Archivo entrada: {}".format(self.ui.txtEntrada.text()))
            print("Archivo salida: {}".format(self.ui.txtSalida.text()))
            # print("Archivo seleccionado: entradas1.csv")
            # leemos y definimos X y Y desde archivos
            X = self.readDataToMatrix(self.ui.txtEntrada.text())
            print("se cargó X")
            Y = self.readDataToMatrix(self.ui.txtSalida.text())
            print("se cargó Y")
        except ():
            print("ERROR con los archivos. Revisa que los archivos existen.")
            return

        # red aprende e imprime resultados
        # print("Pre entrenamiento: ",self.red.predict(X))
        self.red.fit(X, Y, self.ui)
        print("Post entrenamiento: \n", self.red.predict(X))
        print("Pesos W: \n", self.red.w)
        print("Y esperada: \n", Y)

        # limpiarplot
        # self.red.plotClear()

        # Use testData
        X_test = self.readDataToMatrix(self.testDataFile)
        Y_test = self.red.predict(X_test)

        # formateo de estimaciones para poder plotear
        # print("Estimaciones antes: \n", Y_test.T)
        Y_test = self.formatearEstimacionesTest(Y_test.T)  # Y_test transpuesta
        # print("Estimaciones despues: \n", Y_test)

        # plot data
        self.red.graficador.plotMatrix(X_test, Y_test)

        # save plot and update to UI
        self.red.guardarActualizar(self.ui)

    # lee los datos de una archivo y retorna una matriz forma m x n
    # m es el numero de patrones de entrenamiento
    # n es el numero de entradas

    def formatearEstimacionesTest(self, estimaciones):

        newEstimacion = []  # lista con las estimaciones formateadas

        # convertimos numpy array to list
        for array in estimaciones:
            newEstimacion.append(array.tolist())

        return newEstimacion

    def readDataToMatrix(self, dataFile):

        with open(dataFile, 'r') as file:
            lines = file.readlines()

        lines = self.limpiarSaltos(lines)  # limpiar saltos de linea

        # Creamos matriz
        matrix = np.zeros(shape=(len(lines), lines[0].count(",")+1))

        for i, line in enumerate(lines):
            for j, column in enumerate(line.split(",")):
                matrix[i, j] = float(column)

        return matrix.transpose()

    # Recibe una lista de lineas,
    # Retorna la lista de lineas sin los saltos de linea finales
    def limpiarSaltos(self, lineas):
        for i in range(len(lineas)):
            # si es igual a ascii de salto de linea, elimina
            if lineas[i][-1] == chr(10):
                lineas[i] = lineas[i][:-1]

        return lineas
