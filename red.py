import numpy as np
from PyQt5 import QtTest
import matplotlib.pyplot as plt
from graficador import Graficafor
from PyQt5 import QtGui


class Red:
    # defininmos limite de de epocas para evitar
    # bucles infinitos en caso de que los puntos
    # insertados no sean linealmente separables
    EPOCH = 50
    LEARN_RATE = 0.1  # learning rate
    N_INPUT = 2  # dimension de los patrones de entrenamiento

    contEpoch = 0  # contador de epocas

    # Constructor toma numero de inputs y learning rate
    def __init__(self, n_input=N_INPUT, learning_rate=LEARN_RATE):

        # # inicializamos los pesos "w" con un vector de
        # # dimension "n_input" con rango [-1,1] random
        # self.w = -1 + (1 - (-1)) * np.random.rand(n_input)
        # # bias random con rango [-1,1]
        # self.b = -1 + (1 - (-1)) * np.random.rand()
        # self.eta = learning_rate
        self.clear(n_input, learning_rate)
        self.graficador = Graficafor()

    # funcion de activacion escal
    # recibe una matriz de cualquier dimension y
    # retorna una igual con [0,1]
    def f_activacion(self, nums):
        salida = np.zeros(shape=nums.shape)
        for i in range(nums.shape[0]):
            for j in range(nums.shape[1]):
                if nums[i, j] >= 0:
                    salida[i, j] = 1
                else:
                    salida[i, j] = 0

        return salida

    # Entrega un vector de salidas, dada una matriz de
    # entradas para la neurona
    def predict(self, X):

        # nPatrones = # de colmunas en la matriz X (candidad de patrones de entrenamiento)
        nPatrones = X.shape[1]
        # print("nPatrones: \n{}\nX: \n{}".format(nPatrones, X))

        # neuronas = cantidad de neuronas
        nNeuronas = self.b.shape[1]
        # print("nNeuronas: \n{}\nBias: \n{}".format(nNeuronas, self.b))

        # y_est guardará la salidas, se inicializa
        # como matriz de numNeuronas * nPatrones en 0s
        y_est = np.zeros(shape=(nNeuronas, nPatrones))
        # print("y_est inicializado: \n{}".format(y_est))

        # Producto punto de la matriz de pesos y matriz de entradas (w * X) + bias
        # dot() realiza el producto punto de cada fila en W contra toda la matriz X (nNeuronas veces)
        # Acomoda cada resultado en una matriz de nNeuronas * nPatrones
        # el bias se transpone para ser vector columna y sumarse a la fila correspondiente de la matriz y_est
        y_est = np.dot(self.w, X) + self.b.transpose()
        # print("Producto punto: \n", y_est)

        # mandamos salida estimada a funcion de activación
        y_est = self.f_activacion(y_est)
        # print("Funcion de activacion, y_est: \n", y_est)

        # retornamos vector con las salidas binarias
        return y_est

    # Realiza aprendizaje en epocas, actualiza
    # puntos y division en grafica
    def fit(self, X, Y, ui, epoch=EPOCH):

        # **************** Inicializamos pesos y bias *****************

        # creamos matriz de pesos W. 1 fila por neurona y 1 columna por cada entrada
        self.w = (-1 + (1 - (-1)) *
                  np.random.rand(X.shape[0], Y.shape[0])).transpose()
        print("Pesos W:\n", self.w)

        # creamos vector de bias b. 1 elemento por cada neurona
        self.b = np.random.rand(1, Y.shape[0])
        print("Bias: ", self.b)

        # numero de lineas a plottear
        nLineas = self.b.shape[1]

        # nPatrones = numero de conjuntos de entrada (patrones)
        nPatrones = X.shape[1]

        # lista para comparar estimaciones con resultados esperados
        estimaciones = []

        self.contEpoch = 0

        # iteramos por cada epoca
        for _ in range(epoch):
            # resetea estimaciones y aumenta contador
            estimaciones = []
            self.contEpoch += 1

            # iteramos por cada patron de entrenamiento
            for i in range(nPatrones):
                print("Epoca: {}.{}".format(self.contEpoch, i))

                # calculamos salida dado el patron actual
                # reshape para asegurar que tenemos vector columna porque ese slicing retorna vector regular
                y_est = self.predict(X[:, i].reshape(-1, 1))
                print("y_est: ", y_est)
                estimaciones.append(y_est.transpose())  # guardamos estimacion

                # actualizacion de peso y bias basado en el error
                print("W antes: \n{} \nBias antes: {}".format(self.w, self.b))
                self.w = self.w + self.eta * \
                    (Y[:, i].reshape(-1, 1) - y_est) * X[:, i]
                self.b = self.b + self.eta * \
                    (Y[:, i].reshape(-1, 1) - y_est).transpose()
                print("W despues: \n{} \nBias despues: {}".format(self.w, self.b))
                # checar si es necesario hacer reshape del X[:,i]

            # limpiar plot
            self.plotClear()

            # formatear estimaciones
            print("Estimaciones antes: \n", estimaciones)
            estimaciones = self.formatearEstimaciones(estimaciones)
            print("Estimaciones despues: \n", estimaciones)

            # plottear puntos a color
            self.graficador.plotMatrix(X, estimaciones)

            # plottear linea
            for i in range(len(self.w)):
                self.graficador.drawDivision([self.punto(self.w[i, 0], self.w[i, 1], -self.b[0, i], -5),
                                              self.punto(self.w[i, 0], self.w[i, 1], -self.b[0, i], 5)],)

            # actualizar
            self.guardarActualizar(ui)

            # se logró el objetivo?
            if self.aprendizajeTerminado(Y, estimaciones):
                print("X: \n{} \nY: \n{} \n Estimaciones: \n{}".format(
                    X, Y, estimaciones))
                break

            # retraso para visualizar
            QtTest.QTest.qWait(100)

        print("Epocas: ", self.contEpoch)

    # ¿todas las estimaciones son iguales a las salidas esperadas?
    def aprendizajeTerminado(self, Y, estimaciones):
        print("Y: ", Y)
        print("estimaciones: ", estimaciones)

        for i in range(len(estimaciones)):
            # print("Y.transpuesta()[i].tolist(): \n",Y.transpose()[i].tolist())
            # print("Estimacion[i]: \n",estimaciones[i])
            if Y.transpose()[i].tolist() != estimaciones[i]:
                return False
        print("*******************Se alcanzó el objetivo ALVVV!************************")
        return True

    # calcular punto para pendiente
    def punto(self, w1, w2, teta, x):
        if w2 == 0:
            print("No se puede dividir entre cero, cambia el valor de W2")
            return

        m = -1*(w1/w2)
        c = teta/w2
        y = (m*x) + c
        return y

    # calcular pendiente de recta que separa 1s y 0s
    def calcularPendiente(self, columna):
        # linea para dividir
        limLinea = [-5, 5]
        p1 = self.punto(self.w[0, columna],
                        self.w[1, columna], -self.b, limLinea[0]),
        p2 = self.punto(self.w[0, columna],
                        self.w[1, columna], -self.b, limLinea[0]),

        return p1, p2

    # save fig and update UI
    def guardarActualizar(self, ui):
        plt.savefig("prueba.png")
        ui.label.setPixmap(QtGui.QPixmap("prueba.png"))

    # retorna lista de estimaciones formateada
    def formatearEstimaciones(self, estimaciones):

        newEstimacion = []  # lista con las estimaciones formateadas

        # convertimos numpy array to list
        for array in estimaciones:
            newEstimacion.append(array.tolist()[0])

        return newEstimacion

    # limpia SOLAMENTE la interfaz no modifica pesos, etc
    def plotClear(self):
        plt.cla()

    # limpia puntos viejos y reinicia pesos y bias
    def clear(self, n_input=2, learning_rate=0.1):
        # plt.clf()
        plt.cla()

        # inicializamos los pesos "w" con un vector de
        # dimension "n_input" con rango [-1,1] random
        self.w = -1 + (1 - (-1)) * np.random.rand(n_input)
        # bias random con rango [-1,1]
        self.b = -1 + (1 - (-1)) * np.random.rand()
        self.eta = learning_rate
