import matplotlib.pyplot as plt
import numpy as np

class Graficafor:
    # Select length of axes and the space between tick labels
    xmin, xmax, ymin, ymax = -3, 3, -3, 3
    ticks_frequency = 1

    # Plot points
    fig, ax = plt.subplots(figsize=(10, 10))    

    # rango de linea divisoria
    rangoLinea = [-5, 5]

    # Set identical scales for both axes
    ax.set(xlim=(xmin-1, xmax+1), ylim=(ymin-1, ymax+1), aspect='equal')

    # Set bottom and left spines as x and y axes of coordinate system
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create 'x' and 'y' labels placed at the end of the axes
    ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

    # Create custom major ticks to determine position of tick labels
    x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])

    # Create minor ticks placed at each integer to enable drawing of minor grid
    # lines: note that this has no effect in this example with ticks_frequency=1
    ax.set_xticks(np.arange(xmin, xmax+1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax+1), minor=True)

    # Draw major and minor grid lines
    ax.grid(which='both', color='grey',
            linewidth=1, linestyle='-', alpha=0.2)
    
    # Draw arrows
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.plot((1), (0), marker='>',
            transform=ax.get_yaxis_transform(), **arrow_fmt)
    ax.plot((0), (1), marker='^',
            transform=ax.get_xaxis_transform(), **arrow_fmt)
    
    
    def __init__(self) -> None:
        # Create figure 
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
    
    # plotmatriz
    def plotMatrix(self,X,Y):
        
        nColores = len(Y)
        dicColores = {}
        colorList = []
        # agregamos las combinaciones de las salidas Y como llave a un diccionario 
        # y les asignamos un numero en rango[0,nColores-1] como valor
        # garantiza mismo valor para patrones repetidos en Y
        for opc in range(nColores):
            dicColores[str(Y[opc])] = opc
            
        # el valor del patron se usa para calcular 
        # un numero en rango [0,100] 
        for e in Y:
            colorList.append( int( ( 100 / (nColores-1)) * dicColores[str(e)]))

        # colorList = np.array(colorList)
        
        print("Dic Colores: ",dicColores)
        print("ColorList: ",colorList)

        # ploteamos todos los puntos en x,y con su color asignago en colorList
        self.ax.scatter(X[0],X[1],c=colorList,cmap="nipy_spectral",s=300)
        self.ax.set_ylim([-5, 5])




    # plotteamos linea 
    def drawDivision(self, pts, _color="r"):
        # Plot de la linea divisoria
        plt.plot([self.rangoLinea[0], self.rangoLinea[1]], [pts[0], pts[1]], color=_color)


    

