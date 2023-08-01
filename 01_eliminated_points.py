import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from matplotlib.lines import Line2D

# importar el nombre del archivo txt
with open('names_data.py') as f:
    code = compile(f.read(), 'names_data.py', 'exec')
    exec(code)

# importar la data a un dataframe
# name_data = 'SU04C-1.txt'
data = pd.read_table('00_Data_exp/' + name_data, sep = '\t', names = ['step', 'date', 'ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'cabezal', 'nucleo', 'esf', 'def'], skiprows = 3, header = None, encoding = 'latin')

# convertir a listas las fuerzas, los desplazamientos y pasos
P = np.array(data['ch0'])
D = np.array(data['nucleo'])

# definir listas para graficar el historial de fuerzas
steps_P = list(range(len(P)))
steps_D = list(range(len(D)))

# crear una lista donde se guardarán los índices seleccionados a eliminar
selected_points = []

# graficar los datos desplazamiento vs fuerza
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (14,8))
line, = ax1.plot(D, P, 'o', color = 'black', ms = 2, picker = True, pickradius = 5)
# line, = ax2.plot(D, P, 'o-', color = 'black', ms = 2, picker = True, pickradius = 5)
line, = ax2.plot(D, P, linestyle = '-', color = 'gray', linewidth=0.5, marker = 'o', markerfacecolor='none', markeredgecolor='red', markersize=2)
line, = ax3.plot(steps_P, P, 'o', color = 'black', ms = 2, picker = True, pickradius = 5)
line, = ax4.plot(steps_D, D, 'o', color = 'black', ms = 2, picker = True, pickradius = 5)
crs = mplcursors.cursor([ax1, ax3, ax4], hover = True)
crs.connect("add", lambda sel: sel.annotation.set_text('x: {} \ny: {} \nstep: {}'.format(sel.target[0], sel.target[1], sel.target.index)))

# función para vincular los ejes de los subplots 1 y 2
def on_release_ax1(event):
    if event.button == 1 and event.inaxes == ax1:
        ax2.set_xlim(*event.inaxes.get_xlim())
        ax2.set_ylim(*event.inaxes.get_ylim())

# función para vincular los ejes de los subplots 3 y 4        
def on_release_ax3(event):
    if event.button == 1 and event.inaxes == ax3:
        ax4.set_xlim(*event.inaxes.get_xlim())
        ax4.set_ylim(*event.inaxes.get_ylim())

# conectamos los eventos de zoom de los subplots 1 y 3 a las funciones correspondientes        
fig.canvas.mpl_connect('button_release_event', on_release_ax1)
fig.canvas.mpl_connect('button_release_event', on_release_ax3)

# ajustamos los márgenes de la figura
fig.tight_layout()


# definir una función que obtenga los puntos al hacer click en la gráfica
def onpick1(event):
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        selected_points.append(int(ind))
        print('onpick1 line:', int(ind), float(xdata[ind]), float(ydata[ind]))

# definir una función que modifique la data
def on_close(event):
    #obtener lista con los índices seleccionados
    lista_sin_repetidos = list(set(selected_points))
    lista_indices = sorted(lista_sin_repetidos)
    lista_indices_txt = [num + 3 for num in lista_indices]
    
    print('Índices de los puntos eliminados de la data original:\n', lista_indices_txt)
    
    # Leer el archivo de texto en memoria y almacenarlo en una lista
    with open('00_Data_exp/' + name_data, 'r') as f:
        lista_datos = f.readlines()

    # Eliminar la primera y quinta fila de la lista
    filas_a_eliminar = lista_indices_txt
    lista_datos = [dato for i, dato in enumerate(lista_datos) if i not in filas_a_eliminar]

    # Escribir la lista actualizada en el archivo de texto
    with open('01_Data_pre_procesada/' + name_data[:-4] + '.txt', 'w') as f:
        f.writelines(lista_datos)

fig.canvas.mpl_connect('close_event', on_close)
fig.canvas.mpl_connect('pick_event', onpick1)

