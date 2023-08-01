import pandas as pd
import numpy as np
from smoothing_functions import *
import matplotlib.pyplot as plt
import mplcursors
import itertools
import os

# importar el nombre del archivo modificado
with open('names_data.py') as f:
    code = compile(f.read(), 'names_data.py', 'exec')
    exec(code)
    
# datos de entrada según el espécimen
A = 15*15 # [cm²]
Hn = 225 # [mm]

# importar la data a un dataframe
# name_data = 'C140C_new.txt'
# name_data_test = 'C140C.txt'
data = pd.read_table('01_Data_pre_procesada/' + name_data, sep = '\t', names = ['step', 'date', 'ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'cabezal', 'nucleo', 'esf', 'def'], skiprows = 3, header = None, encoding = 'latin')
data1 = pd.read_table('00_Data_exp/' + name_data, sep = '\t', names = ['step', 'date', 'ch0', 'ch1', 'ch2', 'ch3', 'ch4', 'cabezal', 'nucleo', 'esf', 'def'], skiprows = 3, header = None, encoding = 'latin')

# convertir a listas las fuerzas, los desplazamientos
P = data['ch0'].tolist()
D = data['nucleo'].tolist()
P1 = data1['ch0'].tolist()
D1 = data1['nucleo'].tolist()

# índice donde se encuentra la máxima carga
indice_fc = P.index(max(P))

# lista de los índices seleccionados
# lista_indices = [0, 158, 256, 404, 471, 584, 762, 850, 883, 1032, 1257, 1269, 1401, 1638, 1651, 1778, 1989, 2023, 2147, 2359, 2460, 2604, 2765, 3382, 3529]
# lista_indices = [0, 158, 256, 404, 471, 584, 762, 850, 883, 1032, 1257, 1269, 1401, 1638, 1651, 1778, 1989, 2023, 2152, 2359, 2460, 2604, 2765, 3382, 3529]
lista_indices = [0, 130, 204, 305, 364, 447, 599, 639, 728, 917, 938, 960, 1070, 1253, 1284, 1377, 1534, 1607, 1715, 1878, 2119, 2229]

# exportar a un archivo .txt los valores de la lista de los índices seleccionados
id_archivo = open('07_Id_puntos/' + name_data[:-4] + '_ID.txt', 'w')
cadena = ", ".join(str(elemento) for elemento in lista_indices)
id_archivo.write(cadena)
id_archivo.close()

index = lista_indices.index(indice_fc)
pre_peak_indices = lista_indices[:index+1]
pos_peak_indices = lista_indices[index:]

# diccionario donde se guardará la data filtrada
fuerza_desplazamiento_x = {}
fuerza_desplazamiento_y = {}

# curvas de carga
load_x = []
load_y = []
# seleccionar las curvas de carga pre pico
for i in range(len(pre_peak_indices)):
    if i % 3 == 0:
        curva_ini = pre_peak_indices[i]
        curva_fin = pre_peak_indices[i+1]
        carga_x = D[curva_ini:curva_fin+1]
        carga_y = P[curva_ini:curva_fin+1]
        
        # Combinar las listas X y Y en una lista de pares ordenados
        puntos_0 = list(zip(carga_x, carga_y))

        # Pares ordenados cuyas ordenadas son el promedio
        puntos_1 = ordenadas_prom(puntos_0, 0.001)

        # Pares ordenados cuyas ordenadas hacia arriba son modificadas
        puntos_2 = points_up(puntos_1)

        # Pares ordenados cuyas ordenadas hacia abajo son modificadas
        puntos_3 = points_down(puntos_2)
        
        # convertir a lista plana la lista de pares ordenados
        x1 = [p[0] for p in puntos_3]
        y1 = [p[1] for p in puntos_3]
        
        # guardar la lista plana en el diccionario
        fuerza_desplazamiento_x[i] = x1
        fuerza_desplazamiento_y[i] = y1
        
        # almacenar en una lista cada curva
        load_x.append(x1)
        load_y.append(y1)

# seleccionar las curvas de carga pos pico
for i in range(len(pos_peak_indices)):
    if i % 3 == 0:
        curva_ini = pos_peak_indices[i]
        curva_fin = pos_peak_indices[i+1]
        carga_x = D[curva_ini:curva_fin+1]
        carga_y = P[curva_ini:curva_fin+1]
        
        # Combinar las listas X y Y en una lista de pares ordenados
        puntos_0 = list(zip(carga_x, carga_y))

        # Pares ordenados cuyas ordenadas son el promedio
        puntos_1 = ordenadas_prom(puntos_0, 0.001)

        # Pares ordenados cuyas ordenadas hacia arriba son modificadas
        puntos_2 = points_up(puntos_1)

        # Pares ordenados cuyas ordenadas hacia abajo son modificadas
        puntos_3 = points_down(puntos_2)
        
        # convertir a lista plana la lista de pares ordenados
        x2 = [p[0] for p in puntos_3]
        y2 = [p[1] for p in puntos_3]
        
        # guardar la lista plana en el diccionario
        fuerza_desplazamiento_x[i + len(pre_peak_indices)] = x2
        fuerza_desplazamiento_y[i + len(pre_peak_indices)] = y2
        
        # almacenar en una lista cada curva
        load_x.append(x2)
        load_y.append(y2)

# convertir a lista plana la lista de listas donde se almacenan las curvas
load_curve_x = sum(load_x, [])
load_curve_y = sum(load_y, [])

# curvas de descarga
unload_x = []
unload_y = []
# seleccionar las curvas de descarga pre pico
for i in range(len(pre_peak_indices)-2):
    if i % 3 == 0:
        curva_ini = pre_peak_indices[i+1]
        curva_fin = pre_peak_indices[i+2]
        carga_x = D[curva_ini:curva_fin+1]
        carga_y = P[curva_ini:curva_fin+1]
        
        # Combinar las listas X y Y en una lista de pares ordenados
        puntos_0 = list(zip(carga_x, carga_y))

        # Pares ordenados cuyas ordenadas son el promedio
        puntos_1 = ordenadas_prom(puntos_0, 0.001)

        # Pares ordenados cuyas ordenadas hacia arriba son modificadas
        puntos_2 = points_up(puntos_1)

        # Pares ordenados cuyas ordenadas hacia abajo son modificadas
        puntos_3 = points_down(puntos_2)
        
        # convertir a lista plana la lista de pares ordenados
        x1 = [p[0] for p in puntos_3]
        y1 = [p[1] for p in puntos_3]
        
        # guardar la lista plana en el diccionario
        fuerza_desplazamiento_x[i+1] = x1
        fuerza_desplazamiento_y[i+1] = y1
        
        # almacenar en una lista cada curva
        unload_x.append(x1)
        unload_y.append(y1)

# seleccionar las curvas de descarga pos pico
for i in range(len(pos_peak_indices)-2):
    if i % 3 == 0:
        curva_ini = pos_peak_indices[i+1]
        curva_fin = pos_peak_indices[i+2]
        carga_x = D[curva_ini:curva_fin+1]
        carga_y = P[curva_ini:curva_fin+1]
        
        # Combinar las listas X y Y en una lista de pares ordenados
        puntos_0 = list(zip(carga_x, carga_y))

        # Pares ordenados cuyas ordenadas son el promedio
        puntos_1 = ordenadas_prom(puntos_0, 0.001)

        # Pares ordenados cuyas ordenadas hacia arriba son modificadas
        puntos_2 = points_up(puntos_1)

        # Pares ordenados cuyas ordenadas hacia abajo son modificadas
        puntos_3 = points_down(puntos_2)
        
        # convertir a lista plana la lista de pares ordenados
        x2 = [p[0] for p in puntos_3]
        y2 = [p[1] for p in puntos_3]
        
        # guardar la lista plana en el diccionario
        fuerza_desplazamiento_x[i + 1 + len(pre_peak_indices)] = x2
        fuerza_desplazamiento_y[i + 1 + len(pre_peak_indices)] = y2
        
        # almacenar en una lista cada curva
        unload_x.append(x2)
        unload_y.append(y2)

# convertir a lista plana la lista de listas donde se almacenan las curvas
unload_curve_x = sum(unload_x, [])
unload_curve_y = sum(unload_y, [])

# curvas de recarga
reload_x = []
reload_y = []
# seleccionar las curvas de recarga pre pico
for i in range(len(pre_peak_indices)-3):
    if i % 3 == 0:
        curva_ini = pre_peak_indices[i+2]
        curva_fin = pre_peak_indices[i+3]
        carga_x = D[curva_ini:curva_fin+1]
        carga_y = P[curva_ini:curva_fin+1]
        
        # Combinar las listas X y Y en una lista de pares ordenados
        puntos_0 = list(zip(carga_x, carga_y))

        # Pares ordenados cuyas ordenadas son el promedio
        puntos_1 = ordenadas_prom(puntos_0, 0.001)

        # Pares ordenados cuyas ordenadas hacia arriba son modificadas
        puntos_2 = points_up(puntos_1)

        # Pares ordenados cuyas ordenadas hacia abajo son modificadas
        puntos_3 = points_down(puntos_2)
        
        # convertir a lista plana la lista de pares ordenados
        x1 = [p[0] for p in puntos_3]
        y1 = [p[1] for p in puntos_3]
        
        # guardar la lista plana en el diccionario
        fuerza_desplazamiento_x[i+2] = x1
        fuerza_desplazamiento_y[i+2] = y1
        
        # almacenar en una lista cada curva
        reload_x.append(x1)
        reload_y.append(y1)

# seleccionar las curvas de recarga pos pico
for i in range(len(pos_peak_indices)-3):
    if i % 3 == 0:
        curva_ini = pos_peak_indices[i+2]
        curva_fin = pos_peak_indices[i+3]
        carga_x = D[curva_ini:curva_fin+1]
        carga_y = P[curva_ini:curva_fin+1]
        
        # Combinar las listas X y Y en una lista de pares ordenados
        puntos_0 = list(zip(carga_x, carga_y))

        # Pares ordenados cuyas ordenadas son el promedio
        puntos_1 = ordenadas_prom(puntos_0, 0.001)

        # Pares ordenados cuyas ordenadas hacia arriba son modificadas
        puntos_2 = points_up(puntos_1)

        # Pares ordenados cuyas ordenadas hacia abajo son modificadas
        puntos_3 = points_down(puntos_2)
        
        # convertir a lista plana la lista de pares ordenados
        x2 = [p[0] for p in puntos_3]
        y2 = [p[1] for p in puntos_3]
        
        # guardar la lista plana en el diccionario
        fuerza_desplazamiento_x[i + 2 + len(pre_peak_indices)] = x2
        fuerza_desplazamiento_y[i + 2 + len(pre_peak_indices)] = y2
        
        # almacenar en una lista cada curva
        reload_x.append(x2)
        reload_y.append(y2)

# convertir a lista plana la lista de listas donde se almacenan las curvas
reload_curve_x = sum(reload_x, [])
reload_curve_y = sum(reload_y, [])

# ordenar el diccionario donde se almacenan todas las curvas
fuerza_desplazamiento_x_sorted = dict(sorted(fuerza_desplazamiento_x.items()))
fuerza_desplazamiento_y_sorted = dict(sorted(fuerza_desplazamiento_y.items()))

# pasar a lista el diccionario creado
desplazamiento = [valor for valor in fuerza_desplazamiento_x_sorted.values()]
fuerza = [valor for valor in fuerza_desplazamiento_y_sorted.values()]

# convertir a lista plana la lista antes creada
desplazamiento_plano = list(itertools.chain(*desplazamiento))
fuerza_plano = list(itertools.chain(*fuerza))

# exportar a txt la data filtrada
data_txt = {'X [mm]': desplazamiento_plano, 'Y [tonf]': fuerza_plano}
data_to_txt = pd.DataFrame(data_txt)
data_to_txt.to_csv('03_Data_txt_02/' + name_data[:-4] + '.txt', sep = '\t', index = False)
        
############################################################################### RUN N°01 ###############################################################################
# Graficar los puntos
plt.plot(load_curve_x, load_curve_y, 'o', linewidth=1, markersize=2, color='red')
plt.plot(unload_curve_x, unload_curve_y, 'o', linewidth=1, markersize=5, color='cyan', fillstyle='none')
# plt.plot(unload_curve_x, unload_curve_y, '-', linewidth=0.5, color='cyan', alpha=0.25)
plt.plot(reload_curve_x, reload_curve_y, 'o', linewidth=1, markersize=2, color='black', fillstyle='none')
# plt.plot(reload_curve_x, reload_curve_y, '-', linewidth=0.5, color='black', alpha=0.25)
plt.plot(D1, P1, '-', linewidth=0.5, markersize=2, color='gray')
plt.show()

############################################################################### RUN N°02 ###############################################################################
# graficar la curva esfuerzo vs deformación
εc, σc, ε, σ = esf_def(P, D, desplazamiento_plano, fuerza_plano, A, Hn)

# exportar a un archivo .txt los esfuerzos y deformaciones
data_txt = {'ε': ε, 'σ': σ}
data_to_txt = pd.DataFrame(data_txt)
data_to_txt.to_csv('06_Data_procesada/' + name_data[:-4] + '.txt', sep = '\t', index = False)

############################################################################### RUN N°03 ###############################################################################
# normalizar las curvas de carga, descarga y recarga experimentales
curvas = [(load_x, load_y), (unload_x, unload_y), (reload_x, reload_y)]
load_norm, unload_norm, reload_norm = normalizar(P, D, desplazamiento_plano, fuerza_plano, curvas)

fig, ax = plt.subplots(figsize=(12,8))
for i in range(len(load_norm)):
    line, = ax.plot(load_norm[i][0], load_norm[i][1], 'o-', color = 'red', ms = 2, picker = True, pickradius = 5)
for i in range(len(unload_norm)):
    line, = ax.plot(unload_norm[i][0], unload_norm[i][1], 'o-', color = 'black', ms = 5, picker = True, pickradius = 5, fillstyle='none')
for i in range(len(reload_norm)):
    line, = ax.plot(reload_norm[i][0], reload_norm[i][1], 'o-', color = 'cyan', ms = 2, picker = True, pickradius = 5)
crs = mplcursors.cursor(ax, hover=True)
crs.connect("add", lambda sel: sel.annotation.set_text('x: {} \ny: {} \nstep: {}'.format(sel.target[0], sel.target[1], sel.target.index)))

def onpick1(event):
    if isinstance(event.artist, Line2D):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        print(float(xdata[ind]))
        
fig.canvas.mpl_connect('pick_event', onpick1)

############################################################################### RUN N°04 ###############################################################################
# deformación de descarga vs deformación plástica (normalizados)
def_plas = deformacion_plastica(unload_norm)
εun = [tupla[0] for tupla in def_plas]
εpl = [tupla[1] for tupla in def_plas]

# exportar a txt las deformaciones
data_txt = {'εun': εun, 'εpl': εpl}
data_to_txt = pd.DataFrame(data_txt)
data_to_txt.to_csv('04_εun_εpl/' + name_data[:-4] + '.txt', sep = '\t', index = False)

# graficar las deformaciones
plt.plot(εun, εpl, 'o', markersize = 2, color = 'red')
plt.show()

############################################################################### RUN N°05 ###############################################################################
# separar la curva envolvente antes y después de la fuerza máxima
env_x = []
env_y = []
for i in range(len(load_norm)):
    env_x.append(load_norm[i][0])
    env_y.append(load_norm[i][1])
env_x_plano = list(itertools.chain(*env_x))
env_y_plano = list(itertools.chain(*env_y))
id_fc = env_y_plano.index(max(env_y_plano))

env_pre = [(env_x_plano[:id_fc + 1], env_y_plano[:id_fc + 1])]
env_pos = [(env_x_plano[id_fc:], env_y_plano[id_fc:])]

plt.plot(env_x_plano, env_y_plano, 'o-', markersize = 2, color  = 'green')

############################################################################### RUN N°06 ###############################################################################
# graficar las curvas de carga pre pico en coordenadas locales
coordenadas_locales = coord_locales(env_pre, cond = [0])

# crear las listas donde se almacenaran los puntos en coordenadas locales
# OBSERVACIÓN: tener en cuenta comentar estas listas para cada txt después del primer txt
coordenadas_locales_x = []
coordenadas_locales_y = []

# crear el archivo excel
archivo_excel = openpyxl.Workbook()

for i in range(len(coordenadas_locales)):
    
    # almacenar los puntos de cada tupla en las listas antes creadas
    coordenadas_locales_x.append(coordenadas_locales[i][0])
    coordenadas_locales_y.append(coordenadas_locales[i][1])
    
    # crear la hoja excel para cada curva
    hoja_trabajo = archivo_excel.create_sheet(f"Curva {i+1}")
    
    # crear el título para las columnas
    hoja_trabajo.cell(row=1, column=1).value = "ξ"
    hoja_trabajo.cell(row=1, column=2).value = "η"
    
    for j in range(len(coordenadas_locales[i][0])):
        
        # escribir cada coordenada en la hoja de excel creada
        hoja_trabajo.cell(row=j+2, column=1).value = coordenadas_locales[i][0][j]
        hoja_trabajo.cell(row=j+2, column=2).value = coordenadas_locales[i][1][j]
        
# exportar el archivo excel
archivo_excel.save('05_Curvas_CoordLocal/' + name_data[:-4] + '_load_pre.xlsx')

# convertir a lista plana las listas donde se almacenaron los puntos de cada tupla
coordenadas_locales_x_flat = list(itertools.chain(*coordenadas_locales_x))
coordenadas_locales_y_flat = list(itertools.chain(*coordenadas_locales_y))

############################################################################### RUN N°07 ###############################################################################
# graficar las curvas de carga pos pico en coordenadas locales
coordenadas_locales = coord_locales(env_pos, cond = [0])

# crear las listas donde se almacenaran los puntos en coordenadas locales
# OBSERVACIÓN: tener en cuenta comentar estas listas para cada txt después del primer txt
coordenadas_locales_x = []
coordenadas_locales_y = []

# crear el archivo excel
archivo_excel = openpyxl.Workbook()

for i in range(len(coordenadas_locales)):
    
    # almacenar los puntos de cada tupla en las listas antes creadas
    coordenadas_locales_x.append(coordenadas_locales[i][0])
    coordenadas_locales_y.append(coordenadas_locales[i][1])
    
    # crear la hoja excel para cada curva
    hoja_trabajo = archivo_excel.create_sheet(f"Curva {i+1}")
    
    # crear el título para las columnas
    hoja_trabajo.cell(row=1, column=1).value = "ξ"
    hoja_trabajo.cell(row=1, column=2).value = "η"
    
    for j in range(len(coordenadas_locales[i][0])):
        
        # escribir cada coordenada en la hoja de excel creada
        hoja_trabajo.cell(row=j+2, column=1).value = coordenadas_locales[i][0][j]
        hoja_trabajo.cell(row=j+2, column=2).value = coordenadas_locales[i][1][j]
        
# exportar el archivo excel
archivo_excel.save('05_Curvas_CoordLocal/' + name_data[:-4] + '_load_pos.xlsx')

# convertir a lista plana las listas donde se almacenaron los puntos de cada tupla
coordenadas_locales_x_flat = list(itertools.chain(*coordenadas_locales_x))
coordenadas_locales_y_flat = list(itertools.chain(*coordenadas_locales_y))

############################################################################### RUN N°08 ###############################################################################
# graficar las curvas de descarga en coordenadas locales
coordenadas_locales = coord_locales(unload_norm, cond = [0])

# crear las listas donde se almacenaran los puntos en coordenadas locales
# OBSERVACIÓN: tener en cuenta comentar estas listas para cada txt después del primer txt
coordenadas_locales_x = []
coordenadas_locales_y = []

# crear el archivo excel
archivo_excel = openpyxl.Workbook()

for i in range(len(coordenadas_locales)):
    
    # almacenar los puntos de cada tupla en las listas antes creadas
    coordenadas_locales_x.append(coordenadas_locales[i][0])
    coordenadas_locales_y.append(coordenadas_locales[i][1])
    
    # crear la hoja excel para cada curva
    hoja_trabajo = archivo_excel.create_sheet(f"Curva {i+1}")
    
    # crear el título para las columnas
    hoja_trabajo.cell(row=1, column=1).value = "ξ"
    hoja_trabajo.cell(row=1, column=2).value = "η"
    
    for j in range(len(coordenadas_locales[i][0])):
        
        # escribir cada coordenada en la hoja de excel creada
        hoja_trabajo.cell(row=j+2, column=1).value = coordenadas_locales[i][0][j]
        hoja_trabajo.cell(row=j+2, column=2).value = coordenadas_locales[i][1][j]
        
# exportar el archivo excel
archivo_excel.save('05_Curvas_CoordLocal/' + name_data[:-4] + '_un.xlsx')

# convertir a lista plana las listas donde se almacenaron los puntos de cada tupla
coordenadas_locales_x_flat = list(itertools.chain(*coordenadas_locales_x))
coordenadas_locales_y_flat = list(itertools.chain(*coordenadas_locales_y))

############################################################################### RUN N°09 ###############################################################################
# graficar las curvas de recarga en coordenadas locales
coordenadas_locales = coord_locales(reload_norm, cond = [0])

# crear las listas donde se almacenaran los puntos en coordenadas locales
# OBSERVACIÓN: tener en cuenta comentar estas listas para cada txt después del primer txt
coordenadas_locales_x = []
coordenadas_locales_y = []

# crear el archivo excel
archivo_excel = openpyxl.Workbook()

for i in range(len(coordenadas_locales)):
    
    # almacenar los puntos de cada tupla en las listas antes creadas
    coordenadas_locales_x.append(coordenadas_locales[i][0])
    coordenadas_locales_y.append(coordenadas_locales[i][1])
    
    # crear la hoja excel para cada curva
    hoja_trabajo = archivo_excel.create_sheet(f"Curva {i+1}")
    
    # crear el título para las columnas
    hoja_trabajo.cell(row=1, column=1).value = "ξ"
    hoja_trabajo.cell(row=1, column=2).value = "η"
    
    for j in range(len(coordenadas_locales[i][0])):
        
        # escribir cada coordenada en la hoja de excel creada
        hoja_trabajo.cell(row=j+2, column=1).value = coordenadas_locales[i][0][j]
        hoja_trabajo.cell(row=j+2, column=2).value = coordenadas_locales[i][1][j]
        
# exportar el archivo excel  
archivo_excel.save('05_Curvas_CoordLocal/' + name_data[:-4] + '_re.xlsx')

# convertir a lista plana las listas donde se almacenaron los puntos de cada tupla
coordenadas_locales_x_flat = list(itertools.chain(*coordenadas_locales_x))
coordenadas_locales_y_flat = list(itertools.chain(*coordenadas_locales_y))

############################################################################### RUN N°10 ###############################################################################
# exportar los parámetros a una tabla de excel
workbook = openpyxl.load_workbook('___parametros.xlsx')
hoja = workbook['parametros']
fila_vacia = hoja.max_row + 1
while hoja.cell(row=fila_vacia, column=1).value is not None:
    fila_vacia += 1

hoja.cell(row = fila_vacia, column = 1, value = name_data[:-4])
hoja.cell(row = fila_vacia, column = 2, value = σc)
hoja.cell(row = fila_vacia, column = 3, value = εc)

workbook.save('___parametros.xlsx')









