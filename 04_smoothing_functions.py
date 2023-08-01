import numpy as np
import operator
import matplotlib.pyplot as plt
import openpyxl

# def ordenadas_prom(puntos, error_maximo):
    
#     Y = [p[1] for p in puntos]
    
#     # Definir el diccionario para almacenar las ordenadas
#     ordenadas_promedio_por_abscisa = {}

#     # Recorrer la lista de puntos y agregar las ordenadas al diccionario correspondiente
#     for abscisa, ordenada in puntos:
#         abscisa_coincidente = False
#         for abscisa_dic, ordenadas in ordenadas_promedio_por_abscisa.items():
#             if abs(abscisa - abscisa_dic) <= error_maximo:
#                 abscisa_coincidente = True
#                 ordenadas.append(ordenada)
#                 break
#         if not abscisa_coincidente:
#             ordenadas_promedio_por_abscisa[abscisa] = [ordenada]

#     # Calcular el promedio de las ordenadas para cada abscisa
#     for abscisa, ordenadas in ordenadas_promedio_por_abscisa.items():
#         ordenadas_promedio_por_abscisa[abscisa] = sum(ordenadas) / len(ordenadas)

#     if Y[-1] > Y[0]:
#         # Ordenar el diccionario por clave de menor a mayor
#         ordenadas_promedio_por_abscisa = dict(sorted(ordenadas_promedio_por_abscisa.items()))
#     elif Y[0] > Y[-1]:
#         # Ordenar el diccionario por clave de mayor a menor
#         ordenadas_promedio_por_abscisa = dict(sorted(ordenadas_promedio_por_abscisa.items(), reverse=True))
        
#     # Obtener la menor y mayor ordenada y agregarlas como valores de la primera y última clave, respectivamente
#     primer_clave = list(ordenadas_promedio_por_abscisa.keys())[0]
#     ultima_clave = list(ordenadas_promedio_por_abscisa.keys())[-1]
#     primera_ordenada = Y[0]
#     ultima_ordenada = Y[-1]
    
#     ordenadas_promedio_por_abscisa[primer_clave] = primera_ordenada
#     ordenadas_promedio_por_abscisa[ultima_clave] = ultima_ordenada
    
#     points_new = list(ordenadas_promedio_por_abscisa.items())
    
#     return points_new

def ordenadas_prom(puntos, error_maximo):
    
    Y = [p[1] for p in puntos]
    X = [p[0] for p in puntos]
    
    # Definir el diccionario para almacenar las ordenadas
    ordenadas_promedio_por_abscisa = {}

    # Recorrer la lista de puntos y agregar las ordenadas al diccionario correspondiente
    for abscisa, ordenada in puntos:
        abscisa_coincidente = False
        for abscisa_dic, ordenadas in ordenadas_promedio_por_abscisa.items():
            if abs(abscisa - abscisa_dic) <= error_maximo:
                abscisa_coincidente = True
                ordenadas.append(ordenada)
                break
        if not abscisa_coincidente:
            ordenadas_promedio_por_abscisa[abscisa] = [ordenada]

    # Calcular el promedio de las ordenadas para cada abscisa
    for abscisa, ordenadas in ordenadas_promedio_por_abscisa.items():
        ordenadas_promedio_por_abscisa[abscisa] = sum(ordenadas) / len(ordenadas)

    if Y[-1] > Y[0]:
        # Ordenar el diccionario por clave de menor a mayor
        ordenadas_promedio_por_abscisa = dict(sorted(ordenadas_promedio_por_abscisa.items()))
    elif Y[0] > Y[-1]:
        if X[-1] > X[0]:
            # Ordenar el diccionario por clave de menor a mayor
            ordenadas_promedio_por_abscisa = dict(sorted(ordenadas_promedio_por_abscisa.items()))
        elif X[-1] < X[0]:
            # Ordenar el diccionario por clave de mayor a menor
            ordenadas_promedio_por_abscisa = dict(sorted(ordenadas_promedio_por_abscisa.items(), reverse=True))
        
    # Obtener la menor y mayor ordenada y agregarlas como valores de la primera y última clave, respectivamente
    primer_clave = list(ordenadas_promedio_por_abscisa.keys())[0]
    ultima_clave = list(ordenadas_promedio_por_abscisa.keys())[-1]
    primera_ordenada = Y[0]
    ultima_ordenada = Y[-1]
    
    ordenadas_promedio_por_abscisa[primer_clave] = primera_ordenada
    ordenadas_promedio_por_abscisa[ultima_clave] = ultima_ordenada
    
    points_new = list(ordenadas_promedio_por_abscisa.items())
    
    return points_new


def points_up(puntos):
    """
    Modifica la ordenada de un punto si existe un punto cuya ordenada es mayor que la ordenada del anterior punto y la ordenada de los siguientes dos puntos.

    Args:
        points: una lista de tuplas (x, y) que representan los puntos.

    Returns:
        La lista de tuplas (x, y) con las posibles modificaciones de ordenada.

    """
    
    x_up = [p[0] for p in puntos]
    y_up = [p[1] for p in puntos]
    
    x_values = list(range(len(x_up)))
    # y_values = x_up
    
    if y_up[0] > y_up[-1]:
        y_values = x_up[::-1]
    else:
        y_values = x_up
        
    points = list(zip(x_values, y_values))
    
    modified_points = []

    for i in range(len(points)):
        x, y = points[i]

        if i > 0 and i < len(points) - 2:
            prev_y = points[i - 1][1]
            next_y1 = points[i + 1][1]
            next_y2 = points[i + 2][1]

            if prev_y < y and next_y1 < y and next_y2 < y:
                y = prev_y

        modified_points.append((x, y))
        
    x_new = [p[1] for p in modified_points]
    y_new = y_up
    
    if y_up[0] > y_up[-1]:
        x_new = x_new[::-1]
    else:
        pass
    
    points_new = list(zip(x_new, y_new))

    return points_new


def points_down(puntos):
    """
    Recibe una lista de pares ordenados y actualiza la ordenada de los puntos que cumplen
    la condición especificada en el enunciado del problema.
    """
    
    x_down = [p[0] for p in puntos]
    y_down = [p[1] for p in puntos]
    
    x_values = list(range(len(x_down)))
    # y_values = x_down
    
    if y_down[0] > y_down[-1]:
        y_values = x_down[::-1]
    else:
        y_values = x_down
    
    points = list(zip(x_values, y_values))
    
    n = len(points)
    for i in range(1, n - 1):
        if points[i-1][1] > points[i][1] <= points[i+1][1]:
            points[i] = (points[i][0], points[i-1][1])
            i = 0  # Reiniciamos el índice para empezar desde el primer punto
    
    x_new = [p[1] for p in points]
    y_new = y_down
    
    if y_down[0] > y_down[-1]:
        x_new = x_new[::-1]
    else:
        pass
    
    points_new = list(zip(x_new, y_new))
    
    return points_new

def points_left(puntos):
    
    if puntos[0][0] < puntos[-1][0] and puntos[0][1] < puntos[-1][1]:
        primera_abscisa = puntos[0][0]
        nueva_lista = [par for par in puntos if par[0] >= primera_abscisa]
        return nueva_lista
    else:
        return puntos

# def error2_min(curva_x, curva_y, caso):
#     curva_xg = []
#     curva_yg = []
#     for i in range(len(curva_x)):
#         # obtener los puntos extremos de cada curva
#         x0 = curva_x[i][0]
#         x1 = curva_x[i][-1]
#         y0 = curva_y[i][0]
#         y1 = curva_y[i][-1]
        
#         # trasladar los puntos de la data al sistema local
#         ξ = list(map(lambda x: (x-x0)/(x1-x0), curva_x[i]))
#         η = list(map(lambda x: (x-y0)/(y1-y0), curva_y[i]))
        
#         # construir la matriz inversa
#         p = list(map(lambda x,y: y+12*x**2-28*x**3+15*x**4, ξ, η))
#         q = list(map(lambda x: x-4.5*x**2+6*x**3-2.5*x**4, ξ))
#         r = list(map(lambda x: 1.5*x**2-4*x**3+2.5*x**4, ξ))
#         s = list(map(lambda x: 30*x**2-60*x**3+30*x**4, ξ))
        
#         q2 = sum(list(map(operator.mul, q, q)))
#         r2 = sum(list(map(operator.mul, r, r)))
#         s2 = sum(list(map(operator.mul, s, s)))
        
#         qr = sum(list(map(operator.mul, q, r)))
#         qs = sum(list(map(operator.mul, q, s)))
#         rs = sum(list(map(operator.mul, r, s)))
        
        
#         qp = sum(list(map(operator.mul, q, p)))
#         rp = sum(list(map(operator.mul, r, p)))
#         sp = sum(list(map(operator.mul, s, p)))
        
#         # construir la curva ajustada a cada curva experimental
#         xl = np.linspace(0, 1, 100).tolist()
#         yl = []
        
#         pl = list(map(lambda x: 12*x**2-28*x**3+15*x**4, xl))
#         ql = list(map(lambda x: x-4.5*x**2+6*x**3-2.5*x**4, xl))
#         rl = list(map(lambda x: 1.5*x**2-4*x**3+2.5*x**4, xl))
#         sl = list(map(lambda x: 30*x**2-60*x**3+30*x**4, xl))
        
#         #case=0 (ninguna propiedad física es igual a 0)
#         #case=1 (la pendiente final es igual a 0)
#         #case=2 (la pendiente inicial es igual a 0)
        
#         if caso == 0:
#             matrix = np.array([[q2, qr, qs], [qr, r2, rs], [qs, rs, s2]])
#             vector = np.array([[qp], [rp], [sp]])
#             [n0, n1, s01] = np.linalg.inv(matrix)@vector
            
#             for j in range(len(xl)):
#                 yl.append(float(-pl[j]+ql[j]*n0+rl[j]*n1+sl[j]*s01))
                
#             # print(yl)
            
#         if caso == 1:
#             matrix = np.array([[q2, qs], [qs, s2]])
#             vector = np.array([[qp], [sp]])
#             [n0, s01] = np.linalg.inv(matrix)@vector
            
#             for j in range(len(xl)):
#                 yl.append(float(-pl[j]+ql[j]*n0+sl[j]*s01))
                
            
#         if caso == 2:
#             matrix = np.array([[r2, rs], [rs, s2]])
#             vector = np.array([[rp], [sp]])
#             [n1, s01] = np.linalg.inv(matrix)@vector
            
#             for j in range(len(xl)):
#                 yl.append(float(-pl[j]+rl[j]*n1+sl[j]*s01))
        
#         # trasladar la curva ajustada al sistema global
#         xg = list(map(lambda x: x*(x1-x0)+x0, xl))
#         yg = list(map(lambda x: x*(y1-y0)+y0, yl))
        
#         curva_xg.append(xg)
#         curva_yg.append(yg)
        
#     return curva_xg, curva_yg

def normalizar(P, D, desplazamiento, fuerza, curvas):
    curvas_norm_x = []
    curvas_norm_y = []
    load = []
    unload = []
    reload = []
    fc_ens = max(P)
    εc_ens = D[P.index(max(P))]
    fc_fil = max(fuerza)
    εc_fil = desplazamiento[fuerza.index(max(fuerza))]
    if fc_ens == fc_fil and εc_ens == εc_fil:
        for curva_x, curva_y in curvas:
            curvas_norm_x.append([[elemento / εc_fil for elemento in lista] for lista in curva_x])
            curvas_norm_y.append([[elemento / fc_fil for elemento in lista] for lista in curva_y])
              
        for i in range(len(curvas_norm_x[0])):
              load.append((curvas_norm_x[0][i],curvas_norm_y[0][i]))
            
        for i in range(len(curvas_norm_x[1])):
            unload.append((curvas_norm_x[1][i],curvas_norm_y[1][i]))
            
        for i in range(len(curvas_norm_x[2])):
            reload.append((curvas_norm_x[2][i],curvas_norm_y[2][i]))
    
    return load, unload, reload

def esf_def(P, D, desplazamiento, fuerza, A, Hn):
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    fc_ens = max(P)
    εc_ens = D[P.index(max(P))]
    fc_fil = max(fuerza)
    εc_fil = desplazamiento[fuerza.index(max(fuerza))]
    if fc_ens == fc_fil and εc_ens == εc_fil:
        ε = [x / Hn for x in desplazamiento]
        σ = [x*1000 / A for x in fuerza]
        σc = max(σ)
        εc = ε[σ.index(max(σ))]
    else:
        print('Los datos experimentales no coinciden con la data filtrada')
    
    # eliminar los pares ordenados repetidos
    pares_ordenados = list(zip(ε, σ))
    
    pares_ordenados_sin_puntos_repetidos = []
    pares_unicos = set()
    
    for par in pares_ordenados:
        if par not in pares_unicos:
            pares_ordenados_sin_puntos_repetidos.append(par)
            pares_unicos.add(par)

    ε_new = [tupla[0] for tupla in pares_ordenados_sin_puntos_repetidos]
    σ_new = [tupla[1] for tupla in pares_ordenados_sin_puntos_repetidos]
    
    # ax.plot(ε, σ, 'o', markersize = 2, linewidth = 0.5, color = 'black')
    ax.plot(ε_new, σ_new, 'o-', markersize = 4, linewidth = 0.5, color = 'red', fillstyle = 'none')
    plt.show()
    
    return εc, σc, ε_new, σ_new
        
    

def error2_min(curva, caso):
    curva_teorica = []
    for i in range(len(curva)):
        # obtener los puntos extremos de cada curva
        x0 = curva[i][0][0]
        x1 = curva[i][0][-1]
        y0 = curva[i][1][0]
        y1 = curva[i][1][-1]
        
        # trasladar los puntos de la data al sistema local
        ξ = list(map(lambda x: (x-x0)/(x1-x0), curva[i][0]))
        η = list(map(lambda x: (x-y0)/(y1-y0), curva[i][1]))
        
        # construir la matriz inversa
        p = list(map(lambda x,y: y+12*x**2-28*x**3+15*x**4, ξ, η))
        q = list(map(lambda x: x-4.5*x**2+6*x**3-2.5*x**4, ξ))
        r = list(map(lambda x: 1.5*x**2-4*x**3+2.5*x**4, ξ))
        s = list(map(lambda x: 30*x**2-60*x**3+30*x**4, ξ))
        
        q2 = sum(list(map(operator.mul, q, q)))
        r2 = sum(list(map(operator.mul, r, r)))
        s2 = sum(list(map(operator.mul, s, s)))
        
        qr = sum(list(map(operator.mul, q, r)))
        qs = sum(list(map(operator.mul, q, s)))
        rs = sum(list(map(operator.mul, r, s)))
        
        
        qp = sum(list(map(operator.mul, q, p)))
        rp = sum(list(map(operator.mul, r, p)))
        sp = sum(list(map(operator.mul, s, p)))
        
        # construir la curva ajustada a cada curva experimental
        xl = np.linspace(0, 1, 100).tolist()
        yl = []
        
        pl = list(map(lambda x: 12*x**2-28*x**3+15*x**4, xl))
        ql = list(map(lambda x: x-4.5*x**2+6*x**3-2.5*x**4, xl))
        rl = list(map(lambda x: 1.5*x**2-4*x**3+2.5*x**4, xl))
        sl = list(map(lambda x: 30*x**2-60*x**3+30*x**4, xl))
        
        #case=0 (ninguna propiedad física es igual a 0)
        #case=1 (la pendiente final es igual a 0)
        #case=2 (la pendiente inicial es igual a 0)
        
        if caso == 0:
            matrix = np.array([[q2, qr, qs], [qr, r2, rs], [qs, rs, s2]])
            vector = np.array([[qp], [rp], [sp]])
            [n0, n1, s01] = np.linalg.inv(matrix)@vector
            
            for j in range(len(xl)):
                yl.append(float(-pl[j]+ql[j]*n0+rl[j]*n1+sl[j]*s01))
            
        if caso == 1:
            matrix = np.array([[q2, qs], [qs, s2]])
            vector = np.array([[qp], [sp]])
            [n0, s01] = np.linalg.inv(matrix)@vector
            
            for j in range(len(xl)):
                yl.append(float(-pl[j]+ql[j]*n0+sl[j]*s01))
                
        if caso == 2:
            matrix = np.array([[r2, rs], [rs, s2]])
            vector = np.array([[rp], [sp]])
            [n1, s01] = np.linalg.inv(matrix)@vector
            
            for j in range(len(xl)):
                yl.append(float(-pl[j]+rl[j]*n1+sl[j]*s01))
        
        # trasladar la curva ajustada al sistema global
        xg = list(map(lambda x: x*(x1-x0)+x0, xl))
        yg = list(map(lambda x: x*(y1-y0)+y0, yl))
        
        curva_teorica.append((xg, yg))
        
    return curva_teorica

def deformacion_plastica(unload_norm):
    def_plas = []
    for i in range(len(unload_norm)):
        εun = unload_norm[i][0][0]
        εpl = unload_norm[i][0][-1]
        def_plas.append((εun, εpl))
        
    return def_plas

def coord_locales(curva, cond):
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    coordenadas_locales = []
    
    if len(cond) == 1 and cond[0] == 0:
    
        for i in range(len(curva)):
            x0 = curva[i][0][0]
            x1 = curva[i][0][-1]
            y0 = curva[i][1][0]
            y1 = curva[i][1][-1]
            
            # trasladar los puntos de la data al sistema local
            ξ = list(map(lambda x: (x-x0)/(x1-x0), curva[i][0]))
            η = list(map(lambda x: (x-y0)/(y1-y0), curva[i][1]))
            
            coordenadas_locales.append((ξ,η))
        
            ax.plot(ξ, η, 'o', markersize = 0.5, label = f'Trayectoria N°{i+1}')
            ax.legend()
    
    else:
        
        for i in range(len(curva)):
            
            if i+1 in cond:
            
                x0 = curva[i][0][0]
                x1 = curva[i][0][-1]
                y0 = curva[i][1][0]
                y1 = curva[i][1][-1]
                
                # trasladar los puntos de la data al sistema local
                ξ = list(map(lambda x: (x-x0)/(x1-x0), curva[i][0]))
                η = list(map(lambda x: (x-y0)/(y1-y0), curva[i][1]))
                
                coordenadas_locales.append((ξ,η))
            
                ax.plot(ξ, η, 'o', markersize = 0.5, label = f'Trayectoria N°{i+1}')
                ax.legend()

    plt.show()
    
    return coordenadas_locales
    
    
        
        
    
    
        
    
    