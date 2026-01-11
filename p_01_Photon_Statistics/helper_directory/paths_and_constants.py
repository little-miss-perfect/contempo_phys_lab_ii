# mediciones para el promedio (primera muestra):

# day_1
path_0_day_1 = "samples/day_1/m_0.csv"
# que tomamos a "500000E-6 s"
t_0_day_1 = 500000E-6

# day_2
path_0_day_2 = "samples/day_2/01_1micros/Cuentas.csv"
# que tomamos a "500000E-6 s"
t_0_day_2 = 1E-6

# muestras para el reporte:


# íbamos a definirlas de uno en uno, pero mejor tomemos una lista de los paths

# día 1
paths_day_1 = [ "samples/day_1/equipo_3_n1/Cuentas.csv",          # 'df_1': np.float64(1.04)
                "samples/day_1/equipo_3_n5/Cuentas.csv",          # 'df_2': np.float64(4.71)
                "samples/day_1/equipo_3_n10/Cuentas.csv",         # 'df_3': np.float64(4.925)
                "samples/day_1/equipo_3_n100/Cuentas.csv",        # 'df_4': np.float64(5.095)
              ]  # los promedios que se muestran a la derecha de cada path, ya los obtuvimos en "main.py"

# print(len(paths))  # es "4" porque no consideramos la medida usada para "e_v"

# día 2
paths_day_2 = [ "samples/day_2/01_1micros/Cuentas.csv",          # 'df_1': np.float64(1.04)
                "samples/day_2/02_5micros/Cuentas.csv",          # 'df_2': np.float64(4.71)
                "samples/day_2/03_10.6micros/Cuentas.csv",       # 'df_3': np.float64(4.925)
                "samples/day_2/04_100micros/Cuentas.csv",        # 'df_4': np.float64(5.095)
                "samples/day_2/05_106micros/Cuentas.csv",
              ]

# y para "la regla de tres":

# 500000 (estámos midiendo en micro segundos)  -->  e_v (número de fotones)  # ya conocemos estos dos valores de la primera muestra (nosotros ajustamos el valor de "500000E-6 s")
#                 ?                            -->   n  (número de fotones)

def tiempo_requerido(n, e_v, escala="micro"):
    '''
    nos regresa el tiempo (en una escala preseleccionada) requerido
    para ver "n" fotones (en promedio)
    dado que sabemos que vimos "e_v" fotones en "500E-6 s".
    inicialmente, la escala está en microsegundos
    porque así funciona el programa que tiene el laboratorio.
    '''

    # para las unidades
    not_scient = {
                "nano": 1E-9,
                "micro": 1E-6,
                "mili": 1E-3,
                "": 1,
                }

    # el tiempo en segundos está dado por: t = n / proporción, proporción = e_v / (500000E-6 s)
    # así que: t = n * 500000E-6 / e_v
    t_segundos = (n * 500000E-6 / e_v)

    # y para la conversión de escala tomamos: segundos / escala_de_segundos
    return t_segundos / not_scient[escala]

# (ya usamos esta función en "main.py", por eso conocemos la lista "tiempos_en_micro_segundos")

# de nuestros promedios propuestos para un valor esperado ("expected value") dado por los elementos de la siguiente lista

# pruebas = [e_v, 1, 5, 10, 100]

# probamos los siguientes tiempos (correspondientes a la lista anterior)

tiempos_en_micro_segundos = [500000, 0.5, 2.5, 5.1, 50.8]
# print(len(tiempos_en_micro_segundos))  # son "5" muestras que tomamos. o sea, "5" archivos ".csv"
