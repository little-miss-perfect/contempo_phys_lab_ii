import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper_directory.paths_and_constants import (
    path_0_day_1, path_0_day_2, t_0_day_1, t_0_day_2, pruebas_day_1, pruebas_day_2, paths_day_1, paths_day_2, tiempos_en_micro_segundos, tiempo_requerido
)

from helper_directory.load_csv_files import build_original_y_copia

from helper_directory.plotting import fdp_histograma, best_grid


# y aquí tenemos que decidir si usamos la medición del día uno o del día dos.

def main_func(path_0_day, t_0_day, pruebas, paths_day, escala="micro"):
    # choose the day
    df_0_original = pd.read_csv(path_0_day, encoding='utf-8')
    df_0_copia = df_0_original.copy(deep=True)

    df_0_copia.head()

    e_v = df_0_copia.mean()
    e_v = e_v.iloc[0]

    # print(e_v)  # tiene decimales, como es de esperarse

    print(f"el primer valor esperado fue de unos '{round(e_v)}' fotones a '{t_0_day} s'.")

    # y aplicamos la regla de "3" que nos pidieron
    # pruebas = [e_v, 1, 5, 10, 100]  # (por ejemplo, sin "e_v") esto lo debemos convertir en un parámetro importado de "paths_and_constants" que nos diga el promedio del conteo de fotones de cada muestra que tomamos en un día dado

    print()
    print("en la regla de tres: \n")
    for i in pruebas:
        print(f"para un valor esperado de '{i}' fotones, \n"
              f"requerimos de '{tiempo_requerido(i, e_v, escala=escala):.6f}' {escala}segundos \n")

    # y creamos los dataframes
    original, copia = build_original_y_copia(paths_day, encoding='utf-8')  # que indexa los dataframes con naturales (checa la documentación)
    # print(copia['df_1'].head())  # un sanity check

    # donde los demás valores esperados fueron
    for i in range(1, len(paths_day) + 1):  # ojo con los índices aquí y el "range"
        name = f"df_{i}"
        print(f"'{name}' con valor esperado:\n{copia[name].mean()}\n")

    # para luego hacer los histogramas, tenemos dos opciones.
    names = list(copia.keys())

    option = input("selecciona '1' para un archivo de los histogramas y '2' para archivos separados de los histogramas: ")

    while True:

        if option == "1":
            # (1) todos los histogramas en un archivo
            n_plots = len(names)
            rows, cols = best_grid(n_plots)  # para visualizar estéticamente los plots en una sóla imagen
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), constrained_layout=True)

            axes = np.atleast_1d(axes)

            for ax, name in zip(axes.ravel(), names):
                fdp_histograma(copia[name], ax=ax, title=name)  # no save_path here

            fig.savefig("histograms/all_histograms.svg", bbox_inches="tight")

            for ax in axes.ravel()[len(names):]:
                ax.axis("off")                      # if you made a "2×3" grid ("6" slots) but only plotted "4" datasets,
                                                    # this turns off the last "2" empty slots so the figure looks clean.

            plt.show()

            break

        elif option == "2":

            # (2) todos los histogramas en archivos separados
            for name in names:
                fdp_histograma(copia[name], title=name, save_path=f"histograms/{name}.svg")
            plt.show()

            break

        else:
            print("opción no valida")
            option = input(
                "selecciona '1' para un archivo de los histogramas y '2' para archivos separados de los histogramas: ")  # y así seguimos preguntando

    # los valores esperados que probamos fueron los siguientes
    # [1, 5, 10, 100]

# y a ver si funcionó
main_func(path_0_day_1, t_0_day_1, pruebas_day_1, paths_day_1)
