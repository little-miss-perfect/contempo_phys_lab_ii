import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from helper_directory.paths_and_constants import (
    path_0_day_1, path_0_day_2, t_0_day_1, t_0_day_2, paths_day_1, paths_day_2, tiempos_en_micro_segundos, tiempo_requerido
)

from helper_directory.load_csv_files import build_original_y_copia

from helper_directory.plotting import fdp_histograma, best_grid


# la siguiente lista son los promedios con los que trabajaremos

# [e_v, 1, 5, 10, 100]
# sin contar el primer elemento, que fue nuestra referencia para "la regla de tres". es decir, tendremos listas como
# day_1
pruebas_day_1 = [1, 5, 10, 100]  #[0.96, 4.86, 10.35, 100.465]
# day_2
pruebas_day_2 = [1, 5, 10, 100, 106]  # [1.03, 5.365, 10.66, 104.41, 110.195]
# y recuerda que para ambos días, estas listas las obtenemos
# del segundo loop de prints que hace nuestra función grandota en "main.py".

# y aquí tenemos que decidir si usamos la medición del día uno o del día dos.

def main_func(path_0_day, t_0_day, pruebas, paths_day, escala="micro",  # checa el parámetro de "pruebas"
                hist_color="steelblue", overlay_color="darkorange", errorbar_color="0.2"):
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
    for i in range(1, len(paths_day) + 1):
        name = f"df_{i}"
        x = copia[name].iloc[:, 0]  # first (and only) column as a Series

        mu = x.mean()  # pandas mean
        var = x.var(ddof=1)  # pandas sample variance (ddof=1)
        dev = x.std(ddof=1)

        print(f"'{name}' con valor esperado, varianza, y desviación estándar (respectivamente):\n"
              f"{mu:.6f}, {var:.6f}, {dev:.6f}\n")

    # para luego hacer los histogramas, tenemos dos opciones.
    names = list(copia.keys())

    overlay = input("sobreponer densidad teórica de Poisson con medias muestrales/personalizadas? (y/n): ").strip().lower()
    while overlay not in ("y", "n"):
        overlay = input("opción inválida. escribe 'y' o 'n': ").strip().lower()

    overlay_poisson = (overlay == "y")

    custom_mus = None  # will remain None -> plotting uses sample mean

    if overlay_poisson:
        use_custom = input("usar medias ('μ') personalizadas para las densidades teóricas? (y/n): ").strip().lower()
        while use_custom not in ("y", "n"):
            use_custom = input("opción inválida. escribe 'y' o 'n': ").strip().lower()

        if use_custom == "y":
            n = len(names)
            raw = input(f"da {n} medias separadas por comas (respecto de la lista 'df_1, df_2, ...'): ").strip()

            # parse + validate
            try:
                mus = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
            except ValueError:
                print("esas medias no se pudieron pasar a números. usaremos las medias muestrales.")
                mus = None

            if mus is None or len(mus) != n:
                print(f"cantidad incorrecta de medias (recibí '{0 if mus is None else len(mus)}'; se necesitan '{n}'). "
                      f"se usarán las medias muestrales.")
            else:
                custom_mus = mus

    if overlay_poisson:
        connect = input("conectar los puntos de la densidad de Poisson mediante una línea? (y/n): ").strip().lower()
        while connect not in ("y", "n"):
            connect = input("opción inválida. escribe 'y' o 'n': ").strip().lower()
        overlay_connect = (connect == "y")
    else:
        overlay_connect = False  # doesn't matter, but keeps variable defined

    option = input("selecciona '1' para un archivo de los histogramas y '2' para archivos separados de los histogramas: ")

    while True:

        if option == "1":
            # (1) todos los histogramas en un archivo
            n_plots = len(names)
            rows, cols = best_grid(n_plots)  # para visualizar estéticamente los plots en una sóla imagen
            fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), constrained_layout=True)

            axes = np.atleast_1d(axes)

            for idx, (ax, name) in enumerate(zip(axes.ravel(), names)):
                fdp_histograma(
                    copia[name],
                    ax=ax,
                    title=name,
                    overlay_poisson=overlay_poisson,
                    overlay_connect=overlay_connect,
                    poisson_mu=(custom_mus[idx] if custom_mus is not None else None),  # ✅ ADD THIS
                    hist_color=hist_color,
                    overlay_color=overlay_color,
                    errorbar_color=errorbar_color,
                )

            for ax in axes.ravel()[len(names):]:
                fig.delaxes(ax)  # this removes the axes completely.
                                 # if you made a "2×3" grid ("6" slots) but only plotted "4" datasets,
                                 # this removes the last "2" empty slots so the figure looks clean.

            fig.savefig("histograms/all_histograms.png", bbox_inches="tight")

            plt.show()

            break

        elif option == "2":

            # (2) todos los histogramas en archivos separados
            for idx, name in enumerate(names):
                fdp_histograma(
                    copia[name],
                    title=name,
                    save_path=f"histograms/{name}.png",
                    overlay_poisson=overlay_poisson,
                    overlay_connect=overlay_connect,
                    poisson_mu=(custom_mus[idx] if custom_mus is not None else None),  # ✅ ADD THIS
                    hist_color=hist_color,
                    overlay_color=overlay_color,
                    errorbar_color=errorbar_color,
                )

            plt.show()

            break

        else:
            print("opción inválida")
            option = input(
                "selecciona '1' para un archivo de los histogramas y '2' para archivos separados de los histogramas: ")  # y así seguimos preguntando

    # los valores esperados que probamos fueron los siguientes
    # [1, 5, 10, 100]




# y a ver si funcionó:

# day_1
main_func(path_0_day_1, t_0_day_1, pruebas_day_1, paths_day_1)

# day_2
main_func(path_0_day_2, t_0_day_2, pruebas_day_2, paths_day_2)
