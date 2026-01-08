import pandas as pd

def build_original_y_copia(paths, encoding='utf-8'):
    '''
    crea el dataframe y su copia para manejar en el archivo principal
    (indexando con números naturales)
    '''

    original = {f"df_{i+1}": pd.read_csv(p, encoding=encoding)  # aquí es donde indexamos con naturales
                for i, p in enumerate(paths)}
    copia = {name: df.copy(deep=True) for name, df in original.items()}
    return original, copia


