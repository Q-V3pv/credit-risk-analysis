import pandas as pd

def load_data(path):
    """
    Cargar dataset 

    Parámetros:
        path (str): Ruta al archivo CSV.

    Retorna:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    try:
        df = pd.read_csv(path)
        print(f"Datos cargados correctamente desde: {path}")
        return df
    except FileNotFoundError:
        print(f"Archivo no encontrado en: {path}")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
