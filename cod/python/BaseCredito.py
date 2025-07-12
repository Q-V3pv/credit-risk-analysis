import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class BaseCredito:

    def __init__(self, ruta = "C:/Users/josep/Documents/credit-risk-analysis/data/credit_risk_dataset.csv"):
        """Carga los datos desde un archivo CSV.

        Parámetros
        ----------
        ruta : str
            Ruta del archivo CSV.
        
        Retorna
        -------
        None
        """
        self._ruta = ruta
        self._df = pd.read_csv(ruta)
        
    @property
    def ruta_archivo(self):
        """Obtiene la ruta del archivo CSV.
        
        Parametros 
        ----------
        None

        Retorna
        -------
        str
            Ruta del archivo utilizado para cargar los datos.
        """
        return self._ruta

    @ruta_archivo.setter
    def ruta_archivo(self, nueva_ruta):
        """Asigna una nueva ruta al archivo CSV.

        Parámetros
        ----------
        nueva_ruta : str
            Nueva ubicación del archivo CSV.

        Retorna
        -------
        None
        """
        self._ruta = nueva_ruta
        self._df = pd.read_csv(nueva_ruta)


    @property
    def df(self):
        """Getter del DataFrame.

        Parámetros
        ----------
        None

        Retorna
        -------
        pd.DataFrame
            Datos de la base de vuelos.
        """
        return self._df

    @df.setter
    def df(self, nuevo_df):
        """Setter para asignar un nuevo DataFrame.

        Parámetros
        ----------
        nuevo_df : pd.DataFrame
            Nuevo DataFrame a asignar.

        Retorna
        -------
        None
        """
        self._df = nuevo_df

    def ver_info(self):
        """Muestra información general del DataFrame.

        Parámetros
        ----------
        None

        Retorna
        -------
        None
        """
        print(self._df.info())
        
    def distribucion_clasi(self):
        sns.countplot(x="loan_grade", data = self.df)
        plt.title("Distribución de la variable objetivo")
        plt.xlabel("Calificación")
        plt.ylabel("Cantidad")
        plt.show()

    def resumen(self):
        """Muestra un resumen estadístico de las variables numéricas.

        Parámetros
        ----------
        None

        Retorna
        -------
        None
        """
        print(self._df.describe())
    

    def __str__(self):
        """Representación en string de la clase.

        Parámetros:
        ----------
        None

        Retorna:
        -------
        str
            Resumen de filas, columnas y nombres de columnas.
        """
        return f"BaseCredito con {self._df.shape[0]} filas y {self._df.shape[1]} columnas.\nColumnas: {list(self._df.columns)}"

    