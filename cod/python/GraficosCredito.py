import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from BaseLimpia import BaseLimpia

class GraficosCredito(BaseLimpia):

    def __init__(self, ruta = "C:/Users/josep/Documents/credit-risk-analysis/data/credit_risk_dataset.csv"):
        super().__init__(ruta)

    @property
    def datos(self):
        """Obtiene el DataFrame procesado desde la clase base.
    
        Retorna
        -------
        pd.DataFrame
            Conjunto de datos limpiado y cargado en memoria.
        """
        return self.df

    @datos.setter
    def datos(self, nuevo_df):
        """Asigna un nuevo DataFrame al atributo de datos.
    
        Parámetros
        ----------
        nuevo_df : pd.DataFrame
            Nuevo DataFrame para ser usado por la clase.
        
        Retorna
        -------
        None
        """
        self.df = nuevo_df

    def distribucion_tasa_interes(self):
        """Grafica la distribución de la tasa de interés de los préstamos.
        
        Parámetros
        ----------
        None

        Retorna
        -------
        None
        """
        df_plot = self.df[self.df["tasa_interes"].between(0, 40)]
        plt.figure(figsize = (10, 6))
        sns.histplot(df_plot["tasa_interes"], bins = 50, kde = True, color = "seagreen")
        plt.axvline(df_plot["tasa_interes"].mean(), color = 'darkorange', linestyle = '--', label = 'Media')
        plt.title("Distribución de tasa de interés")
        plt.xlabel("Tasa de interés (%)")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def boxplot_por_calificacion(self):
        """Genera un diagrama de caja para comparar la tasa de interés por calificación del préstamo.
      
        Parámetros
        ----------
        None
        
        Retorna
        -------
        None
        """
        plt.figure(figsize = (12, 6))
        orden = self.df["calificacion_prestamo"].value_counts().sort_index().index
        sns.boxplot(
            x = "calificacion_prestamo",
            y = "tasa_interes",
            data = self.df,
            order = orden,
            palette = "coolwarm",
            width = 0.6,
            fliersize = 2,
            linewidth = 1.5
        )
        plt.title("Tasa de interés por calificación del préstamo", fontsize = 14, fontweight = "bold")
        plt.ylabel("Tasa de interés (%)")
        plt.xlabel("Calificación")
        plt.grid(axis = "y", linestyle = "--", alpha = 0.5)
        plt.tight_layout()
        plt.show()

    def estado_por_rango_edad(self):
        """Muestra un gráfico de barras del estado del préstamo según rango de edad
        
        Parámetros 
        ----------
        None
        
        Retorna
        -------
        None
        """
        df_plot = self.df.copy()
        df_plot["rango_edad"] = pd.cut(
            df_plot["edad_persona"],
            bins = [17, 25, 35, 45, 55, 65, 100],
            labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
        )
        conteo = df_plot.groupby(["rango_edad", "estado_prestamo"]).size().unstack(fill_value = 0)
        conteo.plot(kind = "bar", stacked = True, figsize = (10, 6), colormap = "Set2")
        plt.title("Estado del préstamo por rango de edad", fontsize = 14, fontweight = "bold")
        plt.xlabel("Rango de edad")
        plt.ylabel("Cantidad de personas")
        plt.legend(title = "Estado del préstamo", labels = ["No incumple", "Incumple"])
        plt.tight_layout()
        plt.show()

    def ingreso_por_proposito(self):
        """Muestra un gráfico de violín del ingreso anual por propósito del préstamo.
         
        Parámetros 
        ----------
        None 
        
        Retorna
        -------
        None
        """
        plt.figure(figsize = (12, 6))
        orden = self.df["proposito_prestamo"].value_counts().index
        sns.violinplot(
            x = "proposito_prestamo",
            y = "ingreso_anual",
            data = self.df,
            order = orden,
            palette = "Set3",
            cut = 0,
            scale = "width"
        )
        plt.title("Ingreso anual por propósito del préstamo", fontsize = 14, fontweight = "bold")
        plt.ylabel("Ingreso anual")
        plt.xlabel("Propósito del préstamo")
        plt.xticks(rotation = 45, ha = "right")
        plt.tight_layout()
        plt.show()

    def __str__(self):
        """Genera una representación del estado de la clase.
        
        Parámetros 
        ----------
        None
    
        Retorna
        -------
        str
            Texto con resumen de la clase.
        """
        return f"GraficosCredito con {self.df.shape[0]} filas y {self.df.shape[1]} columnas."
