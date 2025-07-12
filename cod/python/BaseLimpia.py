from BaseCredito import BaseCredito
class BaseLimpia(BaseCredito):
    
    def __init__(self, ruta = "C:/Users/josep/Documents/credit-risk-analysis/data/credit_risk_dataset.csv"):
        """Inicializa la clase, renombra , limpia.

        Parámetros
        ----------
        ruta : str
            Ruta del archivo.

        Retorna
        -------
        None
        """
        super().__init__(ruta)
        self.renombrar() 
        self.limpiar()
    
    @property
    def datos(self):
        """Obtiene el DataFrame ya renombrado, filtrado y transformado.
        
        Parametros 
        ----------
        None
    
        Retorna
        -------
        pd.DataFrame
            Conjunto de datos limpios.
        """
        return self.df
    
    @datos.setter
    def datos(self, nuevo_df):
        """Asigna un nuevo DataFrame a la clase.
    
        Parámetros
        ----------
        nuevo_df : pd.DataFrame
            Nuevo DataFrame.
        
        Retorna
        -------
        None
        """
        self.df = nuevo_df
        
    def limpiar(self):
        """Limpia y filtra el dataset de riesgo crediticio.
    
        Parámetros
        ----------
        None
    
        Retorna
        -------
        None
        """
        df = self.df

        # Filtrado básico
        df = df[
        (df["ingreso_anual"] > 0) &
        (df["porcentaje_ingreso"] > 0) &
        (~df["calificacion_prestamo"].isna())
        ]
         
        # Mapeo de categorías
        mapeo = {
        "A": "Alta",
        "B": "Alta",
        "C": "Baja",
        "D": "Baja",
        "E": "Baja",
        "F": "Baja",
        "G": "Baja"
        }
        df["calificacion_simplificada"] = df["calificacion_prestamo"].map(mapeo)
         
        # Columnas que introducen fuga
        cols_innecesarias = ["estado_prestamo", "tasa_interes", "monto_prestamo", "porcentaje_ingreso","calificacion_prestamo", "historial_impago"]
        df_1 = df.drop(columns=[c for c in cols_innecesarias if c in df.columns])
         
        # Filas con NA en variables clave
        df_1 = df_1.dropna(subset=[
        "edad_persona", "ingreso_anual", "tenencia_vivienda",
        "antiguedad_empleo", "proposito_prestamo", "calificacion_simplificada"
        ])
         
        self.df = df_1

       
    def renombrar(self):
        """Renombra las columnas del conjunto de datos de riesgo crediticio.
        
        Parámetros
        ----------
        None
        
        Retorna
        -------
        None
        """
        columnas = {
            "person_age": "edad_persona",
            "person_income": "ingreso_anual",
            "person_home_ownership": "tenencia_vivienda",
            "person_emp_length": "antiguedad_empleo",
            "loan_intent": "proposito_prestamo",
            "loan_grade": "calificacion_prestamo",
            "loan_amnt": "monto_prestamo",
            "loan_int_rate": "tasa_interes",
            "loan_status": "estado_prestamo",
            "loan_percent_income": "porcentaje_ingreso",
            "cb_person_default_on_file": "historial_impago",
            "cb_preson_cred_hist_length": "longitud_historial_crediticio"
        }
        self.df.rename(columns = columnas, inplace = True)
        
    
    def __str__(self):
        """Genera un resumen  del objeto.
        
        Parametros
        ----------
        None
    
        Retorna
        -------
        str
            Texto resumen de la clase.
        """
        return f"BaseLimpia con {self.df.shape[0]} filas y {self.df.shape[1]} columnas tras limpieza y codificación."
     