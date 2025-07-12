import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight


class PrediccionEstado:
    
    def __init__(self, base_limpia):
        """Inicializa la clase con una base ya limpia.

        Parámetros
        ----------
        base_limpia : object
           Objeto BaseLimpia.

        Retorna
        -------
        None
        """
        self.df = base_limpia.df
        self._base_limpia = base_limpia

    @property
    def base_limpia(self):
        """Obtiene el objeto BaseLimpia.
        
        Parámetros
        ----------
        None

        Retorna
        -------
        BaseLimpia
            Instancia con los datos limpios.
        """
        return self._base_limpia

    @base_limpia.setter
    def base_limpia(self, nueva_base):
        """Asigna un nuevo objeto BaseLimpia a la clase.

        Parámetros
        ----------
        nueva_base : BaseLimpia
            Instancia nueva que contiene datos.

        Retorna
        -------
        None
        """
        self._base_limpia = nueva_base
        self.df = nueva_base.df

    @property
    def datos(self):
        """Obtiene el DataFrame.
        
        Parámetros 
        ----------
        None

        Retorna
        -------
        pd.DataFrame
            DataFrame con los datos usados por el modelo.
        """
        return self.df

    @datos.setter
    def datos(self, nuevo_df):
        """Asigna un nuevo DataFrame al objeto.

        Parámetros
        ----------
        nuevo_df : pd.DataFrame
            DataFrame que reemplazará al actual.
        
        Retorna
        -------
        None
        """
        self.df = nuevo_df

    @property
    def modelo(self):
        """Obtiene el modelo entrenado.

        Retorna
        -------
        object
            Modelo predictivo ya entrenado.
        """
        return self.model

    @modelo.setter
    def modelo(self, nuevo_modelo):
        """Asigna un nuevo modelo al objeto.

        Parámetros
        ----------
        nuevo_modelo : object
            Modelo ya entrenado.
        
        Retorna
        -------
        None
        """
        self.model = nuevo_modelo

    def preparar(self, objetivo = "calificacion_simplificada", t_size = 0.2, r_state = 42):
        """Prepara los datos antes de entrenar el modelo.


        Parámetros
        ----------
        objetivo : str
            Variable que se va a predecir.
        t_size : float
            Proporción del conjunto de prueba.
        r_state : int
            Semilla aleatoria.

        Retorna
        -------
        tuple
            X_train, X_test, y_train, y_test
        """
        df_dummy = pd.get_dummies(self.df, columns = [
            "tenencia_vivienda",
            "proposito_prestamo"
            ],
        drop_first=True
        )
     
        X = df_dummy.drop(columns=[objetivo])
        y = df_dummy[objetivo]
     
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=t_size, random_state=r_state, stratify=y
        )
        sm = SMOTE(random_state = r_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test

    
    def entrenar(self, X_train, y_train, tipo="r"):
        """Entrena el modelo especificado: Random Forest, Gradient Boosting o XGBoost.
    
        Parámetros
        ----------
        X_train : pd.DataFrame
            Variables predictoras de entrenamiento.
        y_train : pd.Series
            Variable objetivo de entrenamiento.
        tipo : str
            Tipo de modelo: "r" = Random Forest, "g" = Gradient Boosting, "x" = XGBoost
    
        Retorna
        -------
        None
        """
        if tipo == "r":
            self.model = RandomForestClassifier(class_weight = "balanced", random_state = 42)
            self.model.fit(X_train, y_train)
            
        elif tipo == "g":
            self.model = GradientBoostingClassifier(random_state = 42)
            self.model.fit(X_train, y_train)
    
        elif tipo == "x":
            pesos = compute_sample_weight(class_weight = "balanced", y = y_train)
            self.model = XGBClassifier(use_label_encoder = False, eval_metric = "mlogloss", random_state = 42)
            self.model.fit(X_train, y_train, sample_weight = pesos)



    def evaluar(self, X_test, y_test):
        """Evalúa el modelo usando métricas de clasificación.
    
        Parámetros
        ----------
        X_test : pd.DataFrame
            Datos de prueba.
        y_test : pd.Series
            Valores reales.
    
        Retorna
        -------
        dict
            Accuracy, matriz de confusión y reporte de clasificación.
        """
        predicciones = self.model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, predicciones),
            "matriz_confusion": confusion_matrix(y_test, predicciones).tolist(),
            "reporte_clasificacion": classification_report(y_test, predicciones, output_dict = True)
        }
    
    def importancia_variables(self, X_train):
        """Muestra un gráfico de importancia de las variables.

        Parámetros
        ----------
        X_train : pd.DataFrame
            Datos de entrenamiento.

        Retorna
        -------
        None
        """
        importancias = pd.Series(self.model.feature_importances_, index = X_train.columns)
        importancias.nlargest(15).plot(kind = 'barh', figsize = (10, 6))
        plt.title("Importancia de las variables")
        plt.xlabel("Importancia")
        plt.show()

    def grafico_residuos(self, X_test, y_test):
        """Muestra la matriz de confusión del modelo.
    
        Parámetros
        ----------
        X_test : pd.DataFrame
        y_test : pd.Series
    
        Retorna
        -------
        None
        """
        pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap = 'Blues')
        plt.title("Matriz de confusión")
        plt.grid(False)
        plt.show()
        
    def comparar_modelos(self, X_train, X_test, y_train, y_test):
        #Obtenido de https://scikit-learn.org/stable/modules/ensemble.html
        """Compara modelos clasificadores y devuelve sus métricas.
    
        Parámetros
        ----------
        X_train, X_test : pd.DataFrame
        y_train, y_test : pd.Series
    
        Retorna
        -------
        pd.DataFrame
            Métricas para cada modelo.
        """
        resultados = {}

        modelos = {
            "Random Forest": RandomForestClassifier(class_weight = "balanced", random_state = 42),
            "Gradient Boosting": GradientBoostingClassifier(random_state = 42),
            "XGBoost": XGBClassifier(use_label_encoder = False, eval_metric = "mlogloss", random_state = 42)
        }
     
        # Crear mapeo para convertir etiquetas categóricas a numéricas
        etiquetas_unicas = sorted(y_train.unique())
        mapeo_clases = {clase: i for i, clase in enumerate(etiquetas_unicas)}
     
        # Convertir etiquetas para XGBoost
        y_train_num = y_train.map(mapeo_clases)
     
        pesos = compute_sample_weight(class_weight="balanced", y = y_train_num)
     
        for nombre, modelo in modelos.items():
            
            if nombre == "XGBoost":
                modelo.fit(X_train, y_train_num, sample_weight=pesos)
                pred = modelo.predict(X_test)
                # Convertir predicciones numéricas a etiquetas originales para evaluación
                pred_labels = pd.Series(pred).map({v: k for k, v in mapeo_clases.items()})
                resultados[nombre] = {
                    "Accuracy": accuracy_score(y_test, pred_labels),
                    "Precision": classification_report(y_test, pred_labels, output_dict = True)["weighted avg"]["precision"],
                    "Recall": classification_report(y_test, pred_labels, output_dict = True)["weighted avg"]["recall"],
                    "F1-Score": classification_report(y_test, pred_labels, output_dict = True)["weighted avg"]["f1-score"]
                }
            else:
                modelo.fit(X_train, y_train)
                pred = modelo.predict(X_test)
                resultados[nombre] = {
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": classification_report(y_test, pred, output_dict = True)["weighted avg"]["precision"],
                    "Recall": classification_report(y_test, pred, output_dict = True)["weighted avg"]["recall"],
                    "F1-Score": classification_report(y_test, pred, output_dict = True)["weighted avg"]["f1-score"]
                }
     
        return pd.DataFrame(resultados)

    def distribucion_errores(self, X_test, y_test):
        """Muestra la comparación de clases reales vs. predichas.
    
        Parámetros
        ----------
        X_test : pd.DataFrame
        y_test : pd.Series
    
        Retorna
        -------
        None
        """
        pred = self.model.predict(X_test)
        df_errores = pd.DataFrame({"Real": y_test, "Predicho": pred})
        conteo = df_errores.groupby(["Real", "Predicho"]).size().unstack().fillna(0)
        
        conteo.plot(kind="bar", stacked = True, colormap = "viridis", figsize = (8, 5))
        plt.title("Distribución de clasificaciones reales vs. predichas")
        plt.xlabel("Clase real")
        plt.ylabel("Cantidad")
        plt.xticks(rotation = 0)
        plt.tight_layout()
        plt.show()


    def __str__(self):
        """Representación de texto del estado actual.
        
        Parámetros 
        ----------
        None

        Retorna
        -------
        str
            Descripción general de la clase.
        """
        modelo_tipo = type(self.model).__name__
        return f"PrediccionCredito(df con {len(self.df)} registros, modelo: {modelo_tipo})"
