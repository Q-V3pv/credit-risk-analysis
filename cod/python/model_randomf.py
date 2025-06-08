from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class CreditRiskModel:
    """
    Clase para entrenar, predecir y evaluar un modelo de riesgo crediticio
    usando Random Forest.
    """

    def __init__(self, n_estimators=100, test_size=0.2, random_state=42):
        """
        Inicializa el modelo con los hiperparámetros dados.
        
        Args:
            n_estimators (int): Número de árboles en el bosque.
            test_size (float): Proporción de datos para prueba.
            random_state (int): Semilla para reproducibilidad.
        """
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def split_data(self, X, y):
        """
        Divide los datos en sets de entrenamiento y prueba.
        
        Args:
            X (DataFrame o array): Características.
            y (Series o array): Etiquetas.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y)

    def train(self):
        """Entrena el modelo con los datos de entrenamiento."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X=None):
        """
        Genera predicciones.
        
        Args:
            X (DataFrame o array): Datos para predecir.
                                   Si es None, predice en X_test.
        
        Returns:
            np.array: Predicciones.
        """
        if X is None:
            X = self.X_test
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def evaluate(self):
        """
        Imprime métricas de evaluación 
        """
        if self.y_pred is None:
            raise ValueError("Hay que generar predicciones con el método predict()")
        print("Accuracy:", accuracy_score(self.y_test, self.y_pred))
        print("\nMatriz de confusión:")
        print(confusion_matrix(self.y_test, self.y_pred))
        print("\nReporte de clasificación:")
        print(classification_report(self.y_test, self.y_pred))
