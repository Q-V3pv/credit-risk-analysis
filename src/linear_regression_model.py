from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionModel:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

    def split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X=None):
        if X is None:
            X = self.X_test
        self.y_pred = self.model.predict(X)
        return self.y_pred

    def evaluate(self):
        if self.y_pred is None:
            raise ValueError("Se necesita generar predicciones con el método predict().")
        mse = mean_squared_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        print("Evaluación del modelo de regresión lineal:")
        print(f"  - MSE: {mse:.4f}")
        print(f"  - R²: {r2:.4f}")

