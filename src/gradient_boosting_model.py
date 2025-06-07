from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class GradientBoostingModel:
    def __init__(self, param_grid=None):
        self.model = GradientBoostingClassifier()
        self.param_grid = param_grid or {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
        self.grid_search = None

    def train(self, X_train, y_train):
        self.grid_search = GridSearchCV(self.model, self.param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        self.grid_search.fit(X_train, y_train)
        return self.grid_search.best_estimator_

    def predict(self, X_test):
        if self.grid_search is None:
            raise ValueError("El modelo no ha sido entrenado aún.")
        return self.grid_search.best_estimator_.predict(X_test)

    def evaluate(self, X_test, y_test):
        best_model = self.grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"\nAccuracy: {acc}")
        print("\nMatriz de confusión:")
        print(cm)
        print("\nReporte de clasificación:")
        print(report)

        return best_model

