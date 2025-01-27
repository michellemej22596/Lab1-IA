##ESTE ARCHIVO ESTA SEPARADO DEL TASK2 UNICAMENTE POR ORGANIZACION, PERO PARA CORRERLO DEBE SER CORRIDO LUEGO DE TRANSFORMAR LOS DATOS, UTILICE MEJOR EL NOTEBOOK DE JUPYTER O COPIE Y PEGUE ESTE CODIGO EN EL ARCHIVO TASK2.PY

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------------------------
# IMPLEMENTACIÓN DEL ALGORITMO KNN DESDE CERO
# ----------------------------------------------------------------------------------

# Función para calcular la distancia euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Clase para implementar KNN
class KNN:
    def __init__(self, k=3):
        self.k = k  # Número de vecinos

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)

    def _predict(self, x):
        # Calcular la distancia de x a todos los puntos de entrenamiento
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Obtener los índices de los k vecinos más cercanos
        k_indices = np.argsort(distances)[:self.k]
        # Obtener las clases de los k vecinos
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Votación de la clase mayoritaria
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# ----------------------------------------------------------------------------------
# PREPARACIÓN DE LOS DATOS
# ----------------------------------------------------------------------------------

# Selección de dos variables para graficar
selected_features = ['length_url', 'ratio_digits_url']
X = data[selected_features].values
y = data['status'].values

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------------------
# ENTRENAMIENTO Y PREDICCIÓN
# ----------------------------------------------------------------------------------

# Instancia del modelo KNN con k=5
knn = KNN(k=5)
knn.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = knn.predict(X_test)

# ----------------------------------------------------------------------------------
# EVALUACIÓN DEL MODELO: PRECISIÓN, RECALL Y F1-SCORE
# ----------------------------------------------------------------------------------

def calculate_f1(y_true, y_pred):
    # Matriz de confusión: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # Cálculo de precisión y recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Cálculo de F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# Evaluar precisión, recall y F1-Score para el modelo implementado desde cero
precision_manual, recall_manual, f1_manual = calculate_f1(y_test, y_pred)

print("\nEvaluación del modelo KNN desde cero:")
print(f"Precisión: {precision_manual:.2f}")
print(f"Recall: {recall_manual:.2f}")
print(f"F1-Score: {f1_manual:.2f}")


# ----------------------------------------------------------------------------------
# VISUALIZACIÓN DE LOS RESULTADOS
# ----------------------------------------------------------------------------------

# Graficar el espacio de decisión
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', label='Predicciones')
plt.title("KNN - Clasificación de Sitios Phishing")
plt.xlabel("Longitud de la URL")
plt.ylabel("Proporción de dígitos en la URL")
plt.colorbar(label="Clase")
plt.legend()
plt.show()
