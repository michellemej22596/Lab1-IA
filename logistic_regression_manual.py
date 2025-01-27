import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ----------------------------------------------------------------------------------
# CARGA DE LOS DATOS PREPROCESADOS
# ----------------------------------------------------------------------------------

# Cargar datos preparados desde task2.py
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Datos cargados correctamente para la regresión logística manual.")

# ----------------------------------------------------------------------------------
# IMPLEMENTACIÓN DESDE CERO: REGRESIÓN LOGÍSTICA
# ----------------------------------------------------------------------------------

# Función sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Función de pérdida (entropía cruzada)
def compute_loss(y, y_pred):
    m = len(y)
    loss = -1 / m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return loss

# Gradiente descendente para Regresión Logística
def logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    losses = []

    for epoch in range(epochs):
        # Predicción
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        # Cálculo de la pérdida
        loss = compute_loss(y, y_pred)
        losses.append(loss)

        # Gradientes
        dw = 1 / m * np.dot(X.T, (y_pred - y))
        db = 1 / m * np.sum(y_pred - y)

        # Actualizar pesos
        weights -= lr * dw
        bias -= lr * db

        # Imprimir pérdida cada 100 épocas
        if epoch % 100 == 0:
            print(f"Época {epoch}, Pérdida: {loss:.4f}")

    return weights, bias, losses

# ----------------------------------------------------------------------------------
# ENTRENAMIENTO
# ----------------------------------------------------------------------------------

# Entrenar el modelo de Regresión Logística
lr = 0.1
epochs = 1000
weights, bias, losses = logistic_regression(X_train, y_train, lr=lr, epochs=epochs)

# ----------------------------------------------------------------------------------
# EVALUACIÓN
# ----------------------------------------------------------------------------------

# Predicción
linear_model = np.dot(X_test, weights) + bias
y_pred = sigmoid(linear_model)
y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]

# Cálculo de métricas manuales
cm = confusion_matrix(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

print("\nResultados:")
print(f"Precisión: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Matriz de confusión:\n{cm}")

# ----------------------------------------------------------------------------------
# GRAFICAR RESULTADOS
# ----------------------------------------------------------------------------------

# Visualización de las predicciones
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_class, cmap='viridis', label='Predicciones')
plt.title("Regresión Logística - Clasificación de Sitios Phishing")
plt.xlabel("Longitud de la URL (Escalada)")
plt.ylabel("Proporción de dígitos en la URL (Escalada)")
plt.colorbar(label="Clase Predicha")
plt.legend()
plt.show()

# Graficar la pérdida
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), losses)
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.show()
