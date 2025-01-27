import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ----------------------------------------------------------------------------------
# CARGA DE LOS DATOS PREPROCESADOS
# ----------------------------------------------------------------------------------

# Cargar datos preparados desde task2.py
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print("Datos cargados correctamente para la regresión logística con Scikit-learn.")

# ----------------------------------------------------------------------------------
# IMPLEMENTACIÓN CON SCIKIT-LEARN
# ----------------------------------------------------------------------------------

# Crear el modelo de Regresión Logística
logistic_model = LogisticRegression(max_iter=1000, random_state=42)

# Entrenar el modelo
logistic_model.fit(X_train, y_train)

# Realizar predicciones
y_pred_sklearn = logistic_model.predict(X_test)

# ----------------------------------------------------------------------------------
# EVALUACIÓN DEL MODELO
# ----------------------------------------------------------------------------------

# Calcular métricas con Scikit-learn
precision_sklearn = precision_score(y_test, y_pred_sklearn)
recall_sklearn = recall_score(y_test, y_pred_sklearn)
f1_sklearn = f1_score(y_test, y_pred_sklearn)
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)

# Imprimir resultados
print("\nResultados del modelo de Regresión Logística con Scikit-learn:")
print(f"Precisión: {precision_sklearn:.2f}")
print(f"Recall: {recall_sklearn:.2f}")
print(f"F1-Score: {f1_sklearn:.2f}")
print(f"Matriz de confusión:\n{cm_sklearn}")

# ----------------------------------------------------------------------------------
# VISUALIZACIÓN DE LOS RESULTADOS
# ----------------------------------------------------------------------------------

# Graficar el espacio de decisión
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_sklearn, cmap='viridis', label='Predicciones')
plt.title("Regresión Logística con Scikit-learn - Clasificación de Sitios Phishing")
plt.xlabel("Longitud de la URL (Escalada)")
plt.ylabel("Proporción de dígitos en la URL (Escalada)")
plt.colorbar(label="Clase Predicha")
plt.legend()
plt.show()
