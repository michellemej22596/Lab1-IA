##ESTE ARCHIVO ESTA SEPARADO DEL TASK2 UNICAMENTE POR ORGANIZACION, PERO PARA CORRERLO DEBE SER CORRIDO LUEGO DE TRANSFORMAR LOS DATOS, UTILICE MEJOR EL NOTEBOOK DE JUPYTER O COPIE Y PEGUE ESTE CODIGO EN EL ARCHIVO TASK2.PY

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# IMPLEMENTACIÓN DE KNN CON SCIKIT-LEARN
# ----------------------------------------------------------------------------------

# Crear el modelo KNN con k=5
knn_sklearn = KNeighborsClassifier(n_neighbors=5)

# Entrenar el modelo
knn_sklearn.fit(X_train, y_train)

# Realizar predicciones
y_pred_sklearn = knn_sklearn.predict(X_test)

# ----------------------------------------------------------------------------------
# EVALUACIÓN DEL MODELO: SCIKIT-LEARN
# ----------------------------------------------------------------------------------

# Precisión, Recall y F1-Score usando Scikit-learn
precision_sklearn = precision_score(y_test, y_pred_sklearn)
recall_sklearn = recall_score(y_test, y_pred_sklearn)
f1_sklearn = f1_score(y_test, y_pred_sklearn)

print("\nEvaluación del modelo KNN con Scikit-learn:")
print(f"Precisión: {precision_sklearn:.2f}")
print(f"Recall: {recall_sklearn:.2f}")
print(f"F1-Score: {f1_sklearn:.2f}")

# ----------------------------------------------------------------------------------
# VISUALIZACIÓN DE LOS RESULTADOS
# ----------------------------------------------------------------------------------

# Graficar el espacio de decisión
plt.figure(figsize=(10, 6))
plt.scatter(X_test.to_numpy()[:, 0], X_test.to_numpy()[:, 1], c=y_pred_sklearn, cmap='viridis', label='Predicciones')
plt.title("KNN con Scikit-learn - Clasificación de Sitios Phishing")
plt.xlabel("Longitud de la URL")
plt.ylabel("Proporción de dígitos en la URL")
plt.colorbar(label="Clase")
plt.legend()
plt.show()
