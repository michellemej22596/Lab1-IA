import pandas as pd
import numpy as np
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------------------------------------------------------------
# CARGA DE DATASET Y VISTA RÁPIDA
# ----------------------------------------------------------------------------------

path = kagglehub.dataset_download("shashwatwork/web-page-phishing-detection-dataset")
print("Path to dataset files:", path)

# Ruta al dataset (cambiar según la descarga)
dataset_path = r"C:\Users\usuario\.cache\kagglehub\datasets\shashwatwork\web-page-phishing-detection-dataset\versions\2"

# Cargar el archivo CSV
file_name = "dataset_phishing.csv"  # Cambiar si el archivo tiene otro nombre
full_path = os.path.join(dataset_path, file_name)

# Cargar los datos en un dataframe
data = pd.read_csv(full_path)

# Vista rápida de los datos
print("Primeras filas del dataset:")
print(data.head())
print("\nResumen de las columnas:")
print(data.info())

# ----------------------------------------------------------------------------------
# ENCODING NECESARIO
# ----------------------------------------------------------------------------------

# Convertir la columna 'status' en valores binarios: 1 para phishing, 0 para legítimo
print("\nRealizando encoding de la columna 'status'...")
data['status'] = (data['status'] == 'phishing').astype(int)
print("Encoding completado. Valores únicos en 'status':", data['status'].unique())

# ----------------------------------------------------------------------------------
# VERIFICACIÓN DE BALANCEO
# ----------------------------------------------------------------------------------

# Verificar si el dataset está balanceado en la columna objetivo 'status'
print("\nDistribución de la variable objetivo ('status'):")
print(data['status'].value_counts())

if data['status'].value_counts().min() == data['status'].value_counts().max():
    print("El dataset está perfectamente balanceado.")
else:
    print("El dataset NO está balanceado. Considera aplicar técnicas como SMOTE.")

# ----------------------------------------------------------------------------------
# ESCALADO DE VARIABLES
# ----------------------------------------------------------------------------------

# Seleccionar variables numéricas para escalado
features_to_scale = ['length_url', 'ratio_digits_url', 'nb_dots', 'nb_hyphens']
print("\nVariables seleccionadas para escalado:", features_to_scale)

# Escalar las variables seleccionadas
scaler = StandardScaler()
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

print("\nEscalado completado. Vista previa de las variables escaladas:")
print(data[features_to_scale].head())

# ----------------------------------------------------------------------------------
# SELECCIÓN DE VARIABLES RELEVANTES
# ----------------------------------------------------------------------------------

# Seleccionar las columnas más relevantes
selected_features = ['length_url', 'ratio_digits_url', 'nb_dots', 'nb_hyphens']
X = data[selected_features]  # Variables predictoras
y = data['status']           # Variable objetivo

print("\nVariables seleccionadas para el modelo:", selected_features)

# ----------------------------------------------------------------------------------
# SPLIT PARA ENTRENAMIENTO Y PRUEBAS
# ----------------------------------------------------------------------------------

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDivisión del dataset completada:")
print(f" - Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f" - Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")

# ----------------------------------------------------------------------------------
# MODELO BÁSICO (Random Forest)
# ----------------------------------------------------------------------------------

# Entrenar un modelo básico como ejemplo
print("\nEntrenando un modelo Random Forest básico...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
print("\nResultados del modelo:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------------------------------
# DEFINICIÓN DE LA MÉTRICA DE DESEMPEÑO
# ----------------------------------------------------------------------------------

# Elegimos F1-Score como métrica principal
# Justificación:
# - F1-Score balancea precisión y recall.
# - Es ideal para problemas donde los falsos negativos (no identificar phishing) seria costoso.
# - Aunque el dataset está balanceado, queremos minimizar tanto falsos negativos como falsos positivos.

print("\nMétrica de desempeño principal: F1-Score")
print("Reporte de clasificación (incluye F1-Score):")
print(classification_report(y_test, y_pred))

