# Lab1-IA
Michelle Mejía 22596 y  Silvia Illescas 22376

## Clasificación de Sitios Web de Phishing
Este repositorio contiene la implementación y análisis de modelos para la clasificación de sitios web de phishing. El objetivo principal es identificar sitios sospechosos basándonos en las características de las URLs, utilizando tanto implementaciones desde cero como herramientas de aprendizaje automático preexistentes.

## Contenido
### Preprocesamiento de Datos:

Encoding de variables categóricas.
Verificación del balanceo del dataset.
Escalado de características numéricas.
Selección de variables relevantes.
Modelos Implementados:

Regresión Lineal (Task 2.1):
Implementación desde cero utilizando gradiente descendente.
Visualización de los grupos en un plano cartesiano.
K-Nearest Neighbors (Task 2.2):
Implementación desde cero con cálculo de distancia euclidiana y votación.
Implementación usando Scikit-learn para comparación.
Evaluación utilizando métricas como precisión, recall y F1-Score.
Análisis de Resultados:

Comparación entre implementaciones manuales y Scikit-learn.
Discusión sobre las métricas de desempeño seleccionadas.
Visualización de los resultados en gráficos 2D.
Requisitos

Este proyecto fue desarrollado en Python 3.11. Se requiere instalar las siguientes librerías:

bash
Copiar
Editar
pip install pandas numpy matplotlib scikit-learn imbalanced-learn kagglehub


### Estructura del Proyecto
task2.py: Script principal que incluye todo el flujo desde el preprocesamiento hasta la evaluación de los modelos.
El resto de scripts lo debes pegar en task2 para correr.

README.md: Este archivo de documentación.
Dataset: Asegúrate de descargar el dataset desde Kaggle y colocarlo en la ruta correspondiente.

Guía de Uso
1. Preprocesamiento
El código incluye pasos clave como:

Conversión de variables categóricas (status) a formato binario.
Escalado de variables como length_url y ratio_digits_url para asegurar un correcto desempeño de los modelos basados en distancias.
Verificación del balanceo de las clases y preparación de los conjuntos de entrenamiento y prueba.
2. Modelos
Regresión Logística
Implementada desde cero usando gradiente descendente para predecir si un sitio es phishing o legítimo.
Visualización de las predicciones en un plano cartesiano utilizando dos variables seleccionadas.
K-Nearest Neighbors (KNN)
Desde cero:
Calcula distancias euclidianas entre puntos.
Realiza votación de los
𝑘
k-vecinos más cercanos.
Con Scikit-learn:
Optimizado para calcular distancias de forma eficiente.
Compara los resultados con la implementación manual.
3. Evaluación de Modelos
Los modelos fueron evaluados utilizando las siguientes métricas:

Precisión: Qué tan precisas son las predicciones positivas.
Recall: Qué proporción de sitios phishing reales fueron correctamente identificados.
F1-Score: Métrica que balancea precisión y recall.


Resultados
KNN Desde Cero
Precisión: ~67%
Recall: ~65%
F1-Score: ~66%
KNN con Scikit-learn
Precisión: 73%
Recall: 72%
F1-Score: 72%
El modelo con Scikit-learn mostró un mejor desempeño debido a optimizaciones internas como el uso de estructuras de datos más eficientes para calcular distancias.

Visualizaciones
Se incluyen gráficos 2D para representar las predicciones en función de dos características seleccionadas.
