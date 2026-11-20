# ReciClass ♻️
### Clasificador inteligente de residuos mediante visión artificial

ReciClass es una aplicación web que permite identificar y clasificar residuos a partir de una imagen. El usuario carga una fotografía y el sistema analiza sus características visuales para predecir a cuál de las nueve categorías del modelo pertenece.

El proyecto fue desarrollado como Trabajo Final Integrador del Seminario de Actualización 2026 de la Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial.

## Objetivo

Desarrollar una solución de clasificación de residuos que integre un modelo de Machine Learning propio con una API REST y una interfaz web, aplicando un flujo completo de desarrollo: preparación de datos, extracción de características, entrenamiento, evaluación, serialización e integración.

## Categorías disponibles

El modelo clasifica imágenes en nueve categorías:

1. Cardboard — Cartón
2. Food Organics — Residuos orgánicos
3. Glass — Vidrio
4. Metal — Metal
5. Miscellaneous Trash — Residuos varios
6. Paper — Papel
7. Plastic — Plástico
8. Textile Trash — Residuos textiles
9. Vegetation — Vegetación

## ¿Cómo funciona?

1. El usuario selecciona una imagen desde la interfaz web.
2. El frontend procesa la imagen y la envía a la API mediante una solicitud HTTP.
3. FastAPI recibe la imagen, valida el contenido y la prepara para el análisis.
4. El extractor obtiene características visuales utilizando HOG, LBP y un histograma de color HSV.
5. El modelo SVM realiza la clasificación.
6. La API devuelve la categoría predicha y las probabilidades estimadas para las nueve clases.
7. El frontend presenta el resultado al usuario.

## Tecnologías utilizadas

| Componente | Tecnología |
|---|---|
| Lenguaje | Python |
| Machine Learning | Scikit-learn |
| Clasificador | Support Vector Machine (SVM) |
| Extracción de características | HOG, LBP y HSV |
| Procesamiento de imágenes | Pillow, NumPy, scikit-image |
| Serialización | Joblib |
| Backend | FastAPI |
| Servidor ASGI | Uvicorn |
| Validación de datos | Pydantic |
| Frontend | HTML, CSS y JavaScript |
| Control de versiones | Git y GitHub |

## Dataset

Se utilizó el dataset RealWaste, compuesto por 4.752 imágenes distribuidas en nueve categorías.

El conjunto de datos fue explorado y preparado para el entrenamiento y la evaluación del modelo. Se utilizaron particiones estratificadas:

- 70 % entrenamiento.
- 15 % validación.
- 15 % prueba.

Las imágenes originales del dataset no se incluyen en este repositorio.

## Modelo de Machine Learning

Se compararon tres algoritmos:

- Support Vector Machine (SVM).
- Random Forest.
- Extra Trees.

La selección se realizó utilizando el F1 macro sobre el conjunto de validación. El modelo seleccionado fue SVM con kernel RBF, `C=10`, `gamma=scale` y ponderación balanceada de clases.

El modelo final se reentrenó con los conjuntos de entrenamiento y validación, y se evaluó sobre el conjunto de prueba.

### Resultados del modelo seleccionado

| Métrica | Resultado |
|---|---:|
| Accuracy en prueba | 57,22 % |
| F1 macro en prueba | 55,01 % |
| Accuracy en validación | 59,05 % |
| F1 macro en validación | 58,16 % |

Estas métricas corresponden a la evaluación realizada sobre la partición de prueba de RealWaste. No representan una garantía de rendimiento sobre fotografías externas.

## Arquitectura de la aplicación

La solución está organizada en tres componentes principales:

- **Frontend:** interfaz de usuario desarrollada con HTML, CSS y JavaScript.
- **Backend:** API REST construida con FastAPI, responsable de recibir las imágenes, validarlas y ejecutar las predicciones.
- **Modelo:** clasificador entrenado y serializado mediante Joblib.

## API

La API dispone de los siguientes endpoints:

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Verifica el estado de la API y la disponibilidad del modelo. |
| POST | `/predecir` | Recibe una imagen codificada en Base64 y devuelve la clasificación. |

La API utiliza validaciones de entrada y respuestas de error HTTP para manejar solicitudes incorrectas.

## Ejecución local

### 1. Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd ReciClass_backend_frontend
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

Activarlo en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install -r app/backend/requirements.txt
```

### 4. Ejecutar la API

```bash
cd app/backend
uvicorn main:app --reload
```

### 5. Acceder a la aplicación

Abrir en el navegador:

- Aplicación web: `http://127.0.0.1:8000/`
- Documentación interactiva: `http://127.0.0.1:8000/docs`
- Estado de la API: `http://127.0.0.1:8000/health`

## Despliegue

La aplicación está preparada para publicarse como servicio web mediante Render, conectado al repositorio de GitHub.

**Demo online:** pendiente de publicación.

**Repositorio:** https://github.com/lagabyok/Gabriela-Coronel-seminario-de-actualizacion-2026

## Limitaciones y mejoras futuras

El rendimiento del modelo puede variar según la calidad, iluminación, fondo, encuadre y características de las imágenes ingresadas.

La evaluación actual se realizó principalmente con imágenes del dataset RealWaste. Se identificaron dificultades de generalización frente a algunas fotografías externas.

Como líneas de mejora se propone:

- Ampliar la diversidad de imágenes de evaluación.
- Mejorar la capacidad de generalización.
- Comparar el modelo SVM actual con una arquitectura de Deep Learning mediante Transfer Learning, como MobileNetV2.
- Analizar las métricas por categoría y las matrices de confusión.
- Evaluar alternativas para mejorar el reconocimiento de materiales visualmente complejos.

## Autora

Desarrollado por Gabriela Coronel.

Proyecto académico — Seminario de Actualización 2026.

**ReciClass — Clasificación de residuos mediante Machine Learning.**