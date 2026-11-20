
from pathlib import Path
import os
import random
import json

import numpy as np
import pandas as pd
import joblib
import sklearn

from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ==================================================
# 1. CONFIGURACIÓN
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

CATEGORIAS = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]

# CAMBIAR ESTA RUTA por la ubicación REAL de tu dataset.
DATASET_DIR = Path(
    r"C:\Users\facus\Downloads\RealWaste"
)

# Carpeta donde se guardará el nuevo modelo.
PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "app" / "modelo"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RUTA_MODELO = MODEL_DIR / "modelo.joblib"

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp"}

# ==================================================
# 2. EXTRAER CARACTERÍSTICAS
# ==================================================

TAMANO = (128, 128)

def extraer_caracteristicas(ruta_imagen):

    with Image.open(ruta_imagen) as im:
        im = im.convert("RGB").resize(
            (TAMANO[1], TAMANO[0]),
            Image.Resampling.BILINEAR
        )

        rgb = np.asarray(im, dtype=np.float32) / 255.0

    gris = rgb2gray(rgb)

    vector_hog = hog(
        gris,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    gris_uint8 = np.clip(
        gris * 255, 0, 255
    ).astype(np.uint8)

    lbp = local_binary_pattern(
        gris_uint8,
        P=8,
        R=1,
        method="uniform"
    )

    hist_lbp, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        density=True
    )

    hsv = rgb2hsv(rgb)

    hist_hsv, _ = np.histogramdd(
        hsv.reshape(-1, 3),
        bins=(8, 8, 4),
        range=((0, 1), (0, 1), (0, 1)),
        density=True
    )

    vector = np.concatenate([
        vector_hog,
        hist_lbp.astype(np.float32),
        hist_hsv.ravel().astype(np.float32)
    ]).astype(np.float32)

    return vector


# ==================================================
# 3. LOCALIZAR IMÁGENES
# ==================================================

print("\nBuscando imágenes del dataset...")

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"No se encontró el dataset: {DATASET_DIR}\n"
        "Corregí DATASET_DIR en el código."
    )

registros = []

for categoria in CATEGORIAS:

    archivos = [
        p for p in DATASET_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONES
        and categoria in p.parts
    ]

    for archivo in archivos:
        registros.append({
            "ruta": archivo,
            "categoria": categoria
        })

df = pd.DataFrame(registros).drop_duplicates(
    subset="ruta"
).reset_index(drop=True)

if df.empty:
    raise RuntimeError(
        "No se encontraron imágenes. Revisá la ruta y las carpetas."
    )

print("Total de imágenes:", len(df))
print("\nImágenes por categoría:")
print(df["categoria"].value_counts())

faltantes = set(CATEGORIAS) - set(df["categoria"].unique())

if faltantes:
    raise RuntimeError(f"Faltan categorías: {faltantes}")


# ==================================================
# 4. SEPARAR LOS DATOS
# ==================================================

X = df["ruta"].to_numpy()
y = df["categoria"].to_numpy()

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=SEED,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=(0.15 / 0.85),
    random_state=SEED,
    stratify=y_temp
)

print("\nParticiones:")
print("Entrenamiento:", len(X_train))
print("Validación:", len(X_val))
print("Prueba:", len(X_test))


# ==================================================
# 5. PROCESAR IMÁGENES
# ==================================================

def construir_matriz(rutas, nombre):

    caracteristicas = []

    for i, ruta in enumerate(rutas, start=1):

        vector = extraer_caracteristicas(ruta)
        caracteristicas.append(vector)

        if i % 100 == 0 or i == len(rutas):
            print(f"{nombre}: {i}/{len(rutas)}")

    return np.vstack(caracteristicas)


print("\nExtrayendo características...")

X_train_feat = construir_matriz(
    X_train, "Entrenamiento"
)

X_val_feat = construir_matriz(
    X_val, "Validación"
)

X_test_feat = construir_matriz(
    X_test, "Prueba"
)

print("\nDimensiones:")
print(X_train_feat.shape)
print(X_val_feat.shape)
print(X_test_feat.shape)


# ==================================================
# 6. ENTRENAR Y COMPARAR MODELOS
# ==================================================

modelos = {

    "SVM": Pipeline([
        ("escalado", StandardScaler()),
        ("clasificador", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=SEED
        ))
    ]),

    "Random Forest": Pipeline([
        ("escalado", StandardScaler()),
        ("clasificador", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ))
    ]),

    "Extra Trees": Pipeline([
        ("escalado", StandardScaler()),
        ("clasificador", ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ))
    ])
}

resultados = {}
modelos_entrenados = {}

print("\nComenzando entrenamiento...")

for nombre, modelo in modelos.items():

    print(f"\nEntrenando {nombre}...")

    modelo.fit(X_train_feat, y_train)

    predicciones = modelo.predict(X_val_feat)

    f1 = f1_score(
        y_val,
        predicciones,
        average="macro",
        zero_division=0
    )

    resultados[nombre] = {
        "accuracy": accuracy_score(y_val, predicciones),
        "precision_macro": precision_score(
            y_val, predicciones,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_val, predicciones,
            average="macro",
            zero_division=0
        ),
        "f1_macro": f1
    }

    modelos_entrenados[nombre] = modelo

    print(f"F1 macro de validación: {f1:.4f}")


# ==================================================
# 7. SELECCIONAR EL MEJOR MODELO
# ==================================================

tabla = pd.DataFrame(resultados).T

tabla = tabla.sort_values(
    by="f1_macro",
    ascending=False
)

print("\nRESULTADOS DE VALIDACIÓN")
print(tabla.round(4))

mejor_nombre = tabla.index[0]

print("\nMejor modelo:", mejor_nombre)


# ==================================================
# 8. REENTRENAR CON ENTRENAMIENTO + VALIDACIÓN
# ==================================================

print("\nReentrenando el mejor modelo...")

X_dev_feat = np.vstack([
    X_train_feat,
    X_val_feat
])

y_dev = np.concatenate([
    y_train,
    y_val
])

modelo_final = modelos[mejor_nombre]

modelo_final.fit(X_dev_feat, y_dev)


# ==================================================
# 9. EVALUACIÓN FINAL
# ==================================================

pred_test = modelo_final.predict(X_test_feat)

print("\nRESULTADOS FINALES EN PRUEBA")

print(
    "Accuracy:",
    round(accuracy_score(y_test, pred_test), 4)
)

print(
    "F1 macro:",
    round(f1_score(
        y_test,
        pred_test,
        average="macro",
        zero_division=0
    ), 4)
)

print("\nREPORTE POR CATEGORÍA")

print(classification_report(
    y_test,
    pred_test,
    labels=CATEGORIAS,
    zero_division=0
))


# ==================================================
# 10. GUARDAR MODELO
# ==================================================

print("\nGuardando modelo...")

joblib.dump(
    modelo_final,
    RUTA_MODELO,
    compress=3
)

print("\nMODELO GUARDADO CORRECTAMENTE")
print("Ubicación:", RUTA_MODELO)

peso = os.path.getsize(RUTA_MODELO) / (1024 * 1024)

print(f"Tamaño: {peso:.2f} MB")
print("Scikit-learn:", sklearn.__version__)

# Guardar información del entrenamiento.
metadatos = {
    "proyecto": "ReciClass",
    "modelo_seleccionado": mejor_nombre,
    "resultados_validacion": tabla.to_dict(),
    "accuracy_prueba": float(
        accuracy_score(y_test, pred_test)
    ),
    "f1_macro_prueba": float(
        f1_score(
            y_test,
            pred_test,
            average="macro",
            zero_division=0
        )
    ),
    "categorias": CATEGORIAS,
    "scikit_learn": sklearn.__version__,
    "tamano_mb": round(peso, 2)
}

with open(
    MODEL_DIR / "metadatos_modelo.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadatos,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n¡REENTRENAMIENTO TERMINADO!")
