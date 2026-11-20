
from pathlib import Path
import joblib
import numpy as np

from feature_extractor import extraer_caracteristicas_desde_bytes

BASE_DIR = Path(__file__).resolve().parents[1]
MODELO = BASE_DIR / "modelo" / "modelo.joblib"

ruta = input("Pegá la ruta completa de una foto de comida: ").strip().strip('"')

with open(ruta, "rb") as archivo:
    imagen_bytes = archivo.read()

modelo = joblib.load(MODELO)
caracteristicas = extraer_caracteristicas_desde_bytes(imagen_bytes)

print("\nCaracterísticas generadas:", caracteristicas.shape)
print("Valores finitos:", np.isfinite(caracteristicas).all())
print("Mínimo:", caracteristicas.min())
print("Máximo:", caracteristicas.max())

prediccion = modelo.predict(caracteristicas)[0]
probabilidades = modelo.predict_proba(caracteristicas)[0]

print("\nPREDICCIÓN:", prediccion)
print("\nPROBABILIDADES:")

for clase, prob in sorted(
    zip(modelo.classes_, probabilidades),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{clase}: {prob * 100:.2f}%")
