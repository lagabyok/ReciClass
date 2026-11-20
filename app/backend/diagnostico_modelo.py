import joblib
from pathlib import Path

# Ruta del modelo desde la ubicación del script
RUTA_MODELO = (
    Path(__file__).resolve().parent.parent
    / "Modelo"
    / "modelo.joblib"
)

print("Ruta del modelo:", RUTA_MODELO)
print("¿Existe?", RUTA_MODELO.exists())

if not RUTA_MODELO.exists():
    raise FileNotFoundError(f"No se encontró: {RUTA_MODELO}")

modelo_final = joblib.load(RUTA_MODELO)

print("\n========== MODELO CARGADO ==========")
print("Tipo:", type(modelo_final))

print("\n========== CLASES DEL MODELO ==========")
print(modelo_final.classes_)

print("\n========== PARÁMETROS ==========")
print(modelo_final.get_params())

# =====================================================
# 2. REVISAR LAS ETIQUETAS
# =====================================================

print("\n========== DISTRIBUCIÓN DE CLASES ==========")

# Reemplazá estas rutas por las que correspondan
# a tus archivos, si guardaste los conjuntos por separado.

for archivo in [
    "modelo/y_train.csv",
    "modelo/y_val.csv",
    "modelo/y_test.csv"
]:
    ruta = Path(archivo)

    if ruta.exists():
        etiquetas = pd.read_csv(ruta).iloc[:, 0]
        print(f"\n{archivo}")
        print(etiquetas.value_counts())


# =====================================================
# 3. REVISAR PREDICCIONES
# =====================================================

# Este bloque requiere que tengas X_test_feat disponible.
# Si no está guardado, no se puede ejecutar todavía.

if Path("modelo/X_test_feat.npy").exists():

    X_test_feat = np.load("modelo/X_test_feat.npy")

    print("\n========== DIMENSIONES ==========")
    print("X_test:", X_test_feat.shape)

    pred_test = modelo_final.predict(X_test_feat)

    print("\n========== PREDICCIONES ==========")
    print(Counter(pred_test))

    if hasattr(modelo_final, "predict_proba"):
        probabilidades = modelo_final.predict_proba(X_test_feat[:5])

        print("\n========== PROBABILIDADES ==========")

        for i, fila in enumerate(probabilidades):
            top_indices = np.argsort(fila)[::-1][:3]

            print(f"\nImagen {i + 1}:")
            for indice in top_indices:
                print(
                    modelo_final.classes_[indice],
                    round(float(fila[indice]), 4)
                )

else:
    print(
        "\nNo se encontró X_test_feat.npy. "
        "Necesitamos exportar las características de prueba "
        "desde el notebook para continuar el diagnóstico."
    )

print("\n========== CARACTERÍSTICAS ESPERADAS ==========")

print(
    "Cantidad de características:",
    modelo_final.named_steps["escalado"].n_features_in_
)

print(
    "Cantidad de clases:",
    len(modelo_final.classes_)
)