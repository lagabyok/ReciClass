// En desarrollo, la API corre localmente.
// Cuando se publique, reemplazar por la URL HTTPS del backend.
const API_URL = "";

const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("preview");
const predictButton = document.getElementById("predictButton");
const statusText = document.getElementById("status");
const result = document.getElementById("result");

let selectedFile = null;

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  result.hidden = true;
  statusText.textContent = "";
  statusText.className = "status";

  if (!file) {
    selectedFile = null;
    predictButton.disabled = true;
    return;
  }

  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Formato no vÃ¡lido. SeleccionÃ¡ JPG, PNG o WEBP.");
    imageInput.value = "";
    predictButton.disabled = true;
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("La imagen supera el lÃ­mite de 10 MB.");
    imageInput.value = "";
    predictButton.disabled = true;
    return;
  }

  selectedFile = file;
  fileName.textContent = file.name;
  preview.src = URL.createObjectURL(file);
  previewWrap.hidden = false;
  predictButton.disabled = false;
});

predictButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictButton.disabled = true;
  predictButton.textContent = "Analizando...";
  result.hidden = true;
  statusText.className = "status";
  statusText.textContent = "Procesando imagen. Esto puede tardar unos segundos.";

  try {
    const base64 = await fileToBase64(selectedFile);
    const response = await fetch(`${API_URL}/predecir`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: base64,
        filename: selectedFile.name
      })
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof data.detail === "string"
        ? data.detail
        : `La API respondiÃ³ con el estado ${response.status}.`;
      throw new Error(detail);
    }

    const traducciones = {
      "Cardboard": "CartÃ³n",
      "Food Organics": "Residuos orgÃ¡nicos",
      "Glass": "Vidrio",
      "Metal": "Metal",
      "Miscellaneous Trash": "Residuos varios",
      "Paper": "Papel",
      "Plastic": "PlÃ¡stico",
      "Textile Trash": "Residuos textiles",
      "Vegetation": "VegetaciÃ³n"
    };

    document.getElementById("prediction").textContent =
      traducciones[data.prediccion] || data.prediccion;
    document.getElementById("confidence").textContent =
      `Confianza estimada: ${(data.probabilidad * 100).toFixed(2)} %`;

    const probabilities = document.getElementById("probabilities");
    probabilities.replaceChildren();

    Object.entries(data.probabilidades)
      .sort((a, b) => b[1] - a[1])
      .forEach(([category, probability]) => {
        const row = document.createElement("div");
        row.className = "prob-row";

        const label = document.createElement("div");
        label.className = "prob-label";

        const name = document.createElement("span");
        name.textContent = traducciones[category] || category;

        const value = document.createElement("strong");
        value.textContent = `${(probability * 100).toFixed(1)} %`;

        label.append(name, value);

        const track = document.createElement("div");
        track.className = "track";
        const fill = document.createElement("div");
        fill.className = "fill";
        fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
        track.append(fill);

        row.append(label, track);
        probabilities.append(row);
      });

    result.hidden = false;
    statusText.textContent = "AnÃ¡lisis completado.";
  } catch (error) {
    showError(
      error.message.includes("Failed to fetch")
        ? "No se pudo conectar con la API. VerificÃ¡ que FastAPI estÃ© ejecutÃ¡ndose y que API_URL sea correcto."
        : error.message
    );
  } finally {
    predictButton.disabled = !selectedFile;
    predictButton.textContent = "Clasificar residuo";
  }
});

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result);
      resolve(dataUrl.split(",", 2)[1]);
    };
    reader.onerror = () => reject(new Error("No se pudo leer el archivo seleccionado."));
    reader.readAsDataURL(file);
  });
}

function showError(message) {
  statusText.textContent = message;
  statusText.className = "status error";
}

