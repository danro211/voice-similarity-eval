import os
import glob
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

print("1. Cargando el modelo ECAPA-TDNN de SpeechBrain...")
# Descarga/carga el modelo pre-entrenado para reconocimiento de locutor
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

# 2. Buscar todos los audios listos para evaluación (ignoramos los de 24kHz de xtts)
rutas_audios = glob.glob("data/raw/*.wav")
rutas_validas = rutas_audios 

if not rutas_validas:
    print("No se encontraron audios válidos en data/raw/")
    exit()

nombres_etiquetas = []
embeddings = []

print(f"\n2. Extrayendo embeddings biométricos de {len(rutas_validas)} audios...")
for ruta in sorted(rutas_validas):
    nombre_archivo = os.path.basename(ruta).replace(".wav", "")
    
    # Limpiar el nombre para la gráfica (Ej: daniel_sintetico_final -> Daniel Sintético)
    etiqueta = nombre_archivo.replace("_", " ").title().replace("Final", "").replace("Ref", "Real").replace("Test", "Real")
    nombres_etiquetas.append(etiqueta.strip())
    
    # Cargar audio y extraer el embedding vectorial
    signal, fs = torchaudio.load(ruta)
    
    # Asegurar que esté a 16kHz y en mono
    if fs != 16000:
        print(f"Advertencia: {nombre_archivo} no está a 16kHz. Remuestreando...")
        transform = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = transform(signal)
    
    # Extraer características (1x1x192 vector)
    with torch.no_grad():
        embedding = classifier.encode_batch(signal)
        # Aplanar el tensor a un array de 1D de numpy
        embeddings.append(embedding.squeeze().cpu().numpy())

# Convertir la lista a una matriz numpy
embeddings_matrix = np.array(embeddings)

print("\n3. Calculando Matriz de Similitud Coseno...")
# Calcula la similitud de todos contra todos (valores entre -1 y 1)
similitud_matriz = cosine_similarity(embeddings_matrix)

# 4. Generar la visualización (Heatmap)
print("\n4. Generando el mapa de calor (Heatmap)...")
plt.figure(figsize=(10, 8))
sns.heatmap(similitud_matriz, annot=True, cmap="YlGnBu", xticklabels=nombres_etiquetas, yticklabels=nombres_etiquetas, vmin=0, vmax=1)

plt.title("Matriz de Similitud de Voz (ECAPA-TDNN)\nProtocolo de Evaluación LAPA-CR")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Guardar la imagen en la carpeta del proyecto
archivo_imagen = "matriz_similitud.png"
plt.savefig(archivo_imagen, dpi=300)
print(f"\n¡Análisis completado! La matriz se ha guardado como '{archivo_imagen}'")

# Mostrar la imagen en pantalla (si el entorno gráfico lo permite)
plt.show()
