from itertools import combinations
from pathlib import Path

import torch
import torchaudio
import torch.nn.functional as F
from speechbrain.inference.speaker import SpeakerRecognition


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

# Se utiliza GPU si esta disponible.
# En nuestra maquina virtual se utilizara CPU.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Carpeta donde se encuentran las copias procesadas de los audios.
# Estos archivos ya fueron convertidos a 16 kHz, 16 bits y mono.
AUDIO_DIR = Path("data/pilot_processed")

# Participantes utilizados en la prueba piloto.
# Se utilizan identificadores anonimos en lugar de nombres reales.
PARTICIPANTES = {
    "P01": "P01.wav",
    "P02": "P02.wav",
    "P03": "P03.wav",
}

# Relacion conocida entre cada par de participantes.
# Esta informacion se utiliza solamente para mostrar los resultados
# de una forma mas facil de interpretar.
RELACIONES = {
    ("P01", "P02"): "madre-hija",
    ("P01", "P03"): "no relacionadas",
    ("P02", "P03"): "no relacionadas",
}


# ============================================================
# CARGA DEL MODELO
# ============================================================

print(f"Ejecutando en: {DEVICE.upper()}")
print("Cargando modelo ECAPA-TDNN...")

# Se carga el modelo ECAPA-TDNN previamente entrenado con VoxCeleb.
#
# Este modelo recibe una grabacion de voz y genera una representacion
# numerica del hablante llamada speaker embedding.
#
# Cada embedding generado por este modelo tiene 192 dimensiones.
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/ecapa",
    run_opts={"device": DEVICE},
)

print("Modelo cargado correctamente.\n")


# ============================================================
# FUNCION PARA OBTENER EL EMBEDDING DE UNA VOZ
# ============================================================

def obtener_embedding(ruta_audio):
    """
    Carga un archivo de audio y obtiene su speaker embedding
    utilizando el modelo ECAPA-TDNN.
    """

    # Carga directamente el archivo WAV desde la ruta local.
    # Se evita model.load_audio() para impedir que SpeechBrain
    # cree enlaces simbolicos en la raiz del repositorio.
    signal, sample_rate = torchaudio.load(str(ruta_audio))

    # El modelo ECAPA utilizado trabaja con audio a 16 kHz.
    # Los archivos del piloto ya fueron preparados con este formato.
    if sample_rate != 16000:
        raise ValueError(
            f"Se esperaba audio a 16000 Hz, pero se obtuvo {sample_rate} Hz"
        )

    # torchaudio devuelve una matriz con forma:
    # [numero_de_canales, numero_de_muestras]
    #
    # Los audios del piloto son mono, por lo que normalmente
    # la forma sera [1, numero_de_muestras].
    #
    # Si se recibe accidentalmente un archivo con varios canales,
    # se convierte a mono antes de enviarlo al modelo.
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # La senal se mueve al dispositivo donde esta ejecutandose
    # el modelo, que en esta maquina virtual corresponde a CPU.
    signal = signal.to(DEVICE)

    # No se necesitan gradientes porque ECAPA ya esta entrenado.
    # Solo se utiliza para extraer la representacion de la voz.
    with torch.no_grad():
        embedding = model.encode_batch(signal)

    # ECAPA devuelve inicialmente una estructura [1, 1, 192].
    # squeeze() elimina las dimensiones adicionales y conserva
    # solamente el vector de 192 valores.
    embedding = embedding.squeeze()

    return embedding.cpu()


# ============================================================
# FUNCION PARA CALCULAR SIMILITUD ENTRE DOS VOCES
# ============================================================

def similitud_coseno(embedding_a, embedding_b):
    """
    Calcula la similitud coseno entre dos speaker embeddings.
    """

    # La similitud coseno compara la orientacion de los dos vectores.
    #
    # Un valor mayor indica que las representaciones generadas por
    # ECAPA-TDNN son mas similares entre si.
    #
    # No se utiliza aqui un umbral para decidir si dos personas son
    # la misma identidad. Para este proyecto interesa conservar el
    # valor continuo de similitud.
    score = F.cosine_similarity(
        embedding_a,
        embedding_b,
        dim=0,
    )

    # PyTorch devuelve un tensor.
    # item() permite obtener el valor numerico como un float de Python.
    return score.item()


# ============================================================
# EXTRACCION DE EMBEDDINGS
# ============================================================

# Este diccionario almacenara temporalmente el embedding
# correspondiente a cada participante.
embeddings = {}

print("Extrayendo embeddings:")

# Se procesa cada participante de forma independiente.
for participante, archivo in PARTICIPANTES.items():

    # Se construye la ruta completa del archivo WAV.
    ruta = AUDIO_DIR / archivo

    # Antes de procesar el audio se verifica que el archivo exista.
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    # Se obtiene el embedding de 192 dimensiones para este participante.
    embedding = obtener_embedding(ruta)

    # Se guarda el embedding utilizando el identificador del participante.
    embeddings[participante] = embedding

    print(
        f"  {participante}: embedding generado "
        f"({embedding.shape[0]} dimensiones)"
    )


# ============================================================
# COMPARACION ENTRE PARTICIPANTES
# ============================================================

print("\nResultados de similitud coseno:")
print("-" * 65)

# Aqui se almacenaran los resultados de cada comparacion.
resultados = {}

# combinations() genera todas las parejas posibles sin repetirlas.
#
# Para tres participantes se obtienen:
#
# P01 - P02
# P01 - P03
# P02 - P03
for participante_a, participante_b in combinations(PARTICIPANTES.keys(), 2):

    # Se calcula la similitud entre los embeddings de ambos participantes.
    score = similitud_coseno(
        embeddings[participante_a],
        embeddings[participante_b],
    )

    # Se guarda el resultado para poder utilizarlo despues
    # al construir la matriz de similitud.
    resultados[(participante_a, participante_b)] = score

    # Se obtiene la relacion conocida entre las dos personas.
    relacion = RELACIONES[(participante_a, participante_b)]

    # Se muestra el resultado con cuatro decimales.
    print(
        f"{participante_a} vs {participante_b} "
        f"| {relacion:16} "
        f"| similitud = {score:.4f}"
    )


# ============================================================
# MATRIZ DE SIMILITUD
# ============================================================

# La matriz permite visualizar facilmente todas las comparaciones.
#
# La diagonal siempre vale 1.0 porque corresponde a comparar
# cada participante consigo mismo.
print("\nMatriz de similitud:")

ids = list(PARTICIPANTES.keys())

# Encabezado de la matriz.
print(f"{'':8}", end="")

for participante in ids:
    print(f"{participante:>10}", end="")

print()


# Se construye una fila de la matriz por participante.
for participante_a in ids:

    print(f"{participante_a:8}", end="")

    for participante_b in ids:

        # Comparar una voz consigo misma produce similitud 1.
        if participante_a == participante_b:
            score = 1.0

        else:
            # Los resultados solamente se calcularon una vez por pareja.
            #
            # sorted() permite que P02-P01 encuentre el mismo resultado
            # almacenado anteriormente como P01-P02.
            par = tuple(sorted((participante_a, participante_b)))
            score = resultados[par]

        print(f"{score:10.4f}", end="")

    print()
