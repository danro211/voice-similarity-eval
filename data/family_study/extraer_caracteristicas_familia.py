#!/usr/bin/env python3
"""Mide el tono y las caracteristicas temporales de cada grabacion

Lee los WAV procesados y comprueba sus hashes con el manifiesto existente
Usa Praat-Parselmouth para F0 y WebRTC VAD para estimar actividad de habla
Las tasas de lectura usan el numero de palabras de cada texto asignado
Guarda mediciones, alertas de calidad y parametros sin modificar los audios
"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import wave
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import parselmouth
import webrtcvad

BASE = Path(__file__).resolve().parent
PROCESADOS = BASE / "processed"
RESULTADOS = BASE / "results"
MANIFIESTO = RESULTADOS / "embeddings_manifest.csv"
SALIDA = RESULTADOS / "caracteristicas_audio.csv"
CONFIG = RESULTADOS / "caracteristicas_audio_config.json"
PARCIAL = RESULTADOS / "caracteristicas_audio_parcial.json"
PATRON = re.compile(r"^(F\d{2}_P\d{2})_(E(?:0[1-9]|10))_(T0[12])\.wav$")

# Estas son las frases que leyeron los participantes durante las grabaciones
# El numero de palabras sale del texto escrito, no de un reconocimiento automatico
ENUNCIADOS = {
    "E01": "En gran medida es correcto, de hecho.",
    "E02": "Lo sé porque he dedicado mi vida a él.",
    "E03": "En este tema nos falta mucho por hacer.",
    "E04": "Tal cosa además de imposible sería absurda.",
    "E05": "Entonces, es evidente dónde está el problema.",
    "E06": "Veamos, por ejemplo, el principio de igualdad.",
    "E07": "La población no va a disminuir, todo lo contrario.",
    "E08": "Solo basta imaginarlas cuando seamos diez millones.",
    "E09": "El planeta y el ambiente no pueden aceptar menos que eso.",
    "E10": "En otra zona del país, mi abuelo es un buen ejemplo de cómo vivir más años.",
}
PALABRAS = {k: len(v.split()) for k, v in ENUNCIADOS.items()}

SR = 16000
F0_MIN_HZ = 60.0
F0_MAX_HZ = 700.0
F0_PASO_S = 0.01
F0_UMBRAL_SONORIDAD = 0.45
F0_UMBRAL_SILENCIO = 0.03
VAD_TRAMA_MS = 20
VAD_MODO = 1  # El modo 1 prioriza detectar habla sin ser demasiado estricto
GUARDAR_CADA = 10

COLUMNAS = [
    "archivo", "participante_id", "familia_id", "enunciado", "toma", "sha256_audio",
    "duracion_total_s", "n_palabras_texto", "f0_mediana_hz", "f0_q1_hz",
    "f0_q3_hz", "f0_ric_hz", "f0_desv_semitonos", "n_tramas_f0",
    "n_tramas_sonoras", "fraccion_tramas_sonoras",
    "duracion_habla_estimada_s", "fraccion_habla_estimada",
    "tasa_lectura_total_pal_s", "tasa_articulacion_aprox_pal_s",
    "fraccion_muestras_recortadas", "pico_absoluto", "alertas_calidad",
]


# Si los datos no cumplen las condiciones previstas, detenemos la extraccion
def exigir(condicion, mensaje):
    if not condicion:
        raise ValueError(mensaje)


# Usamos SHA-256 para comprobar que medimos los mismos audios del manifiesto
def sha256_archivo(ruta):
    with ruta.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# Leemos el manifiesto que relaciona las grabaciones con sus huellas
def leer_manifest():
    exigir(MANIFIESTO.is_file(), f"No existe el manifiesto del paso 3: {MANIFIESTO}")
    with MANIFIESTO.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        exigir(reader.fieldnames is not None and {"archivo", "participante_id", "familia_id", "enunciado", "toma", "sha256_audio"} <= set(reader.fieldnames), "Manifest con columnas faltantes")
        filas = list(reader)
    exigir(len(filas) == 298, "El manifiesto debe contener 298 grabaciones")
    exigir(len(set(x["archivo"] for x in filas)) == 298, "Manifest con archivos repetidos")
    # Mantenemos el orden original para que cada fila corresponda a su WAV
    conteos = Counter()
    for fila in filas:
        nombre = fila["archivo"]
        m = PATRON.fullmatch(nombre)
        exigir(m is not None, f"Nombre no reconocido: {nombre}")
        participante, e, t = m.groups()
        exigir((fila["participante_id"], fila["familia_id"], fila["enunciado"], fila["toma"]) ==
                (participante, participante.split("_")[0], e, t), f"Metadatos incorrectos: {nombre}")
        exigir(re.fullmatch(r"[0-9a-f]{64}", fila["sha256_audio"]) is not None, f"Huella SHA-256 invalida: {nombre}")
        ruta = PROCESADOS / nombre
        exigir(ruta.is_file(), f"Falta el WAV: {nombre}")
        exigir(sha256_archivo(ruta) == fila["sha256_audio"], f"WAV modificado desde paso 3: {nombre}")
        conteos[participante] += 1
    exigir(len(conteos) == 15, "Participantes diferentes de 15")
    for p, n in conteos.items():
        exigir(n == (18 if p == "F04_P11" else 20), f"Grabaciones incompletas para {p}")
    exigir(len(list(PROCESADOS.glob("*.wav"))) == 298, "Cantidad de WAV diferente de la validada en paso 3")
    return filas


# Abrimos cada WAV y comprobamos que tenga 16 kHz, un canal y muestras PCM de 16 bits
def abrir_wav(ruta):
    with wave.open(str(ruta), "rb") as w:
        props = (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype())
        exigir(props == (16000, 1, 2, "NONE"), f"Formato WAV incorrecto: {ruta.name}: {props}")
        n = w.getnframes()
        raw = w.readframes(n)
    exigir(n > 0 and len(raw) == 2 * n, f"WAV vacio o truncado: {ruta.name}")
    pcm = np.frombuffer(raw, dtype="<i2")
    muestras = pcm.astype(np.float64) / 32768.0
    exigir(np.isfinite(muestras).all(), f"Audio no finito: {ruta.name}")
    return pcm, muestras


# Praat estima la frecuencia fundamental a lo largo de la grabacion
# Marcamos las grabaciones que tienen poca voz sonora o posibles errores de F0
def extraer_f0(muestras):
    """Estima F0 mediante autocorrelacion en tramas de 10 ms"""
    sonido = parselmouth.Sound(muestras, sampling_frequency=SR)
    pitch = sonido.to_pitch_ac(
        time_step=F0_PASO_S,
        pitch_floor=F0_MIN_HZ,
        pitch_ceiling=F0_MAX_HZ,
        voicing_threshold=F0_UMBRAL_SONORIDAD,
        silence_threshold=F0_UMBRAL_SILENCIO,
    )
    pista = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    validos = pista[np.isfinite(pista) & (pista > 0)]
    n = int(pista.size)
    exigir(n > 0, "Praat no produjo tramas de F0")
    resultado = {"n_tramas_f0": n, "n_tramas_sonoras": int(validos.size),
                 "fraccion_tramas_sonoras": validos.size / n}
    alertas = []
    if validos.size < 15 or validos.size / n < 0.10:
        alertas.append("f0_cobertura_baja")
    if validos.size:
        q1, q3 = np.quantile(validos, [0.25, 0.75])
        mediana = float(np.median(validos))
        st = 12.0 * np.log2(validos / mediana)
        resultado.update({
            "f0_mediana_hz": mediana, "f0_q1_hz": float(q1), "f0_q3_hz": float(q3),
            "f0_ric_hz": float(q3 - q1),
            "f0_desv_semitonos": float(np.std(st, ddof=1)) if validos.size > 1 else None,
        })
        if np.mean(validos <= F0_MIN_HZ * 1.06) >= 0.05:
            alertas.append("f0_cerca_piso")
        if np.mean(validos >= F0_MAX_HZ * 0.94) >= 0.05:
            alertas.append("f0_cerca_techo")
        # Revisamos saltos de F0 solo entre tramas sonoras consecutivas
        antes, despues = pista[:-1], pista[1:]
        juntos = (antes > 0) & (despues > 0)
        if np.count_nonzero(juntos) >= 20:
            saltos_st = np.abs(12.0 * np.log2(despues[juntos] / antes[juntos]))
            if np.mean(saltos_st > 8) > 0.025:
                alertas.append("f0_posibles_saltos_octava")
    else:
        resultado.update({"f0_mediana_hz": None, "f0_q1_hz": None,
                          "f0_q3_hz": None, "f0_ric_hz": None,
                          "f0_desv_semitonos": None})
        alertas.append("f0_no_detectada")
    return resultado, alertas


# Sumamos el tiempo de las tramas que WebRTC clasifica como habla
# No conocemos los limites exactos de cada palabra ni contamos silabas
def duracion_vad(pcm):
    """Estima tiempo con habla sin identificar palabras ni silabas"""
    n_por_trama = SR * VAD_TRAMA_MS // 1000
    # Si la ultima trama es corta, agregamos silencio solo para consultar el detector
    n_tramas = math.ceil(len(pcm) / n_por_trama)
    exigir(n_tramas > 0, "Audio sin tramas VAD")
    relleno = n_tramas * n_por_trama - len(pcm)
    pcm_ext = np.pad(pcm, (0, relleno), mode="constant") if relleno else pcm
    vad = webrtcvad.Vad(VAD_MODO)
    muestras_habla = 0
    for i in range(n_tramas):
        trama = pcm_ext[i * n_por_trama:(i + 1) * n_por_trama]
        if vad.is_speech(trama.tobytes(), SR):
            muestras_habla += min(n_por_trama, len(pcm) - i * n_por_trama)
    tiempo_s = muestras_habla / SR
    duracion = len(pcm) / SR
    fraccion = muestras_habla / len(pcm)
    alertas = []
    if fraccion < 0.20:
        alertas.append("vad_cobertura_baja")
    if fraccion > 0.99:
        alertas.append("vad_casi_todo_audio")
    return tiempo_s, fraccion, alertas


# Reunimos todas las mediciones y alertas correspondientes a una grabacion
def medir(fila):
    pcm, audio = abrir_wav(PROCESADOS / fila["archivo"])
    duracion = len(pcm) / SR
    medidas_f0, alertas = extraer_f0(audio)
    tiempo_habla, fraccion_habla, alertas_vad = duracion_vad(pcm)
    alertas.extend(alertas_vad)
    pico = float(np.max(np.abs(audio)))
    clip = float(np.mean(np.abs(pcm.astype(np.int32)) >= 32766))
    if clip > 0.001:
        alertas.append("posible_saturacion")
    if pico < 0.005:
        alertas.append("amplitud_muy_baja")
    palabra_n = PALABRAS[fila["enunciado"]]
    # Si no se detecta habla, dejamos la tasa de articulacion vacia y registramos una alerta
    if tiempo_habla <= 0:
        alertas.append("tasa_articulacion_no_calculable")
    fila_resultado = {
        "archivo": fila["archivo"], "participante_id": fila["participante_id"],
        "familia_id": fila["familia_id"], "enunciado": fila["enunciado"],
        "toma": fila["toma"], "sha256_audio": fila["sha256_audio"],
        "duracion_total_s": duracion,
        "n_palabras_texto": palabra_n,
        **medidas_f0,
        "duracion_habla_estimada_s": tiempo_habla,
        "fraccion_habla_estimada": fraccion_habla,
        "tasa_lectura_total_pal_s": palabra_n / duracion,
        "tasa_articulacion_aprox_pal_s": palabra_n / tiempo_habla if tiempo_habla > 0 else None,
        "fraccion_muestras_recortadas": clip, "pico_absoluto": pico,
        "alertas_calidad": ";".join(sorted(set(alertas))),
    }
    for campo in COLUMNAS:
        exigir(campo in fila_resultado, f"Campo faltante: {campo}")
        v = fila_resultado[campo]
        if isinstance(v, (float, np.floating)):
            exigir(math.isfinite(float(v)), f"Numero no finito en {fila['archivo']}: {campo}")
    exigir(0 <= tiempo_habla <= duracion and 0 <= fraccion_habla <= 1.0 + 1e-8,
           f"Duracion VAD incoherente: {fila['archivo']}")
    return fila_resultado


# Convertimos los numeros al formato de salida y dejamos vacias las medidas no calculables
def convertir_a_celda(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return format(float(v), ".12g")
    return str(v)


# Preparamos todas las filas en memoria antes de guardar el CSV
def csv_bytes(filas):
    bufer = io.StringIO(newline="")
    w = csv.DictWriter(bufer, fieldnames=COLUMNAS, lineterminator="\n")
    w.writeheader()
    for fila in filas:
        w.writerow({c: convertir_a_celda(fila[c]) for c in COLUMNAS})
    return bufer.getvalue().encode("utf-8")


# Escribimos en un temporal y lo renombramos cuando este completo
def escritura_atomica(ruta, contenido):
    temporal = ruta.with_name(ruta.name + ".tmp")
    exigir(not temporal.exists(), f"Existe temporal de ejecucion previa: {temporal}")
    try:
        with temporal.open("xb") as f:
            f.write(contenido)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporal, ruta)
    finally:
        if temporal.exists():
            temporal.unlink()


# Guardamos las versiones, los parametros y los hashes usados en la extraccion
def metadatos_checkpoint():
    parametros = {
        "f0_metodo": "Praat/Parselmouth to_pitch_ac (autocorrelacion cruda)",
        "f0_piso_hz": F0_MIN_HZ,
        "f0_techo_hz": F0_MAX_HZ,
        "f0_paso_s": F0_PASO_S,
        "f0_voicing_threshold": F0_UMBRAL_SONORIDAD,
        "f0_silence_threshold": F0_UMBRAL_SILENCIO,
        "vad_metodo": "WebRTC VAD", "vad_modo": VAD_MODO,
        "vad_trama_ms": VAD_TRAMA_MS,
        "palabras_esperadas_por_enunciado": PALABRAS,
        "criterio_medicion": "F0 solo en tramas sonoras; tiempo habla VAD; las palabras provienen del texto asignado (no ASR)",
        "sin_normalizar_audio": True,
    }
    return {
        "sha256_script": sha256_archivo(Path(__file__)),
        "sha256_manifest": sha256_archivo(MANIFIESTO),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "praat_parselmouth": version("praat-parselmouth"),
        "webrtcvad_wheels": version("webrtcvad-wheels"),
        "parametros": parametros,
    }


# El avance se guarda por separado para poder continuar si la ejecucion se interrumpe
def guardar_parcial(filas, meta):
    exigir(not SALIDA.exists() and not CONFIG.exists(), "Ya existen archivos finales: no escribir checkpoint")
    # Usamos un solo JSON temporal para conservar juntas las mediciones y su informacion
    # asi evitamos recuperar datos que no correspondan a la misma ejecucion
    checkpoint = {"metadata": meta, "n_filas": len(filas),
                  "filas": [{c: convertir_a_celda(f[c]) for c in COLUMNAS} for f in filas]}
    contenido = (json.dumps(checkpoint, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    escritura_atomica(PARCIAL, contenido)


# Solo retomamos un avance si coinciden los archivos, parametros y version del script
def cargar_parcial(manifest, meta):
    if not PARCIAL.exists():
        return []
    anterior = json.loads(PARCIAL.read_text(encoding="utf-8"))
    exigir(anterior.get("metadata") == meta, "El checkpoint pertenece a otra version de datos, script o parametros")
    filas = anterior["filas"]
    exigir(len(filas) == anterior["n_filas"] and len(filas) <= 298, "Checkpoint tiene numero de filas invalido")
    for i, fila in enumerate(filas):
        ref = manifest[i]
        exigir(set(fila) == set(COLUMNAS), "Columnas del checkpoint incorrectas")
        exigir(all(fila[k] == ref[k] for k in
                    ("archivo", "participante_id", "familia_id", "enunciado", "toma", "sha256_audio")),
                f"Checkpoint no corresponde al manifest: {i}")
    return filas


# Antes de exportar comprobamos que cada grabacion tenga sus medidas y valores validos
def auditoria_final(filas, manifest):
    exigir(len(filas) == 298, "Resultado final no tiene 298 filas")
    exigir(len(set(f["archivo"] for f in filas)) == 298, "Archivos repetidos")
    total_alertas = Counter()
    for r, orig in zip(filas, manifest):
        exigir(r["archivo"] == orig["archivo"] and r["sha256_audio"] == orig["sha256_audio"], "Matriz resultante desordenada")
        exigir(int(r["n_palabras_texto"]) == PALABRAS[r["enunciado"]], "Numero de palabras inconsistente")
        exigir(int(r["n_tramas_f0"]) >= int(r["n_tramas_sonoras"]), "Conteos de F0 incompatibles")
        for k in ("duracion_total_s", "duracion_habla_estimada_s", "fraccion_habla_estimada", "tasa_lectura_total_pal_s"):
            exigir(math.isfinite(float(r[k])), f"Valor no finito: {k}")
        exigir(0 <= float(r["duracion_habla_estimada_s"]) <= float(r["duracion_total_s"]) + 1e-9,
                "Tiempo activo mayor que total")
        if r["f0_mediana_hz"] not in (None, ""):
            exigir(F0_MIN_HZ <= float(r["f0_mediana_hz"]) <= F0_MAX_HZ, "F0 fuera del rango")
        for alerta in filter(None, str(r["alertas_calidad"]).split(";")):
            total_alertas[alerta] += 1
    return total_alertas


# Esta funcion organiza la validacion de audios, la extraccion y la exportacion final
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-verificar", action="store_true", help="Comprueba entradas y dependencias sin crear resultados")
    parser.add_argument("--limite", type=int, default=None, help="Maximo de grabaciones NUEVAS en esta ejecucion")
    args = parser.parse_args()
    if args.limite is not None and args.limite <= 0:
        parser.error("--limite debe ser mayor que cero")
    exigir(RESULTADOS.is_dir(), f"Falta directorio privado: {RESULTADOS}")
    # Si ya hay resultados finales, detenemos la ejecucion para no reemplazarlos
    exigir(not SALIDA.exists() and not CONFIG.exists(), "El paso 5 ya tiene resultados finales: no sobrescribir")
    manifest = leer_manifest()
    meta = metadatos_checkpoint()
    print("=== PASO 5: CARACTERISTICAS ACUSTICAS ===", flush=True)
    print("Audios verificados con SHA-256:", len(manifest), flush=True)
    print("Palabras esperadas E01..E10:", [PALABRAS[f"E{i:02d}"] for i in range(1, 11)], flush=True)
    print(f"F0: Praat raw AC {F0_MIN_HZ:g}-{F0_MAX_HZ:g} Hz; VAD: WebRTC modo {VAD_MODO}, {VAD_TRAMA_MS} ms", flush=True)
    if args.solo_verificar:
        exigir(not PARCIAL.exists(), "Existe parcial; ejecutar modo normal para reanudar")
        print("VERIFICACION CORRECTA: no se generaron archivos.", flush=True)
        return
    anteriores = cargar_parcial(manifest, meta)
    print(f"Recuperadas del parcial: {len(anteriores)}/298", flush=True)
    restantes = manifest[len(anteriores):]
    pendientes = restantes[:args.limite] if args.limite is not None else restantes
    filas = list(anteriores)
    try:
        for ref in pendientes:
            filas.append(medir(ref))
            if len(filas) % GUARDAR_CADA == 0:
                guardar_parcial(filas, meta)
                print(f"Progreso: {len(filas)}/298", flush=True)
    finally:
        # Evitamos guardar un avance vacio si la ejecucion falla antes de medir algo
        if len(filas) > len(anteriores):
            guardar_parcial(filas, meta)
    if len(filas) < len(manifest):
        print(f"PRUEBA/PARCIAL: {len(filas)}/298; ejecutar sin --limite para continuar.", flush=True)
        return
    alertas = auditoria_final(filas, manifest)
    # Guardamos la configuracion y el CSV final sin reemplazar archivos existentes
    meta_final = {**meta, "fecha_utc": datetime.now(timezone.utc).isoformat(),
                  "n_grabaciones": 298, "n_alertas_por_tipo": dict(sorted(alertas.items())),
                  "observacion": "La actividad de habla y las palabras/s son estimaciones, no alineacion manual ni transcripcion."}
    escritura_atomica(CONFIG, (json.dumps(meta_final, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    escritura_atomica(SALIDA, csv_bytes(filas))
    # El avance temporal se elimina solo despues de guardar correctamente ambas salidas
    PARCIAL.unlink(missing_ok=True)
    print("COMPLETADO: caracteristicas_audio.csv (298 filas)", flush=True)
    print("Configuracion: caracteristicas_audio_config.json", flush=True)
    print("Alertas de calidad (no son fallos automaticos):", dict(sorted(alertas.items())), flush=True)
    print("PASO 5: extraccion completa; pendiente auditoria e inspeccion de alertas.", flush=True)


if __name__ == "__main__":
    main()
