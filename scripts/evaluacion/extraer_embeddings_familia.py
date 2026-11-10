# EXTRACCION DE REPRESENTACIONES DE VOZ CON ECAPA-TDNN
#
# Aqui convertimos cada grabacion procesada en un vector de 192 valores
# ECAPA-TDNN obtiene una representacion numerica de la voz que despues
# podemos comparar con la de otros participantes
# Antes de procesar verificamos el formato y la integridad de los audios
# El avance se guarda cada cierto numero de archivos para poder retomarlo
# si la ejecucion se interrumpe, sin mezclar resultados de versiones distintas
# Este programa no determina si dos personas son familiares
import argparse
import csv
import hashlib
import json
import os
import re
import wave
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import SpeakerRecognition

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "family_study"
AUDIOS = DATA / "processed"
RESULTADOS = DATA / "results"
MODELO = REPO / "models" / "ecapa"
PARCIAL = RESULTADOS / "embeddings_parcial.npz"
FINAL = RESULTADOS / "embeddings.npz"
# El nombre del WAV permite identificar al participante, el enunciado y la toma
PATRON = re.compile(r"^(F\d{2}_P\d{2})_E(\d{2})_T(0[12])\.wav$")


# La huella SHA-256 nos permite comprobar si un archivo cambio desde la ultima ejecucion
def sha256_archivo(ruta):
    with ruta.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# Antes de usar el modelo verificamos que esten las grabaciones esperadas
# F04_P11 tiene 18 archivos porque faltan las dos tomas del enunciado E08
def catalogar():
    with (DATA / "metadata" / "participants.csv").open(encoding="utf-8-sig", newline="") as f:
        personas = {fila["participant_id"]: fila for fila in csv.DictReader(f)}
    assert len(personas) == 15, "Se esperaban 15 participantes"
    archivos = sorted(AUDIOS.glob("*.wav"))
    assert len(archivos) == 298, "Se esperaban 298 WAV procesados"
    conteo = Counter()
    for ruta in archivos:
        m = PATRON.fullmatch(ruta.name)
        assert m is not None, f"Nombre inesperado: {ruta.name}"
        assert m[1] in personas, f"Participante desconocido: {ruta.name}"
        assert 1 <= int(m[2]) <= 10, f"Enunciado invalido: {ruta.name}"
        conteo[m[1]] += 1
    for persona in personas:
        assert conteo[persona] == (18 if persona == "F04_P11" else 20), persona
    return archivos


# Leemos el audio de 16 kHz y convertimos sus muestras a valores entre -1 y 1
def leer_wav(ruta):
    with wave.open(str(ruta), "rb") as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getcomptype()) == (
            16000, 1, 2, "NONE"
        ), f"Formato WAV no valido: {ruta.name}"
        frames = f.getnframes()
        datos = f.readframes(frames)
    assert frames > 0 and len(datos) == frames * 2, f"Audio vacio o truncado: {ruta.name}"
    samples = np.frombuffer(datos, dtype="<i2").astype(np.float32) / 32768.0
    assert np.isfinite(samples).all(), f"Muestras no finitas: {ruta.name}"
    return torch.from_numpy(samples).unsqueeze(0)


# Calculamos una huella conjunta de los archivos que componen el modelo
# De esta forma evitamos reanudar una extraccion con pesos diferentes
def huella_modelo():
    resumen = hashlib.sha256()
    for nombre in (
        "embedding_model.ckpt", "hyperparams.yaml", "mean_var_norm_emb.ckpt",
        "classifier.ckpt", "label_encoder.ckpt"
    ):
        archivo = MODELO / nombre
        assert archivo.is_file(), f"Falta archivo del modelo: {archivo}"
        resumen.update(nombre.encode("utf-8"))
        resumen.update(sha256_archivo(archivo).encode("ascii"))
    return resumen.hexdigest()


# Primero guardamos el NPZ completo en un temporal y despues reemplazamos el destino
# Esto evita que una interrupcion deje un archivo parcial con apariencia de resultado final
def guardar_npz(ruta, nombres, vectores, hashes, huella, version_script):
    temporal = ruta.with_suffix(".npz.tmp")
    matriz = np.stack(vectores).astype(np.float32) if vectores else np.empty((0, 192), dtype=np.float32)
    with temporal.open("wb") as f:
        np.savez_compressed(
            f,
            archivos=np.asarray(nombres, dtype=str),
            embeddings=matriz,
            sha256_audio=np.asarray(hashes, dtype=str),
            sha256_modelo=np.asarray(huella),
            sha256_script=np.asarray(version_script),
        )
    os.replace(temporal, ruta)


# Para continuar un avance anterior comprobamos los audios, el modelo y la version del codigo
def cargar_parcial(archivos, huella, version_script):
    if not PARCIAL.exists():
        return [], [], []
    with np.load(PARCIAL, allow_pickle=False) as z:
        nombres = z["archivos"].tolist()
        vectores = z["embeddings"]
        hashes = z["sha256_audio"].tolist()
        registrada = str(z["sha256_modelo"].item())
        registrada_script = str(z["sha256_script"].item())
    assert registrada == huella, "Los pesos del modelo cambiaron; no reutilizar el parcial"
    assert registrada_script == version_script, "El script cambio; revisar el parcial antes de reanudar"
    assert nombres == [p.name for p in archivos[:len(nombres)]], "El dataset cambio de orden/contenido"
    assert vectores.shape == (len(nombres), 192) and len(hashes) == len(nombres)
    assert np.isfinite(vectores).all(), "Checkpoint invalido"
    for ruta, hash_esperado in zip(archivos[:len(nombres)], hashes):
        assert sha256_archivo(ruta) == hash_esperado, f"Cambio el WAV: {ruta.name}"
    return nombres, [fila.copy() for fila in vectores], hashes


# Guardamos que vector corresponde a cada WAV junto con su SHA-256 y su duracion
def registrar(nombres, hashes, huella, version_script):
    manifiesto = RESULTADOS / "embeddings_manifest.csv"
    configuracion = RESULTADOS / "embeddings_config.json"
    campos = ["archivo", "participante_id", "familia_id", "enunciado", "toma", "duracion_s", "sha256_audio"]
    with manifiesto.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for nombre, digest in zip(nombres, hashes):
            m = PATRON.fullmatch(nombre)
            with wave.open(str(AUDIOS / nombre), "rb") as wav:
                duracion = wav.getnframes() / wav.getframerate()
            w.writerow({
                "archivo": nombre, "participante_id": m[1],
                "familia_id": m[1].split("_")[0],
                "enunciado": "E" + m[2], "toma": "T" + m[3],
                "duracion_s": f"{duracion:.6f}", "sha256_audio": digest,
            })
    configuracion.write_text(json.dumps({
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "modelo": "speechbrain/spkrec-ecapa-voxceleb",
        "fuente_modelo": "models/ecapa (archivos locales)",
        "sha256_modelo": huella,
        "sha256_script": version_script,
        "python_torch": torch.__version__,
        "speechbrain": version("speechbrain"),
        "numpy": np.__version__,
        "dispositivo": "cpu", "dimensiones": 192, "cantidad": len(nombres),
        "lectura": "wave; PCM 16 kHz, mono, 16 bits; escala muestra/32768",
        "extraccion": "SpeakerRecognition.encode_batch, torch.inference_mode",
        "transformaciones_adicionales": "ninguna",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Esta funcion comprueba las entradas, carga ECAPA y procesa los audios pendientes
# Si el resultado final ya existe, el programa se detiene para no sobrescribirlo
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limite", type=int, default=None, help="Numero maximo de archivos NUEVOS en esta ejecucion (para prueba)")
    args = parser.parse_args()
    if args.limite is not None and args.limite <= 0:
        parser.error("--limite debe ser positivo")
    assert not FINAL.exists(), f"Ya existe el resultado final: {FINAL} (no se sobrescribe)"
    archivos = catalogar()
    assert RESULTADOS.is_dir(), "Falta la carpeta privada results del paso 2"
    huella = huella_modelo()
    # El checkpoint tambien registra el hash del script para detectar cambios en el codigo
    version_script = sha256_archivo(Path(__file__))
    nombres, vectores, hashes = cargar_parcial(archivos, huella, version_script)
    print(f"Archivos: {len(archivos)} | Recuperados del parcial: {len(nombres)}", flush=True)
    if not (restantes := archivos[len(nombres):]):
        print("Ya hay 298 vectores en el parcial. Completando exportacion.")
    else:
        pendientes = restantes[:args.limite] if args.limite is not None else restantes
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        modelo = SpeakerRecognition.from_hparams(
            source=str(MODELO), savedir=str(MODELO), run_opts={"device": "cpu"}
        )
        if hasattr(modelo, "eval"):
            modelo.eval()
        try:
            # Procesamos una grabacion a la vez y guardamos el avance cada diez archivos
            for ruta in pendientes:
                senal = leer_wav(ruta)
                with torch.inference_mode():
                    emb = modelo.encode_batch(senal).squeeze().detach().cpu().numpy()
                emb = np.asarray(emb, dtype=np.float32)
                assert emb.shape == (192,) and np.isfinite(emb).all(), f"Embedding invalido: {ruta.name}"
                assert float(np.linalg.norm(emb)) > 0, f"Embedding nulo: {ruta.name}"
                nombres.append(ruta.name)
                vectores.append(emb)
                hashes.append(sha256_archivo(ruta))
                if len(nombres) % 10 == 0:
                    guardar_npz(PARCIAL, nombres, vectores, hashes, huella, version_script)
                    print(f"Progreso: {len(nombres)}/{len(archivos)}", flush=True)
        finally:
            if nombres:
                guardar_npz(PARCIAL, nombres, vectores, hashes, huella, version_script)
    if len(nombres) < len(archivos):
        print(f"PRUEBA/PARCIAL: {len(nombres)}/{len(archivos)}. Ejecute sin --limite para continuar.")
        return
    assert len(nombres) == len(archivos) == 298
    guardar_npz(FINAL, nombres, vectores, hashes, huella, version_script)
    registrar(nombres, hashes, huella, version_script)
    PARCIAL.unlink(missing_ok=True)
    print(f"COMPLETADO: {FINAL} | Matriz: (298, 192)")
    print("Manifest y configuracion guardados en results/.")


if __name__ == "__main__":
    main()
