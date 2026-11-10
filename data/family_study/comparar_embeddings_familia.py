#!/usr/bin/env python3
"""Calcula la similitud coseno entre los embeddings de las voces del estudio

Lee las comparaciones definidas, los vectores de ECAPA y sus manifiestos
Guarda similitudes individuales y resumenes por enunciado y por pareja
No vuelve a procesar audios ni interpreta las puntuaciones como parentesco
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "data" / "family_study"
RESULTADOS = BASE / "results"
PROTOCOL = RESULTADOS / "protocolo_comparaciones.csv"
EMBEDDINGS = RESULTADOS / "embeddings.npz"
MANIFEST = RESULTADOS / "embeddings_manifest.csv"
CONFIG = RESULTADOS / "embeddings_config.json"
PATRON = re.compile(r"^(F\d{2}_P\d{2})_E(\d{2})_T(0[12])\.wav$")
NOMENCLATURA = re.compile(r"^E(?:0[1-9]|10)$")
DESTINOS = {
    "individual": RESULTADOS / "ecapa_comparaciones.csv",
    "enunciados": RESULTADOS / "ecapa_resumen_enunciados.csv",
    "parejas": RESULTADOS / "ecapa_resumen_parejas.csv",
}
CAMPOS_BASE = ["id_comparacion", "grupo", "pareja_familiar_referencia", "participante_a", "participante_b"]
CAMPOS_INDIVIDUALES = CAMPOS_BASE + [
    "enunciado", "toma_a", "toma_b", "archivo_a", "archivo_b", "similitud_coseno"
]
CAMPOS_METRICAS = [
    "media_coseno", "mediana_coseno", "desviacion_estandar_coseno",
    "minimo_coseno", "maximo_coseno", "q1_coseno", "q3_coseno", "rango_intercuartil_coseno"
]
CAMPOS_ENUNCIADOS = CAMPOS_BASE + ["enunciado", "n_puntuaciones"] + CAMPOS_METRICAS
CAMPOS_PAREJAS = CAMPOS_BASE + ["n_enunciados", "n_puntuaciones"] + CAMPOS_METRICAS
ESPERADOS = {"misma_persona": 149, "hermanos": 236, "control": 472}


# Si alguna validacion falla, detenemos el calculo en lugar de generar tablas incompletas
def asegurar(condicion, mensaje):
    if not condicion:
        raise ValueError(mensaje)


# Leemos el CSV y conservamos sus encabezados para revisar las columnas
def leer_csv(ruta):
    asegurar(ruta.is_file(), f"No se encuentra: {ruta}")
    with ruta.open("r", encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        asegurar(lector.fieldnames is not None, f"CSV vacio: {ruta}")
        return list(lector), lector.fieldnames


# Antes de usar una tabla revisamos que no falte ninguna columna requerida
def requerir_columnas(campos, obligatorios, origen):
    faltantes = set(obligatorios) - set(campos)
    asegurar(not faltantes, f"Columnas faltantes en {origen}: {sorted(faltantes)}")


# Aceptamos distintas formas de escribir los grupos y las llevamos al mismo nombre
def grupo_normalizado(valor):
    texto = unicodedata.normalize("NFKD", valor.strip().lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[\s-]+", "_", texto)
    aliases = {
        "misma_persona": "misma_persona", "misma": "misma_persona",
        "mismo_hablante": "misma_persona", "mismo_participante": "misma_persona",
        "intrapersona": "misma_persona", "intrapersonal": "misma_persona",
        "hermanos": "hermanos", "hermano": "hermanos",
        "gemelos": "hermanos", "siblings": "hermanos",
        "control": "control", "controles": "control",
        "no_relacionados": "control", "sin_parentesco": "control",
    }
    asegurar(texto in aliases, f"Grupo del protocolo no reconocido: {valor!r}")
    return aliases[texto]


# Convertimos la lista de enunciados del CSV en identificadores ordenados
def parsear_enunciados(texto, etiqueta):
    tokens = [t.strip().upper() for t in re.split(r"[;,\s]+", texto) if t.strip()]
    asegurar(tokens, f"Sin enunciados: {etiqueta}")
    asegurar(all(NOMENCLATURA.fullmatch(e) for e in tokens), f"Enunciado no valido en {etiqueta}: {tokens}")
    asegurar(len(tokens) == len(set(tokens)), f"Enunciados duplicados: {etiqueta}")
    asegurar(tokens == sorted(tokens), f"Enunciados desordenados: {etiqueta}")
    return tokens


# La huella del WAV debe coincidir con la registrada al extraer los embeddings
def verificar_archivo(ruta, esperado):
    with ruta.open("rb") as f:
        real = hashlib.file_digest(f, "sha256").hexdigest()
    asegurar(real == esperado, f"Archivo WAV cambiado despues del paso 3: {ruta.name}")


# Cargamos los 298 embeddings, revisamos su origen y normalizamos cada vector
def cargar_embeddings():
    asegurar(EMBEDDINGS.is_file() and CONFIG.is_file(), "Faltan embeddings.npz o embeddings_config.json")
    with np.load(EMBEDDINGS, allow_pickle=False) as z:
        requerir_columnas(z.files, ["archivos", "embeddings", "sha256_audio", "sha256_modelo", "sha256_script"], EMBEDDINGS.name)
        nombres = list(z["archivos"].tolist())
        vectores = np.asarray(z["embeddings"], dtype=np.float64)
        hashes = list(z["sha256_audio"].tolist())
        hash_modelo = str(z["sha256_modelo"].item())
        hash_script = str(z["sha256_script"].item())
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    asegurar(cfg.get("cantidad") == 298 and cfg.get("dimensiones") == 192, "Configuracion del paso 3 incompatible")
    asegurar(cfg.get("sha256_modelo") == hash_modelo and cfg.get("sha256_script") == hash_script,
            "Huellas del modelo/script no coinciden con el paso 3")
    asegurar(vectores.shape == (298, 192), f"Dimensiones inesperadas: {vectores.shape}")
    asegurar(len(nombres) == len(set(nombres)) == len(hashes) == 298, "Archivos duplicados/faltantes en embeddings.npz")
    asegurar(np.isfinite(vectores).all(), "Embeddings con NaN o infinito")
    normas = np.linalg.norm(vectores, axis=1)
    asegurar(np.isfinite(normas).all() and (normas > 1e-12).all(), "Embedding nulo o invalido")
    vectores /= normas[:, np.newaxis]
    filas_manifest, campos = leer_csv(MANIFEST)
    requerir_columnas(campos, ["archivo", "participante_id", "familia_id", "enunciado", "toma", "sha256_audio"], MANIFEST.name)
    asegurar(len(filas_manifest) == 298, "Manifiesto no tiene 298 filas")
    indices = {}
    enunciados_disponibles = defaultdict(set)
    for i, (nombre, digest, fila) in enumerate(zip(nombres, hashes, filas_manifest)):
        m = PATRON.fullmatch(nombre)
        asegurar(m is not None, f"Nombre de audio inesperado: {nombre}")
        p, e, t = m[1], "E" + m[2], "T" + m[3]
        asegurar(fila["archivo"] == nombre and fila["participante_id"] == p
                and fila["familia_id"] == p.split("_")[0]
                and fila["enunciado"] == e and fila["toma"] == t
                and fila["sha256_audio"] == digest,
                f"Inconsistencia entre NPZ y manifiesto: {nombre}")
        wav = BASE / "processed" / nombre
        asegurar(wav.is_file(), f"Falta el audio: {nombre}")
        verificar_archivo(wav, digest)
        clave = (p, e, t)
        asegurar(clave not in indices, f"Grabacion duplicada: {clave}")
        indices[clave] = i
        enunciados_disponibles[p].add((e, t))
    asegurar(len(enunciados_disponibles) == 15, "No hay exactamente 15 participantes")
    completos = {}
    for persona, muestras in enunciados_disponibles.items():
        e_validos = {e for e, t in muestras if (e, "T01") in muestras and (e, "T02") in muestras}
        asegurar(len(muestras) == 2 * len(e_validos), f"Toma sin contraparte para {persona}")
        esperado = 9 if persona == "F04_P11" else 10
        asegurar(len(e_validos) == esperado, f"Enunciados incompletos para {persona}")
        completos[persona] = e_validos
    return nombres, vectores, indices, completos


# Comprobamos que cada comparacion use las personas y frases previstas
def cargar_protocolo(completos):
    filas, campos = leer_csv(PROTOCOL)
    obligatorias = CAMPOS_BASE + ["enunciados", "n_enunciados", "n_comparaciones_tomas"]
    requerir_columnas(campos, obligatorias, PROTOCOL.name)
    asegurar(len(filas) == 33, f"Se esperaban 33 definiciones y hay {len(filas)}")
    ids = set()
    grupos = Counter()
    parejas_por_ref = defaultdict(list)
    auto = set()
    hermanos = set()
    resultado = []
    for fila in filas:
        ident = fila["id_comparacion"].strip()
        asegurar(ident and ident not in ids, f"ID de comparacion vacio o duplicado: {ident!r}")
        ids.add(ident)
        grupo = grupo_normalizado(fila["grupo"])
        a, b = fila["participante_a"].strip(), fila["participante_b"].strip()
        asegurar(a in completos and b in completos, f"Participante desconocido en {ident}: {a}, {b}")
        enunciados = parsear_enunciados(fila["enunciados"], ident)
        enunciados_validos = completos[a] & completos[b]
        asegurar(set(enunciados) <= enunciados_validos,
                f"Enunciados no disponibles para ambos participantes en {ident}")
        if grupo == "misma_persona":
            asegurar(set(enunciados) == enunciados_validos,
                    f"La comparacion propia debe cubrir todas las frases validas: {ident}")
        asegurar(int(fila["n_enunciados"]) == len(enunciados), f"Conteo de enunciados incorrecto: {ident}")
        esperado = len(enunciados) if grupo == "misma_persona" else 4 * len(enunciados)
        asegurar(int(fila["n_comparaciones_tomas"]) == esperado,
                f"Cantidad de comparaciones incorrecta: {ident}")
        ref = fila["pareja_familiar_referencia"].strip()
        if grupo == "misma_persona":
            asegurar(a == b and a not in auto, f"Misma persona mal definida: {ident}")
            auto.add(a)
        else:
            asegurar(a != b, f"Comparacion distinta entre la misma persona: {ident}")
            asegurar(bool(ref), f"Referencia familiar vacia: {ident}")
            parejas_por_ref[ref].append((grupo, a, b, tuple(enunciados)))
            if grupo == "hermanos":
                asegurar(a.split("_")[0] == b.split("_")[0], f"Hermanos de familias distintas: {ident}")
                clave = frozenset((a, b))
                asegurar(clave not in hermanos, f"Pareja de hermanos duplicada: {ident}")
                hermanos.add(clave)
            else:
                asegurar(a.split("_")[0] != b.split("_")[0], f"Control de la misma familia: {ident}")
        grupos[grupo] += 1
        resultado.append({"id": ident, "grupo": grupo, "ref": ref, "a": a, "b": b, "enunciados": enunciados})
    asegurar(grupos == Counter({"misma_persona": 15, "hermanos": 6, "control": 12}),
            f"Grupos del protocolo inesperados: {dict(grupos)}")
    asegurar(auto == set(completos), "No hay una comparacion propia para cada participante")
    asegurar(len(parejas_por_ref) == 6, f"Se esperaban 6 referencias de hermanos; hay {len(parejas_por_ref)}")
    for ref, elementos in parejas_por_ref.items():
        sibs = [(a, b, en) for g, a, b, en in elementos if g == "hermanos"]
        ctrls = [(a, b, en) for g, a, b, en in elementos if g == "control"]
        asegurar(len(sibs) == 1 and len(ctrls) == 2, f"Trio mal definido en: {ref}")
        x, y, en_base = sibs[0]
        asegurar(all(en == en_base for _, _, en in ctrls), f"Enunciados distintos dentro del trio: {ref}")
        # La pareja de F04 tiene nueve enunciados porque falta E08 de F04_P11
        externos = set(sum(([a, b] for a, b, _ in ctrls), [])) - {x, y}
        asegurar(len(externos) == 1, f"No hay un solo control externo en: {ref}")
        control = next(iter(externos))
        esperados = {frozenset((x, control)), frozenset((y, control))}
        observados = {frozenset((a, b)) for a, b, _ in ctrls}
        asegurar(observados == esperados, f"Los controles no corresponden al trio: {ref}")
        comunes = completos[x] & completos[y] & completos[control]
        asegurar(set(en_base) == comunes, f"El trio {ref} no usa los enunciados comunes")
        n_esperado = 9 if "F04_P11" in (x, y) else 10
        asegurar(len(en_base) == n_esperado, f"Frases incompletas en el trio {ref}")
    asegurar(not any(frozenset((e["a"], e["b"])) == frozenset(("F02_P03", "F02_P07"))
                    for e in resultado if e["grupo"] == "hermanos"),
            "P03-P07 se excluyeron expresamente de los pares principales")
    return resultado


# Calculamos medidas como media, mediana y dispersion para resumir las similitudes
def medidas(valores):
    arr = np.asarray(valores, dtype=np.float64)
    asegurar(arr.ndim == 1 and arr.size > 0 and np.isfinite(arr).all(), "Medidas invalidas")
    q1, q3 = np.quantile(arr, [0.25, 0.75])
    # Con un solo valor no calculamos dispersion porque no hay variacion que estimar
    return {
        "media_coseno": float(np.mean(arr)),
        "mediana_coseno": float(np.median(arr)),
        "desviacion_estandar_coseno": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
        "minimo_coseno": float(np.min(arr)),
        "maximo_coseno": float(np.max(arr)),
        "q1_coseno": float(q1) if arr.size > 1 else None,
        "q3_coseno": float(q3) if arr.size > 1 else None,
        "rango_intercuartil_coseno": float(q3 - q1) if arr.size > 1 else None,
    }


# Recorremos las parejas definidas y calculamos la similitud de cada cruce de tomas
# Una puntuacion alta indica embeddings parecidos, no una probabilidad de parentesco
def elaborar(definiciones, nombres, vectores, indices):
    individuales = []
    por_enunciado = []
    por_pareja = []
    # Comparamos siempre la misma frase para reducir diferencias por el texto hablado
    for d in definiciones:
        base = dict(zip(CAMPOS_BASE, [d["id"], d["grupo"], d["ref"], d["a"], d["b"]]))
        medias_enunciados = []
        n_individuales = 0
        for e in d["enunciados"]:
            pares_tomas = [("T01", "T02")] if d["grupo"] == "misma_persona" else [
                ("T01", "T01"), ("T01", "T02"), ("T02", "T01"), ("T02", "T02")
            ]
            puntuaciones = []
            # Con dos personas cruzamos ambas tomas de cada una, y con una sola comparamos T01 con T02
            for ta, tb in pares_tomas:
                ia = indices[(d["a"], e, ta)]
                ib = indices[(d["b"], e, tb)]
                asegurar(ia != ib, f"Autocomparacion trivial detectada en {d['id']}: {e}")
                valor = float(vectores[ia] @ vectores[ib])
                asegurar(np.isfinite(valor) and -1.00000001 <= valor <= 1.00000001,
                        f"Similitud fuera de rango en {d['id']} {e}: {valor}")
                valor = min(1.0, max(-1.0, valor))
                puntuaciones.append(valor)
                individuales.append({**base, "enunciado": e, "toma_a": ta, "toma_b": tb,
                                     "archivo_a": nombres[ia], "archivo_b": nombres[ib],
                                     "similitud_coseno": valor})
            n_individuales += len(puntuaciones)
            calculos = medidas(puntuaciones)
            medias_enunciados.append(calculos["media_coseno"])
            por_enunciado.append({**base, "enunciado": e, "n_puntuaciones": len(puntuaciones), **calculos})
        # Primero promediamos las tomas de cada frase para que todas las frases pesen igual
        por_pareja.append({**base, "n_enunciados": len(medias_enunciados),
                           "n_puntuaciones": n_individuales, **medidas(medias_enunciados)})
    totales = Counter(f["grupo"] for f in individuales)
    asegurar(dict(totales) == ESPERADOS, f"Totales individuales inesperados: {dict(totales)}")
    asegurar(len(individuales) == 857, "No hay 857 puntuaciones")
    asegurar(len(por_enunciado) == 326, f"Se esperaban 326 resumenes por enunciado; hay {len(por_enunciado)}")
    asegurar(len(por_pareja) == 33, "No hay 33 resumenes por pareja")
    asegurar(sum(r["n_puntuaciones"] for r in por_enunciado) == 857, "Perdida de puntuaciones al agregar")
    asegurar(sum(r["n_puntuaciones"] for r in por_pareja) == 857, "Perdida de puntuaciones al resumir parejas")
    asegurar(all(r["n_puntuaciones"] == (1 if r["grupo"] == "misma_persona" else 4)
                for r in por_enunciado), "Numero de tomas incorrecto en resumenes por enunciado")
    asegurar(all(r["n_enunciados"] == (9 if r["participante_a"] == "F04_P11" else 10)
                for r in por_pareja if r["grupo"] == "misma_persona"),
            "Conteo incorrecto de enunciados en misma_persona")
    return individuales, por_enunciado, por_pareja


# Dejamos los numeros con una precision fija y los datos ausentes como celdas vacias
def dar_formato(valor):
    if valor is None:
        return ""
    if isinstance(valor, float):
        return f"{valor:.12f}"
    return valor


# Escribimos primero un archivo temporal para evitar una tabla final incompleta
def escribir_csv_atomicamente(ruta, filas, columnas):
    buffer = io.StringIO(newline="")
    w = csv.DictWriter(buffer, fieldnames=columnas, extrasaction="raise")
    w.writeheader()
    for fila in filas:
        w.writerow({k: dar_formato(fila[k]) for k in columnas})
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    with temporal.open("x", encoding="utf-8", newline="") as f:
        f.write(buffer.getvalue())
    return temporal


# Aqui cargamos y verificamos los datos, calculamos las tablas y decidimos si guardarlas
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-verificar", action="store_true", help="Validar datos y calculos sin generar CSV")
    args = parser.parse_args()
    asegurar(RESULTADOS.is_dir(), f"No existe carpeta privada: {RESULTADOS}")
    nombres, vectores, indices, completos = cargar_embeddings()
    definiciones = cargar_protocolo(completos)
    individuales, enunciados, parejas = elaborar(definiciones, nombres, vectores, indices)
    print("=== PASO 4: AUDITORIA DEL PROTOCOLO Y CALCULOS ===")
    print(f"Embeddings validados: {len(nombres)} (dimension 192)")
    print(f"Definiciones: {len(definiciones)}")
    for grupo in ("misma_persona", "hermanos", "control"):
        print(f"{grupo}: {sum(r['grupo'] == grupo for r in individuales)} similitudes")
    print(f"Comparaciones: {len(individuales)} | Resumenes por enunciado: {len(enunciados)} | Resumenes por pareja: {len(parejas)}")
    if args.solo_verificar:
        print("VERIFICACION CORRECTA: no se crearon archivos de salida.")
        return
    existentes = [p.name for p in DESTINOS.values() if p.exists()]
    asegurar(not existentes, f"No se sobrescribiran resultados previos: {existentes}")
    temporales = []
    creados = []
    try:
        for llave, registros, campos in (
            ("individual", individuales, CAMPOS_INDIVIDUALES),
            ("enunciados", enunciados, CAMPOS_ENUNCIADOS),
            ("parejas", parejas, CAMPOS_PAREJAS),
        ):
            temporales.append((escribir_csv_atomicamente(DESTINOS[llave], registros, campos), DESTINOS[llave]))
        for temporal, destino in temporales:
            os.replace(temporal, destino)
            creados.append(destino)
    except BaseException:
        for temporal, _ in temporales:
            temporal.unlink(missing_ok=True)
        for destino in creados:
            destino.unlink(missing_ok=True)
        raise
    for ruta, n in zip(DESTINOS.values(), (len(individuales), len(enunciados), len(parejas))):
        print(f"GUARDADO: {ruta.name} ({n} filas)")
    print("PASO 4: resultados generados y coherentes con el protocolo del Paso 2.")


if __name__ == "__main__":
    main()
