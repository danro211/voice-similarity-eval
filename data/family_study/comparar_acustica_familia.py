#!/usr/bin/env python3
"""Calcula diferencias acusticas entre las voces de las parejas del estudio

Lee las mediciones individuales, las comparaciones definidas y las tablas ECAPA
Genera resultados por tomas, enunciados, parejas y grupos familiares
Las distancias menores indican mayor semejanza en cada caracteristica
Conserva las alertas de calidad y no modifica las grabaciones ni sus medidas
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
RES = BASE / "results"
ENTRADAS = {
    "protocolo": RES / "protocolo_comparaciones.csv",
    "ecapa": RES / "ecapa_comparaciones.csv",
    "acustica": RES / "caracteristicas_audio.csv",
    "config_p5": RES / "caracteristicas_audio_config.json",
    "manifiesto": RES / "embeddings_manifest.csv",
}
SALIDAS = {
    "individual": RES / "acustica_comparaciones.csv",
    "enunciado": RES / "acustica_resumen_enunciados.csv",
    "pareja": RES / "acustica_resumen_parejas.csv",
    "trios": RES / "acustica_contrastes_trios.csv",
    "config": RES / "acustica_comparaciones_config.json",
}
PATRON = re.compile(r"^(F\d{2}_P\d{2})_(E(?:0[1-9]|10))_(T0[12])\.wav$")
GRUPOS = ("misma_persona", "hermanos", "control")
ESPERADO = {"misma_persona": 149, "hermanos": 236, "control": 472}
N_ENUNCIADOS = {"misma_persona": 149, "hermanos": 59, "control": 118}
UMBRAL_IQR_F0_ST = 6.0
UMBRAL_DESV_F0_ST = 6.0

# Las distancias son simetricas: intercambiar a los participantes no cambia el valor
# Para la diferencia relativa usamos 200*|x-y|/(x+y)
METRICAS = (
    "delta_f0_st", "delta_rango_f0_st", "delta_rango_f0_st_sin_alertas",
    "delta_duracion_total_s", "delta_duracion_total_pct",
    "delta_habla_estimada_s", "delta_habla_estimada_pct",
    "delta_fraccion_habla_pp", "delta_velocidad_total_pal_s",
    "delta_velocidad_articulacion_pal_s",
)
BASIS = ("id_comparacion", "grupo", "pareja_familiar_referencia", "participante_a", "participante_b")
INDIVIDUAL = list(BASIS) + [
    "enunciado", "toma_a", "toma_b", "archivo_a", "archivo_b",
    "alertas_a", "alertas_b", "f0_alertada", "dispersion_f0_alertada",
    "vad_alertada", "f0_rango_st_a", "f0_rango_st_b",
] + list(METRICAS)
ESTADISTICAS = ("media", "mediana", "desv", "min", "max")
RESUMEN_CAMPOS = [f"{stat}_{m}" for m in METRICAS for stat in ESTADISTICAS]
ENUNCIADO = list(BASIS) + ["enunciado", "n_puntuaciones", "n_f0_alertadas", "n_dispersion_f0_alertadas", "n_vad_alertadas"] + [f"n_{m}" for m in METRICAS] + RESUMEN_CAMPOS
PAREJA = list(BASIS) + ["n_enunciados", "n_puntuaciones", "n_enunciados_f0_alertados", "n_enunciados_dispersion_f0_alertados", "n_enunciados_vad_alertados"] + [f"n_enunciados_{m}" for m in METRICAS] + RESUMEN_CAMPOS
TRIOS = ["pareja_familiar_referencia", "participante_hermano_a", "participante_hermano_b", "participante_control", "n_enunciados_diseno"] + [f"{campo}_{m}" for m in METRICAS for campo in ("n_enunciados_comunes", "media_hermanos", "media_controles", "contraste_hermanos_menos_controles")]


# Una entrada incoherente detiene el programa antes de generar resultados
def asegurar(ok, msg):
    if not ok:
        raise ValueError(msg)


# Leemos las filas del CSV y revisamos que sus columnas sean unicas
def leer_csv(ruta):
    asegurar(ruta.is_file(), f"Falta archivo: {ruta}")
    with ruta.open(encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        asegurar(lector.fieldnames is not None, f"CSV sin cabeceras: {ruta}")
        asegurar(len(lector.fieldnames) == len(set(lector.fieldnames)), f"Cabeceras duplicadas: {ruta}")
        return list(lector), set(lector.fieldnames)


# Revisamos que cada tabla tenga las columnas que vamos a utilizar
def columnas(campos, esperadas, origen):
    faltantes = set(esperadas) - campos
    asegurar(not faltantes, f"Columnas faltantes en {origen}: {sorted(faltantes)}")


# Calculamos su SHA-256 para verificar que las tablas no cambiaron
def digest(ruta):
    with ruta.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# Convertimos el valor a numero y comprobamos si se permiten datos ausentes
def numero(texto, etiqueta, *, permite_vacio=False, minimo=None, maximo=None):
    if texto is None or texto == "":
        asegurar(permite_vacio, f"Valor ausente: {etiqueta}")
        return None
    try:
        v = float(texto)
    except (ValueError, TypeError):
        raise ValueError(f"No es numero: {etiqueta}: {texto!r}") from None
    asegurar(math.isfinite(v), f"Numero no finito: {etiqueta}")
    if minimo is not None:
        asegurar(v >= minimo - 1e-8, f"Debajo del minimo: {etiqueta}")
    if maximo is not None:
        asegurar(v <= maximo + 1e-8, f"Por encima del maximo: {etiqueta}")
    return v


# Revisamos los conteos para evitar cantidades negativas o invalidas
def entero(texto, etiqueta, *, minimo=0):
    try:
        n = int(texto)
    except (ValueError, TypeError):
        raise ValueError(f"No es entero: {etiqueta}={texto!r}") from None
    asegurar(n >= minimo, f"Entero invalido: {etiqueta}")
    return n


# Comparamos valores con tolerancia para evitar diferencias por redondeo
def cerca(a, b, label):
    asegurar(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7), f"Incoherencia numerica en {label}: {a} vs {b}")


# Comprobamos que las tablas de entrada hablen de las mismas tomas y personas
# Si no coinciden sus identificadores o cantidades, no continuamos
def consultar_entradas():
    asegurar(RES.is_dir(), f"Carpeta privada ausente: {RES}")
    for ruta in ENTRADAS.values():
        asegurar(ruta.is_file(), f"Falta entrada: {ruta}")
    cfg = json.loads(ENTRADAS["config_p5"].read_text(encoding="utf-8"))
    asegurar(cfg.get("n_grabaciones") == 298, "Configuracion paso 5 incompatible")
    asegurar(cfg.get("sha256_manifest") == digest(ENTRADAS["manifiesto"]), "Manifiesto distinto al usado en paso 5")
    asegurar(cfg.get("sha256_script") == digest(BASE / "extraer_caracteristicas_familia.py"), "Script paso 5 cambiado despues de la extraccion")
    asegurar("parametros" in cfg, "Configuracion F0/VAD no disponible")
    manifest, fields = leer_csv(ENTRADAS["manifiesto"])
    columnas(fields, ("archivo", "participante_id", "familia_id", "enunciado", "toma", "sha256_audio"), "manifest")
    asegurar(len(manifest) == 298, "Manifest no tiene 298 audios")
    ref = {}
    for r in manifest:
        nombre = r["archivo"]
        m = PATRON.fullmatch(nombre)
        asegurar(m is not None and nombre not in ref, f"Manifest duplicado o incorrecto: {nombre}")
        asegurar((r["participante_id"], r["familia_id"], r["enunciado"], r["toma"]) ==
                (m[1], m[1].split("_")[0], m[2], m[3]), f"Metadatos incorrectos: {nombre}")
        asegurar(re.fullmatch(r"[a-f0-9]{64}", r["sha256_audio"]) is not None, f"Hash incorrecto: {nombre}")
        ref[nombre] = r
    audios, fields = leer_csv(ENTRADAS["acustica"])
    columnas(fields, (
        "archivo", "participante_id", "familia_id", "enunciado", "toma", "sha256_audio", "duracion_total_s",
        "duracion_habla_estimada_s", "fraccion_habla_estimada", "n_palabras_texto", "f0_mediana_hz", "f0_q1_hz",
        "f0_q3_hz", "f0_ric_hz", "f0_desv_semitonos", "n_tramas_f0", "n_tramas_sonoras",
        "fraccion_tramas_sonoras", "tasa_lectura_total_pal_s", "tasa_articulacion_aprox_pal_s", "alertas_calidad",
    ), "caracteristicas_audio")
    asegurar(len(audios) == 298, "Paso 5 no contiene exactamente 298 mediciones")
    muestras = {}
    counts = Counter()
    conteo_alertas = Counter()
    palabras = cfg["parametros"].get("palabras_esperadas_por_enunciado", {})
    for r in audios:
        nombre = r["archivo"]
        asegurar(nombre in ref and nombre not in muestras, f"Audio duplicado o ausente en manifest: {nombre}")
        m = ref[nombre]
        asegurar(all(r[k] == m[k] for k in ("participante_id", "familia_id", "enunciado", "toma", "sha256_audio")), f"Audio desalineado: {nombre}")
        counts[r["participante_id"]] += 1
        dur = numero(r["duracion_total_s"], f"{nombre}/duracion", minimo=0)
        voz = numero(r["duracion_habla_estimada_s"], f"{nombre}/habla", minimo=0)
        asegurar(dur > 0 and voz <= dur + 1e-7, f"VAD mayor que duracion: {nombre}")
        frac = numero(r["fraccion_habla_estimada"], f"{nombre}/fraccion_vad", minimo=0, maximo=1)
        cerca(frac, voz / dur, f"{nombre}/fraccion_vad")
        palabras_n = entero(r["n_palabras_texto"], f"{nombre}/palabras", minimo=1)
        asegurar(palabras_n == int(palabras[r["enunciado"]]), f"Palabras fuera del protocolo: {nombre}")
        total_vel = numero(r["tasa_lectura_total_pal_s"], f"{nombre}/tasa_total", minimo=0)
        cerca(total_vel, palabras_n / dur, f"{nombre}/tasa_total")
        art = numero(r["tasa_articulacion_aprox_pal_s"], f"{nombre}/tasa_art", permite_vacio=True, minimo=0)
        if voz > 0:
            asegurar(art is not None, f"Tasa de articulacion ausente: {nombre}")
            cerca(art, palabras_n / voz, f"{nombre}/tasa_art")
        else:
            asegurar(art is None, f"Tasa de articulacion presente sin habla: {nombre}")
        n = entero(r["n_tramas_f0"], f"{nombre}/tramas_f0", minimo=1)
        nv = entero(r["n_tramas_sonoras"], f"{nombre}/tramas_f0_validas")
        asegurar(nv <= n, f"F0 inconsistente: {nombre}")
        cobertura = numero(r["fraccion_tramas_sonoras"], f"{nombre}/cobertura", minimo=0, maximo=1)
        cerca(cobertura, nv / n, f"{nombre}/cobertura")
        datos_f0 = [numero(r[k], f"{nombre}/{k}", permite_vacio=True) for k in ("f0_q1_hz", "f0_mediana_hz", "f0_q3_hz", "f0_ric_hz", "f0_desv_semitonos")]
        q1, med, q3, ric, desv = datos_f0
        if nv:
            asegurar(all(v is not None for v in (q1, med, q3, ric)), f"F0 incompleta: {nombre}")
            asegurar(60 - 1e-6 <= q1 <= med <= q3 <= 700 + 1e-6, f"Cuartiles F0 invalidos: {nombre}")
            cerca(ric, q3 - q1, f"{nombre}/IQR_F0")
            asegurar(desv is None or desv >= 0, f"Dispersion F0 negativa: {nombre}")
        else:
            asegurar(all(v is None for v in datos_f0), f"F0 presente sin tramas sonoras: {nombre}")
        alertas = set(filter(None, (r["alertas_calidad"] or "").split(";")))
        for a in alertas:
            conteo_alertas[a] += 1
        flag_f0 = bool(any(a.startswith("f0_") for a in alertas))
        rango_st = 12 * math.log2(q3 / q1) if q1 is not None and q1 > 0 else None
        flag_disp = flag_f0 or (rango_st is None) or (rango_st > UMBRAL_IQR_F0_ST) or (desv is not None and desv > UMBRAL_DESV_F0_ST)
        flag_vad = any(a.startswith("vad_") or a == "tasa_articulacion_no_calculable" for a in alertas)
        muestras[nombre] = {"archivo": nombre, "fila": r, "dur": dur, "voz": voz, "frac": frac,
                           "f0": med, "rango_st": rango_st, "vel_total": total_vel, "vel_art": art,
                           "alertas": ";".join(sorted(alertas)), "f0_alerta": flag_f0,
                           "disp_alerta": flag_disp, "vad_alerta": flag_vad}
    asegurar(set(muestras) == set(ref), "WAV de caracteristicas y manifest no coinciden")
    asegurar(len(counts) == 15 and all(n == (18 if pid == "F04_P11" else 20) for pid, n in counts.items()), "Distribucion de audios inesperada")
    asegurar(dict(conteo_alertas) == cfg["n_alertas_por_tipo"], "Alertas incompatibles con paso 5")
    protocolo, fields = leer_csv(ENTRADAS["protocolo"])
    columnas(fields, (*BASIS, "enunciados", "n_enunciados", "n_comparaciones_tomas"), "protocolo")
    asegurar(len(protocolo) == 33, "Protocolo no tiene 33 definiciones")
    defs = {}
    grupos = Counter()
    por_trio = defaultdict(list)
    self_part = set()
    for row in protocolo:
        id_ = row["id_comparacion"]
        g = row["grupo"]
        asegurar(g in GRUPOS and id_ and id_ not in defs, f"Grupo o ID invalidos: {id_}")
        a, b, ref_f = (row[k] for k in ("participante_a", "participante_b", "pareja_familiar_referencia"))
        ens = row["enunciados"].split(";")
        asegurar(ens and len(ens) == len(set(ens)) and ens == sorted(ens) and all(re.fullmatch(r"E(?:0[1-9]|10)", x) for x in ens), f"Enunciados invalidos: {id_}")
        esperado = len(ens) * (1 if g == "misma_persona" else 4)
        asegurar(entero(row["n_enunciados"], "n_enunciados") == len(ens) and entero(row["n_comparaciones_tomas"], "n_tomas") == esperado, f"Conteos incorrectos: {id_}")
        asegurar(all(f"{p}_{e}_{t}.wav" in muestras for p in (a,b) for e in ens for t in ("T01", "T02")), f"Audio faltante: {id_}")
        if g == "misma_persona":
            asegurar(a == b and a not in self_part, f"Duplicado misma_persona: {id_}")
            self_part.add(a)
        else:
            asegurar(a != b and ref_f, f"Par distinto invalido: {id_}")
            if g == "hermanos":
                asegurar(a.split("_")[0] == b.split("_")[0], f"Hermanos familias distintas: {id_}")
            else:
                asegurar(a.split("_")[0] != b.split("_")[0], f"Control de misma familia: {id_}")
            por_trio[ref_f].append((g,a,b,tuple(ens)))
        grupos[g] += 1
        defs[id_] = {"id": id_, "grupo": g, "a": a, "b": b, "ref": ref_f, "ens": ens}
    asegurar(grupos == Counter({"misma_persona": 15, "hermanos": 6, "control": 12}), "Numero de definiciones por grupo incorrecto")
    asegurar(len(self_part) == 15 and len(por_trio) == 6, "Trio/participantes incompletos")
    for ref_f, regs in por_trio.items():
        h = [(a,b,e) for g,a,b,e in regs if g == "hermanos"]
        c = [(a,b,e) for g,a,b,e in regs if g == "control"]
        asegurar(len(h) == 1 and len(c) == 2, f"Trio incompleto: {ref_f}")
        a,b,ens = h[0]
        control_ids = set(sum(([x,y] for x,y,_ in c), [])) - {a,b}
        asegurar(len(control_ids) == 1, f"No existe control unico: {ref_f}")
        ctr = control_ids.pop()
        asegurar({frozenset((x,y)) for x,y,_ in c} == {frozenset((a,ctr)),frozenset((b,ctr))}, f"Control incorrecto: {ref_f}")
        asegurar(all(en == ens for _,_,en in c), f"Enunciados no pareados: {ref_f}")
        asegurar(len(ens) == (9 if "F04_P11" in (a,b) else 10), f"Enunciados incorrectos trio {ref_f}")
    ecapa, fields = leer_csv(ENTRADAS["ecapa"])
    columnas(fields, (*BASIS, "enunciado", "toma_a", "toma_b", "archivo_a", "archivo_b", "similitud_coseno"), "ecapa")
    asegurar(len(ecapa) == 857, "ECAPA no tiene 857 comparaciones")
    esperado_keys = set()
    for d in defs.values():
        tomas = [("T01","T02")] if d["grupo"] == "misma_persona" else [(x,y) for x in ("T01","T02") for y in ("T01","T02")]
        for e in d["ens"]:
            for ta,tb in tomas:
                esperado_keys.add((d["id"],e,ta,tb))
    asegurar(len(esperado_keys) == 857, "Protocolo no genera 857 comparaciones")
    seen=set(); conteos_ecapa=Counter()
    for row in ecapa:
        id_,e,ta,tb=(row[k] for k in ("id_comparacion","enunciado","toma_a","toma_b"))
        d=defs.get(id_)
        asegurar(d is not None and (id_,e,ta,tb) in esperado_keys and (id_,e,ta,tb) not in seen, f"Comparacion no prevista/duplicada: {id_} {e} {ta} {tb}")
        seen.add((id_,e,ta,tb))
        asegurar(all(row[k] == expected for k, expected in zip(BASIS,(id_,d["grupo"],d["ref"],d["a"],d["b"]))), f"Referencia ECAPA alterada: {id_}")
        asegurar(row["archivo_a"] == f'{d["a"]}_{e}_{ta}.wav' and row["archivo_b"] == f'{d["b"]}_{e}_{tb}.wav' and row["archivo_a"] != row["archivo_b"], f"WAV no coincide: {id_}")
        numero(row["similitud_coseno"], f"ECAPA {id_}", minimo=-1,maximo=1)
        conteos_ecapa[d["grupo"]]+=1
    asegurar(seen == esperado_keys and conteos_ecapa == Counter(ESPERADO), "Comparaciones ECAPA no completas")
    return muestras, ecapa, defs, cfg


# La diferencia absoluta permite comparar dos valores en la misma unidad
def absdiff(x,y):
    return None if x is None or y is None else abs(x-y)


# La diferencia porcentual usa como referencia el promedio de ambos valores
def simetrica_pct(x,y):
    if x is None or y is None or x+y<=0:
        return None
    return 200.0*abs(x-y)/(x+y)


# Pasamos la razon de frecuencias a semitonos para comparar registros de voz
# Una diferencia de 12 semitonos corresponde a una octava
def distancia_f0_st(x,y):
    if x is None or y is None or x<=0 or y<=0:
        return None
    return abs(12 * math.log2(x/y))


# Calculamos media, mediana y dispersion solo con mediciones disponibles
def estadisticas(vals):
    arr=np.asarray([x for x in vals if x is not None], dtype=np.float64)
    asegurar(np.isfinite(arr).all(), "Metricas con valor no finito")
    if not arr.size:
        return {"media":None, "mediana":None,"desv":None,"min":None,"max":None}
    return {"media":float(np.mean(arr)), "mediana":float(np.median(arr)),
            "desv":float(np.std(arr,ddof=1)) if arr.size>1 else None,
            "min":float(np.min(arr)), "max":float(np.max(arr))}


# Guardamos el resumen y cuantas observaciones sirvieron para calcularlo
def resumen_metricas(registros, campo_n):
    res={}
    for nombre in METRICAS:
        arr=[r[nombre] for r in registros if r[nombre] is not None]
        res[f"{campo_n}{nombre}"]=len(arr)
        for est, val in estadisticas(arr).items():
            res[f"{est}_{nombre}"]=val
    return res


# Comparamos tomas individuales y despues agrupamos por enunciado, pareja y familia
# Los dos hermanos se comparan con la misma persona de control
def ejecutar(muestras, ecapa, defs):
    indiv=[]
    acum_enunciado=defaultdict(list)
    cont_grupo=Counter()
    # Buscamos la misma grabacion en cada tabla antes de calcular una distancia
    for r in ecapa:
        a=muestras[r["archivo_a"]]; b=muestras[r["archivo_b"]]
        f0_flag=a["f0_alerta"] or b["f0_alerta"]
        disp_flag=a["disp_alerta"] or b["disp_alerta"]
        vad_flag=a["vad_alerta"] or b["vad_alerta"]
        delta_rango=absdiff(a["rango_st"], b["rango_st"])
        distancias = {
            "delta_f0_st": distancia_f0_st(a["f0"],b["f0"]),
            "delta_rango_f0_st": delta_rango,
            "delta_rango_f0_st_sin_alertas": None if disp_flag else delta_rango,
            "delta_duracion_total_s": absdiff(a["dur"],b["dur"]),
            "delta_duracion_total_pct": simetrica_pct(a["dur"],b["dur"]),
            "delta_habla_estimada_s": absdiff(a["voz"],b["voz"]),
            "delta_habla_estimada_pct": simetrica_pct(a["voz"],b["voz"]),
            "delta_fraccion_habla_pp": 100*absdiff(a["frac"],b["frac"]),
            "delta_velocidad_total_pal_s": absdiff(a["vel_total"],b["vel_total"]),
            "delta_velocidad_articulacion_pal_s": absdiff(a["vel_art"],b["vel_art"]),
        }
        asegurar(all(v is None or math.isfinite(v) and v>=0 for v in distancias.values()),f"Comparacion acustica invalida: {r['archivo_a']} {r['archivo_b']}")
        base={k:r[k] for k in BASIS}
        fila={**base, "enunciado":r["enunciado"],"toma_a":r["toma_a"],"toma_b":r["toma_b"],
              "archivo_a":r["archivo_a"],"archivo_b":r["archivo_b"],
              "alertas_a":a["alertas"],"alertas_b":b["alertas"],
              "f0_alertada":int(f0_flag),"dispersion_f0_alertada":int(disp_flag),"vad_alertada":int(vad_flag),
              "f0_rango_st_a":a["rango_st"],"f0_rango_st_b":b["rango_st"], **distancias}
        indiv.append(fila)
        acum_enunciado[(r["id_comparacion"],r["enunciado"])].append(fila)
        cont_grupo[r["grupo"]]+=1
    asegurar(cont_grupo==Counter(ESPERADO) and len(indiv)==857 and len(acum_enunciado)==326,"Conteos individuales incorrectos")
    enunc=[]; por_pareja=defaultdict(list)
    for d in defs.values():
        for e in d["ens"]:
            registros=acum_enunciado[(d["id"],e)]
            esperado=1 if d["grupo"]=="misma_persona" else 4
            asegurar(len(registros)==esperado, f"Numero de tomas incorrecto: {d['id']} {e}")
            base={k: registros[0][k] for k in BASIS}
            fila={**base, "enunciado":e, "n_puntuaciones":len(registros),
                  "n_f0_alertadas":sum(r["f0_alertada"] for r in registros),
                  "n_dispersion_f0_alertadas":sum(r["dispersion_f0_alertada"] for r in registros),
                  "n_vad_alertadas":sum(r["vad_alertada"] for r in registros),
                  **resumen_metricas(registros, "n_")}
            # Para la medida filtrada de variacion tonal necesitamos que las cuatro tomas
            # hayan superado los controles de calidad
            # No mezclamos promedios basados en cantidades distintas de tomas
            especial = "delta_rango_f0_st_sin_alertas"
            if fila[f"n_{especial}"] != esperado:
                for est in ESTADISTICAS:
                    fila[f"{est}_{especial}"] = None
            enunc.append(fila)
            por_pareja[d["id"]].append(fila)
    parejas=[]
    for d in defs.values():
        registros=por_pareja[d["id"]]
        base={k:registros[0][k] for k in BASIS}
        fila={**base, "n_enunciados":len(registros),
              "n_puntuaciones":sum(r["n_puntuaciones"] for r in registros),
              "n_enunciados_f0_alertados":sum(r["n_f0_alertadas"]>0 for r in registros),
              "n_enunciados_dispersion_f0_alertados":sum(r["n_dispersion_f0_alertadas"]>0 for r in registros),
              "n_enunciados_vad_alertados":sum(r["n_vad_alertadas"]>0 for r in registros)}
        # Primero promediamos las combinaciones de tomas correspondientes a cada frase
        # y despues las frases para que todas tengan el mismo peso
        for m in METRICAS:
            valores=[r[f"media_{m}"] for r in registros if r[f"media_{m}"] is not None]
            fila[f"n_enunciados_{m}"]=len(valores)
            for est,val in estadisticas(valores).items():
                fila[f"{est}_{m}"]=val
        parejas.append(fila)
    asegurar(len(enunc)==326 and len(parejas)==33,"Cantidad de resúmenes incorrecta")
    asegurar(Counter(r["grupo"] for r in enunc)==Counter(N_ENUNCIADOS),"Resumenes por enunciado: grupos incorrectos")
    asegurar(Counter(r["grupo"] for r in parejas)==Counter({"misma_persona":15,"hermanos":6,"control":12}), "Parejas por grupo incorrectas")
    asegurar(sum(r["n_puntuaciones"] for r in enunc)==sum(r["n_puntuaciones"] for r in parejas)==857,"Se perdieron tomas al agrupar")
    # El contraste entre hermanos y controles se calcula por enunciado
    # sin tratar cada grabacion como una familia diferente
    indices={(r["id_comparacion"],r["enunciado"]):r for r in enunc}
    por_ref=defaultdict(list)
    for d in defs.values():
        if d["grupo"] != "misma_persona":
            por_ref[d["ref"]].append(d)
    trios=[]
    for ref_f, miembros in sorted(por_ref.items()):
        h=next(d for d in miembros if d["grupo"]=="hermanos")
        cs=[d for d in miembros if d["grupo"]=="control"]
        asegurar(len(cs)==2 and all(d["ens"]==h["ens"] for d in cs),f"Trio incompleto {ref_f}")
        a,b=h["a"],h["b"]
        externo=(set([cs[0]["a"],cs[0]["b"],cs[1]["a"],cs[1]["b"]])-{a,b})
        asegurar(len(externo)==1,f"Control no unico {ref_f}")
        fila={"pareja_familiar_referencia":ref_f,"participante_hermano_a":a,
              "participante_hermano_b":b,"participante_control":next(iter(externo)),
              "n_enunciados_diseno":len(h["ens"])}
        for m in METRICAS:
            sh=[];co=[]
            # Usamos una frase solo cuando hay valores para hermanos y ambos controles
            for e in h["ens"]:
                s=indices[(h["id"],e)][f"media_{m}"]
                c1=indices[(cs[0]["id"],e)][f"media_{m}"]
                c2=indices[(cs[1]["id"],e)][f"media_{m}"]
                if s is not None and c1 is not None and c2 is not None:
                    sh.append(s);co.append((c1+c2)/2)
            fila[f"n_enunciados_comunes_{m}"]=len(sh)
            fila[f"media_hermanos_{m}"]=float(np.mean(sh)) if sh else None
            fila[f"media_controles_{m}"]=float(np.mean(co)) if co else None
            fila[f"contraste_hermanos_menos_controles_{m}"]=float(np.mean(np.asarray(sh)-np.asarray(co))) if sh else None
        trios.append(fila)
    asegurar(len(trios)==6,"Se esperaban 6 contrastes de trios")
    asegurar(all(r["n_enunciados_comunes_delta_f0_st"]==r["n_enunciados_diseno"] for r in trios),"F0 mediana incompleta en un trio")
    return indiv,enunc,parejas,trios


# Dejamos vacios los valores que no se pudieron calcular
def celda(valor):
    if valor is None:
        return ""
    if isinstance(valor,(float,np.floating)):
        return format(float(valor),".12g")
    return str(valor)


# Preparamos las columnas y los registros del CSV
def csv_bytes(registros,fields):
    buffer=io.StringIO(newline="")
    w=csv.DictWriter(buffer,fieldnames=fields,extrasaction="raise",lineterminator="\n")
    w.writeheader()
    for r in registros:
        w.writerow({k:celda(r[k]) for k in fields})
    return buffer.getvalue().encode("utf-8")


# Guardamos las tablas nuevas sin sobrescribir resultados anteriores
def escribir_atomicamente(resultados,config):
    rutas=[*SALIDAS.values()]
    existentes=[p.name for p in rutas if p.exists()]
    asegurar(not existentes, f"No se sobrescriben resultados existentes: {existentes}")
    contenido={
        "individual":csv_bytes(resultados[0],INDIVIDUAL),
        "enunciado":csv_bytes(resultados[1],ENUNCIADO),
        "pareja":csv_bytes(resultados[2],PAREJA),
        "trios":csv_bytes(resultados[3],TRIOS),
        "config":(json.dumps(config,ensure_ascii=False,indent=2)+"\n").encode("utf-8"),
    }
    temporales=[]; creados=[]
    try:
        for k,ruta in SALIDAS.items():
            tmp=ruta.with_name(ruta.name+".tmp")
            with tmp.open("xb") as f:
                f.write(contenido[k]);f.flush();os.fsync(f.fileno())
            temporales.append((tmp,ruta))
        for tmp,ruta in temporales:
            os.replace(tmp,ruta)
            creados.append(ruta)
    except BaseException:
        for tmp,ruta in temporales:
            tmp.unlink(missing_ok=True)
        for ruta in creados:
            ruta.unlink(missing_ok=True)
        raise


# Esta funcion valida las entradas, calcula distancias y decide si guardar las tablas
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo-verificar",action="store_true",help="Verificar y calcular sin escribir salidas")
    args=parser.parse_args()
    muestras,ecapa,defs,cfg=consultar_entradas()
    resultados=ejecutar(muestras,ecapa,defs)
    conteo_f0=sum(r["f0_alertada"] for r in resultados[0])
    conteo_disp=sum(r["dispersion_f0_alertada"] for r in resultados[0])
    conteo_vad=sum(r["vad_alertada"] for r in resultados[0])
    print("=== PASO 6: COMPARACIONES ACUSTICAS ===")
    print(f"Audio y protocolo: 298 grabaciones, {len(defs)} definiciones")
    print(f"Individuales: {len(resultados[0])} | Enunciados: {len(resultados[1])} | Parejas: {len(resultados[2])} | Trios: {len(resultados[3])}")
    print("Por grupo:", dict(Counter(r["grupo"] for r in resultados[0])))
    print(f"Comparaciones con alerta F0: {conteo_f0}; variabilidad F0 marcada: {conteo_disp}; VAD marcado: {conteo_vad}")
    if args.solo_verificar:
        print("VERIFICACION CORRECTA: no se crearon archivos del Paso 6.")
        return
    hashes={k:digest(v) for k,v in ENTRADAS.items()}
    config={"descripcion":"Paso 6 - diferencias acusticas simetricas dentro del mismo enunciado",
            "fecha_utc":datetime.now(timezone.utc).isoformat(), "python":sys.version.split()[0],"numpy":np.__version__,
            "sha256_script":digest(Path(__file__)), "sha256_entradas":hashes,
            "n_individuales":857,"n_enunciados":326,"n_parejas":33,"n_trios":6,
            "metricas":list(METRICAS),"umbral_dispersión_f0_iqr_st":UMBRAL_IQR_F0_ST,
            "umbral_dispersión_f0_desv_st":UMBRAL_DESV_F0_ST,
            "criterio_f0":"Mediana F0: abs(12*log2(F0a/F0b)); rangos: abs(12*log2(q3a/q1a)-12*log2(q3b/q1b))",
            "criterio_duracion_pct":"Diferencia simetrica = 200*abs(a-b)/(a+b). Datos ausentes -> celda vacia.",
            "criterio_calidad":"F0 mediana siempre conservada con alertas; variabilidad sin alertas solo cuando ninguno tiene f0_* y ambos tienen RIC F0 <=6 st y desviacion F0 <=6 st; para resumen filtrado por enunciado se requieren las cuatro combinaciones validas entre diferentes personas; sin filtrar resultados brutos",
            "criterio_contrastes_trios":"Para cada medida, tomar solo enunciados disponibles y validos simultaneamente para hermanos y ambos controles; promedio por enunciado, luego promedio entre enunciados; contraste = hermanos - media de controles; negativo indica menor diferencia acustica entre hermanos (no prueba parentesco)",
            "advertencias":"La tasa de palabras/s proviene del texto asignado; no es ASR. Velocidad y duracion del mismo enunciado no son evidencias independientes. La variabilidad F0 puede contener errores de octava. Tomas y controles reutilizados no son muestras independientes.",
            "n_comparaciones_advertidas":{"f0":conteo_f0,"dispersion_f0":conteo_disp,"vad":conteo_vad}}
    escribir_atomicamente(resultados,config)
    for k,n in (("individual",857),("enunciado",326),("pareja",33),("trios",6)):
        print(f"GUARDADO: {SALIDAS[k].name} ({n} filas)")
    print(f"GUARDADO: {SALIDAS['config'].name}")
    print("PASO 6: generado; pendiente auditoria independiente e interpretacion.")


if __name__ == "__main__":
    main()
