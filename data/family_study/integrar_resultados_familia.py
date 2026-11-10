#!/usr/bin/env python3
"""Reune las similitudes ECAPA y las distancias acusticas del estudio

Comprueba las tablas de entrada y calcula contrastes frente a controles
Usa el mismo conjunto de enunciados para cada grupo familiar
Un valor positivo indica mayor parecido entre hermanos en la medida analizada
Tambien calcula la sensibilidad al excluir una frase cada vez
No combina todas las medidas en una puntuacion unica ni estima parentesco
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
RES = BASE / 'results'
ENTRADAS = {
    'protocolo': 'protocolo_comparaciones.csv',
    'ecapa_individual': 'ecapa_comparaciones.csv',
    'ecapa_enunciado': 'ecapa_resumen_enunciados.csv',
    'ecapa_pareja': 'ecapa_resumen_parejas.csv',
    'acustica_individual': 'acustica_comparaciones.csv',
    'acustica_enunciado': 'acustica_resumen_enunciados.csv',
    'acustica_pareja': 'acustica_resumen_parejas.csv',
    'acustica_trios': 'acustica_contrastes_trios.csv',
    'acustica_config': 'acustica_comparaciones_config.json',
}
SALIDAS = {
    'enunciados': 'resumen_integrado_enunciados.csv',
    'parejas': 'resumen_integrado_parejas.csv',
    'frases_trios': 'contrastes_integrados_enunciados.csv',
    'trios': 'contrastes_integrados_trios.csv',
    'robustez': 'robustez_integrada.csv',
    'config': 'integracion_paso7_config.json',
}
FAMILIAS = [
    ('F01_P01', 'F01_P02', 'F02_P07'),
    ('F02_P04', 'F02_P06', 'F03_P08'),
    ('F03_P08', 'F03_P09', 'F02_P07'),
    ('F04_P10', 'F04_P11', 'F02_P03'),
    ('F05_P12', 'F05_P13', 'F02_P05'),
    ('F06_P14', 'F06_P15', 'F03_P09'),
]
N_GRUPO_INDIV = {'misma_persona': 149, 'hermanos': 236, 'control': 472}
N_GRUPO_FRASE = {'misma_persona': 149, 'hermanos': 59, 'control': 118}
N_GRUPO_PAREJA = {'misma_persona': 15, 'hermanos': 6, 'control': 12}
BASE_CAMPOS = ['id_comparacion', 'grupo', 'pareja_familiar_referencia',
               'participante_a', 'participante_b']
IDENT_INDIV = BASE_CAMPOS + ['enunciado', 'toma_a', 'toma_b', 'archivo_a', 'archivo_b']
METRICAS_ACUSTICAS = [
    'delta_f0_st', 'delta_rango_f0_st', 'delta_rango_f0_st_sin_alertas',
    'delta_duracion_total_s', 'delta_duracion_total_pct',
    'delta_habla_estimada_s', 'delta_habla_estimada_pct',
    'delta_fraccion_habla_pp', 'delta_velocidad_total_pal_s',
    'delta_velocidad_articulacion_pal_s',
]
# Las once medidas describen aspectos relacionados de la voz, no once pruebas independientes
METRICAS = ['similitud_coseno'] + METRICAS_ACUSTICAS
ALERTAS = ['f0_alertada', 'dispersion_f0_alertada', 'vad_alertada']
N_ALERTAS = ['n_f0_alertadas', 'n_dispersion_f0_alertadas', 'n_vad_alertadas']
N_ALERTAS_PAREJAS = ['n_enunciados_f0_alertados',
                     'n_enunciados_dispersion_f0_alertados',
                     'n_enunciados_vad_alertados']
# La tolerancia se utiliza para decidir si un contraste es positivo o negativo
EPS_SIGNO = 1e-9   # No se usa para cambiar las puntuaciones calculadas
TOL = 1e-8


# Si encontramos datos incoherentes detenemos el analisis
def exigir(cond, msg):
    if not cond:
        raise ValueError(msg)


# Calculamos el SHA-256 para saber si una tabla cambio desde su generacion
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


# Leemos la tabla y comprobamos que no tenga filas incompletas
def leer(nombre):
    p = RES / ENTRADAS[nombre]
    exigir(p.is_file(), f'Falta entrada privada: {p}')
    with p.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        exigir(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
               f'Encabezados invalidos: {p.name}')
        rows = list(reader)
    exigir(all(None not in r and all(v is not None for v in r.values()) for r in rows),
           f'Filas incompletas o columnas adicionales: {p.name}')
    return rows, set(reader.fieldnames)


# Revisamos las columnas antes de utilizar los datos
def cols(have, expected, name):
    faltan = set(expected) - have
    exigir(not faltan, f'{name}: faltan columnas: {sorted(faltan)}')


# Convertimos el valor y rechazamos numeros invalidos o fuera de rango
def numero(x, contexto, low=None, high=None):
    if x in ('', None):
        return None
    try:
        v = float(x)
    except (ValueError, TypeError):
        raise ValueError(f'{contexto}: no es numero {x!r}') from None
    exigir(math.isfinite(v) and (low is None or v >= low - TOL)
           and (high is None or v <= high + TOL),
           f'{contexto}: fuera de rango o no finito: {x!r}')
    return v


# Comprobamos que los conteos sean enteros y no negativos
def entero(x, contexto, minimo=0):
    try:
        v = int(x)
    except (TypeError, ValueError):
        raise ValueError(f'{contexto}: entero invalido: {x!r}') from None
    exigir(v >= minimo, f'{contexto}: entero menor que {minimo}')
    return v


# Recalculamos el valor para comprobar que coincida con la tabla
def igual(valor, ref, contexto):
    if ref is None:
        exigir(valor in ('', None), f'{contexto}: se esperaba ausencia de valor')
    else:
        v = numero(valor, contexto)
        exigir(v is not None and math.isclose(v, ref, rel_tol=TOL, abs_tol=TOL),
               f'{contexto}: obtenido={valor}, esperado={ref:.12g}')


# Buscamos cada fila por sus identificadores y detectamos duplicados
def indice(rows, llaves, nombre):
    d = {}
    for r in rows:
        k = tuple(r[x] for x in llaves)
        exigir(k not in d, f'{nombre}: clave duplicada: {k}')
        d[k] = r
    return d


# Un valor no disponible queda vacio y no se convierte en cero
def estadistica(vals):
    if not vals:
        return {'media': None, 'mediana': None, 'desv': None,
                'min': None, 'max': None}
    return {
        'media': statistics.fmean(vals), 'mediana': statistics.median(vals),
        'desv': statistics.stdev(vals) if len(vals)>1 else None,
        'min': min(vals), 'max': max(vals),
    }


# Calculamos el cuantil a partir de los valores ordenados
def quantil(vals, q):
    if len(vals) == 1:
        return None
    s = sorted(vals)
    idx = (len(s)-1) * q
    i = math.floor(idx)
    j = math.ceil(idx)
    return s[i] + (s[j] - s[i])*(idx-i)


# Verificamos que los resumenes de ECAPA coincidan con sus puntuaciones individuales
def comprobar_ecapa_stats(f, vals, contexto):
    st = estadistica(vals)
    q1, q3 = quantil(vals, .25), quantil(vals, .75)
    ref = {
        'media_coseno': st['media'], 'mediana_coseno': st['mediana'],
        'desviacion_estandar_coseno': st['desv'],
        'minimo_coseno': st['min'], 'maximo_coseno': st['max'],
        'q1_coseno': q1, 'q3_coseno': q3,
        'rango_intercuartil_coseno': q3-q1 if q1 is not None else None,
    }
    for k, v in ref.items():
        igual(f[k], v, f'{contexto}/{k}')


# Hacemos lo mismo con las distancias acusticas y sus promedios
def comprobar_acustica_stats(f, rows, contexto, nivel):
    for m in METRICAS_ACUSTICAS:
        if nivel == 'frase':
            vals = [numero(r[m], f'{contexto}/{m}') for r in rows]
            vals = [v for v in vals if v is not None]
            exigir(entero(f[f'n_{m}'], contexto) == len(vals),
                   f'{contexto}: n_{m} incorrecto')
            # La distancia tonal filtrada solo se calcula para un enunciado
            # si estan disponibles todas las combinaciones de tomas
            datos = ([] if m.endswith('_sin_alertas') and len(vals) != len(rows)
                     else vals)
        else:
            vals = [numero(r[f'media_{m}'], f'{contexto}/{m}') for r in rows]
            datos = [v for v in vals if v is not None]
            exigir(entero(f[f'n_enunciados_{m}'], contexto) == len(datos),
                   f'{contexto}: n_enunciados_{m} incorrecto')
        st = estadistica(datos)
        for tipo, resultado in st.items():
            igual(f[f'{tipo}_{m}'], resultado, f'{contexto}/{tipo}_{m}')


# Distinguimos resultados favorables, contrarios o practicamente empatados
def signo(v):
    if v is None:
        return None
    return 1 if v > EPS_SIGNO else (-1 if v < -EPS_SIGNO else 0)


# Consultamos el promedio de una medida dentro de una pareja
def valf(row, metrica):
    return numero(row[f'media_{metrica}'], f'{row["id_comparacion"]}/{metrica}')


# Cargamos los resultados previos y verificamos sus hashes y cantidades
def descargar_y_validar():
    for n in ENTRADAS.values():
        exigir((RES/n).is_file(), f'Falta archivo {n}')
    cfg = json.loads((RES/ENTRADAS['acustica_config']).read_text(encoding='utf-8'))
    hashes = cfg.get('sha256_entradas', {})
    for origen, dest in [('protocolo','protocolo'), ('ecapa','ecapa_individual'),
                         ('acustica','acustica_caracteristicas'),
                         ('config_p5','config_p5'), ('manifiesto','manifiesto')]:
        # Revisamos estas entradas con los hashes guardados en la configuracion
        # sin volver a procesar ninguna grabacion
        if origen == 'acustica':
            ruta = RES/'caracteristicas_audio.csv'
        elif origen == 'config_p5':
            ruta = RES/'caracteristicas_audio_config.json'
        elif origen == 'manifiesto':
            ruta = RES/'embeddings_manifest.csv'
        else:
            ruta = RES/ENTRADAS[dest]
        exigir(ruta.is_file() and hashes.get(origen) == sha(ruta),
               f'Procedencia Paso 6 modificada: {ruta.name}')
    p6_script = BASE/'comparar_acustica_familia.py'
    exigir(p6_script.is_file() and cfg.get('sha256_script') == sha(p6_script),
           'El programa Paso 6 no coincide con su configuracion registrada')

    tablas = {}
    for name in ENTRADAS:
        if name.endswith('_config'):
            continue
        tablas[name], h = leer(name)
        schema = BASE_CAMPOS
        if name == 'protocolo':
            schema += []
            cols(h, ['enunciados','n_enunciados','n_comparaciones_tomas'], name)
        elif name.endswith('_individual'):
            cols(h, IDENT_INDIV, name)
        elif name.endswith('_enunciado'):
            cols(h, BASE_CAMPOS + ['enunciado','n_puntuaciones'], name)
        elif name.endswith('_pareja'):
            cols(h, BASE_CAMPOS + ['n_enunciados','n_puntuaciones'], name)
        if name.startswith('ecapa_'):
            cols(h, ['similitud_coseno'] if name == 'ecapa_individual'
                 else ['media_coseno', 'mediana_coseno','desviacion_estandar_coseno',
                       'minimo_coseno', 'maximo_coseno','q1_coseno','q3_coseno',
                       'rango_intercuartil_coseno'], name)
        if name.startswith('acustica_'):
            if name == 'acustica_individual':
                cols(h, METRICAS_ACUSTICAS + ALERTAS, name)
            elif name == 'acustica_enunciado':
                cols(h, [f'n_{m}' for m in METRICAS_ACUSTICAS]
                     + [f'{s}_{m}' for m in METRICAS_ACUSTICAS
                        for s in ('media','mediana','desv','min','max')]
                     + N_ALERTAS, name)
            elif name == 'acustica_pareja':
                cols(h, [f'n_enunciados_{m}' for m in METRICAS_ACUSTICAS]
                     + [f'{s}_{m}' for m in METRICAS_ACUSTICAS
                        for s in ('media','mediana','desv','min','max')]
                     + N_ALERTAS_PAREJAS, name)
        if name == 'acustica_trios':
            cols(h, ['pareja_familiar_referencia','participante_hermano_a',
                     'participante_hermano_b','participante_control',
                     'n_enunciados_diseno'] + [f'{p}_{m}' for m in METRICAS_ACUSTICAS
                     for p in ('n_enunciados_comunes','media_hermanos','media_controles',
                               'contraste_hermanos_menos_controles')], name)
    counts = {'protocolo':33, 'ecapa_individual':857, 'ecapa_enunciado':326,
              'ecapa_pareja':33, 'acustica_individual':857, 'acustica_enunciado':326,
              'acustica_pareja':33, 'acustica_trios':6}
    for name, n in counts.items():
        exigir(len(tablas[name]) == n, f'{name}: se esperaban {n} filas')
    for name, expected in [
        ('ecapa_individual', N_GRUPO_INDIV), ('acustica_individual', N_GRUPO_INDIV),
        ('ecapa_enunciado', N_GRUPO_FRASE), ('acustica_enunciado', N_GRUPO_FRASE),
        ('ecapa_pareja', N_GRUPO_PAREJA), ('acustica_pareja', N_GRUPO_PAREJA),
    ]:
        exigir(Counter(r['grupo'] for r in tablas[name]) == Counter(expected),
               f'{name}: distribucion por grupo no coincide')
    return tablas, cfg


# Combinamos ECAPA y las distancias acusticas para comparar hermanos y controles
# Despues repetimos el contraste quitando un enunciado cada vez
def calcular(tablas):
    protocolo = {k[0]: v for k, v in indice(tablas['protocolo'], ['id_comparacion'], 'protocolo').items()}
    ec_ind = indice(tablas['ecapa_individual'], IDENT_INDIV, 'ECAPA indiv')
    ac_ind = indice(tablas['acustica_individual'], IDENT_INDIV, 'acustica indiv')
    exigir(set(ec_ind) == set(ac_ind), 'ECAPA y acustica tienen distinta clave individual')
    ec_en = indice(tablas['ecapa_enunciado'], BASE_CAMPOS+['enunciado'], 'ECAPA frase')
    ac_en = indice(tablas['acustica_enunciado'], BASE_CAMPOS+['enunciado'], 'acustica frase')
    exigir(set(ec_en) == set(ac_en), 'ECAPA y acustica tienen distintas claves de enunciados')
    ec_pa = indice(tablas['ecapa_pareja'], BASE_CAMPOS, 'ECAPA parejas')
    ac_pa = indice(tablas['acustica_pareja'], BASE_CAMPOS, 'acustica parejas')
    exigir(set(ec_pa) == set(ac_pa), 'ECAPA y acustica tienen distintas claves de parejas')
    exigir(set(k[0] for k in ec_pa) == set(protocolo),
           'Protocolo y resumenes difieren en IDs')

    # Ademas de coincidir entre si, las tablas deben corresponder exactamente
    # a las personas, enunciados y tomas definidos para el estudio
    por_fra = defaultdict(list)
    por_par = defaultdict(list)
    for key, e in ec_ind.items():
        a = ac_ind[key]
        exige_id = e['id_comparacion']
        exigir(exige_id in protocolo, f'ID no reconocido: {exige_id}')
        d = protocolo[exige_id]
        exigir(all(e[z] == d[z] for z in BASE_CAMPOS),
               f'ECAPA no corresponde a definicion {exige_id}')
        enuns = d['enunciados'].split(';')
        exigir(e['enunciado'] in enuns, f'Enunciado fuera del protocolo {key}')
        ta, tb = e['toma_a'], e['toma_b']
        esperadas = [('T01','T02')] if e['grupo']=='misma_persona' else [
            ('T01','T01'),('T01','T02'),('T02','T01'),('T02','T02')]
        exigir((ta,tb) in esperadas, f'Toma fuera de protocolo: {key}')
        exigir(e['archivo_a'] == f'{e["participante_a"]}_{e["enunciado"]}_{ta}.wav'
               and e['archivo_b'] == f'{e["participante_b"]}_{e["enunciado"]}_{tb}.wav',
               f'Archivo incongruente: {key}')
        x = numero(e['similitud_coseno'], f'ECAPA {key}', low=-1,high=1)
        exigir(x is not None, f'Coseno ausente {key}')
        for m in METRICAS_ACUSTICAS:
            numero(a[m], f'{key}/{m}', low=0)
        for flag in ALERTAS:
            exigir(a[flag] in ('0','1'), f'Flag de calidad no binario: {key}/{flag}')
        por_fra[tuple(e[z] for z in BASE_CAMPOS+['enunciado'])].append((e,a))

    # Cada comparacion debe tener todas las combinaciones esperadas
    # No aceptamos una seleccion incompleta de tomas o frases
    por_id_ind = defaultdict(list)
    for k in ec_ind:
        por_id_ind[k[0]].append(k)
    for id_, d in protocolo.items():
        ens = d['enunciados'].split(';')
        exigir(ens and len(set(ens))==len(ens) and ens==sorted(ens),
               f'Protocolo tiene frases duplicadas o desordenadas: {id_}')
        if d['grupo'] == 'misma_persona':
            expected = {(e,'T01','T02') for e in ens}
        else:
            expected = {(e,ta,tb) for e in ens for ta in ('T01','T02')
                        for tb in ('T01','T02')}
        found = {(k[5],k[6],k[7]) for k in por_id_ind[id_]}
        exigir(expected==found and len(found)==len(por_id_ind[id_]),
               f'Comparaciones de tomas incompletas o duplicadas: {id_}')
        exigir(entero(d['n_enunciados'], id_)==len(ens) and
               entero(d['n_comparaciones_tomas'], id_)==len(expected),
               f'Conteos de protocolo incorrectos: {id_}')

    # Comprobamos los promedios por enunciado a partir de las tomas originales
    # y despues los promedios por pareja antes de continuar
    frases = []
    ind_frases = {}
    for k, e in ec_en.items():
        grupo = e['grupo']
        obs = por_fra.get(k, [])
        esperado = 1 if grupo == 'misma_persona' else 4
        exigir(len(obs)==esperado, f'Faltan tomas en {k}')
        vals = [numero(row['similitud_coseno'], str(k), -1, 1) for row,_ in obs]
        comprobar_ecapa_stats(e, vals, str(k))
        a = ac_en[k]
        exigir(entero(e['n_puntuaciones'], str(k))==esperado and
               entero(a['n_puntuaciones'], str(k))==esperado,
               f'n_puntuaciones incorrecto: {k}')
        comprobar_acustica_stats(a, [r for _,r in obs], str(k), 'frase')
        for i,flag in enumerate(ALERTAS):
            exigir(entero(a[N_ALERTAS[i]], str(k)) == sum(int(x[flag]) for _,x in obs),
                   f'Alertas agregadas incorrectas {k}: {flag}')
        f = dict(zip(BASE_CAMPOS+['enunciado'], k))
        f.update(n_puntuaciones=esperado, similitud_coseno=numero(e['media_coseno'], str(k)))
        for m in METRICAS_ACUSTICAS:
            f[m] = numero(a[f'media_{m}'], f'{k}/{m}')
            f[f'n_{m}'] = entero(a[f'n_{m}'], str(k))
        for alert in N_ALERTAS:
            f[alert] = entero(a[alert], str(k))
        frases.append(f)
        ind_frases[k] = f
        por_par[k[:5]].append((e,a,f))
    exigir(len(frases)==326 and len(por_par)==33, 'Recuperacion de frases incorrecta')

    parejas = []
    ind_parejas = {}
    for k, e in ec_pa.items():
        obs = por_par.get(k, [])
        exigir(entero(e['n_enunciados'],str(k))==len(obs) and
               entero(e['n_puntuaciones'],str(k))==sum(int(x[0]['n_puntuaciones']) for x in obs),
               f'Conteos por pareja ECAPA incorrectos {k}')
        vals = [x[2]['similitud_coseno'] for x in obs]
        comprobar_ecapa_stats(e, vals, str(k))
        a=ac_pa[k]
        exigir(entero(a['n_enunciados'],str(k))==len(obs) and
               entero(a['n_puntuaciones'],str(k))==sum(int(x[1]['n_puntuaciones']) for x in obs),
               f'Conteos por pareja acusticos incorrectos {k}')
        comprobar_acustica_stats(a, [row for _,row,_ in obs], str(k), 'pareja')
        for i,name in enumerate(N_ALERTAS):
            exigir(entero(a[N_ALERTAS_PAREJAS[i]],str(k)) == sum(
                entero(x[1][name],str(k))>0 for x in obs), f'Alertas por pareja {k}')
        p = dict(zip(BASE_CAMPOS,k))
        p.update(n_enunciados=len(obs), n_puntuaciones=entero(e['n_puntuaciones'],str(k)),
                 similitud_coseno=numero(e['media_coseno'],str(k)))
        for m in METRICAS_ACUSTICAS:
            p[m] = numero(a[f'media_{m}'],f'{k}/{m}')
            p[f'n_enunciados_{m}'] = entero(a[f'n_enunciados_{m}'],str(k))
        for alert in N_ALERTAS_PAREJAS:
            p[alert] = entero(a[alert],str(k))
        parejas.append(p); ind_parejas[k] = p

    # Comprobamos que cada pareja mantenga su control asignado originalmente
    refs_def = defaultdict(list)
    for d in protocolo.values():
        if d['grupo'] != 'misma_persona':
            refs_def[d['pareja_familiar_referencia']].append(d)
    esperado_refs = {f'{a}__{b}': (a,b,c) for a,b,c in FAMILIAS}
    exigir(set(refs_def)==set(esperado_refs), 'Referencias familiares diferentes al protocolo prefijado')
    indices_p = {r['id_comparacion']:r for r in parejas}
    indices_f = {(r['id_comparacion'],r['enunciado']):r for r in frases}
    indices_auditoria_trios = indice(tablas['acustica_trios'],
                                    ['pareja_familiar_referencia'], 'paso6 trios')

    trios = []
    contrastes_frases = []
    robustez = []
    # Las dos comparaciones de control comparten una sola persona externa
    for ref, (h1,h2,c) in esperado_refs.items():
        ds = refs_def[ref]
        hs = [d for d in ds if d['grupo']=='hermanos']
        cs = [d for d in ds if d['grupo']=='control']
        exigir(len(hs)==1 and len(cs)==2, f'{ref}: trios incompletos')
        h = hs[0]
        exigir((h['participante_a'],h['participante_b'])==(h1,h2),
               f'{ref}: hermanos cambiados de orientacion')
        ca = next((d for d in cs if {d['participante_a'],d['participante_b']}=={h1,c}),None)
        cb = next((d for d in cs if {d['participante_a'],d['participante_b']}=={h2,c}),None)
        exigir(ca is not None and cb is not None and ca is not cb,
               f'{ref}: controles no coinciden con protocolo')
        en_h = h['enunciados'].split(';')
        for d in (ca,cb):
            exigir(d['enunciados'].split(';')==en_h, f'{ref}: frases no pareadas')
        en_esperados = [f'E{i:02d}' for i in range(1,11)
                       if (ref != 'F04_P10__F04_P11' or i!=8)]
        exigir(en_h==en_esperados, f'{ref}: cobertura no esperada (E08)')
        salida = {'pareja_familiar_referencia':ref, 'participante_hermano_a':h1,
                  'participante_hermano_b':h2, 'participante_control':c,
                  'n_enunciados_diseno':len(en_h)}
        f_by_metric = defaultdict(list)
        # Comparamos hermanos y controles frase por frase antes de promediar la familia
        for en in en_h:
            rh = indices_f[(h['id_comparacion'], en)]
            ra = indices_f[(ca['id_comparacion'], en)]
            rb = indices_f[(cb['id_comparacion'], en)]
            cf = {'pareja_familiar_referencia':ref,'enunciado':en,
                  'id_hermanos':h['id_comparacion'],
                  'id_control_a':ca['id_comparacion'], 'id_control_b':cb['id_comparacion']}
            for alerta in N_ALERTAS:
                suma = sum(r[alerta] for r in (rh,ra,rb))
                cf[f'total_{alerta}'] = suma
                cf[f'alguna_{alerta}'] = int(suma > 0)
            for m in METRICAS:
                hval, aval, bval = (r[m] for r in (rh,ra,rb))
                if hval is None or aval is None or bval is None:
                    cf[f'hermanos_{m}'] = hval
                    cf[f'control_a_{m}'] = aval
                    cf[f'control_b_{m}'] = bval
                    cf[f'control_promedio_{m}'] = ((aval+bval)/2 if aval is not None and bval is not None else None)
                    cf[f'contraste_favorable_{m}'] = None
                    cf[f'gana_ambos_{m}'] = None
                    continue
                controles = (aval+bval)/2
                # Dejamos positivo el contraste cuando los hermanos resultan mas parecidos
                favorable = ((hval-controles) if m=='similitud_coseno'
                             else (controles-hval))
                gana_ambos = (hval > aval+EPS_SIGNO and hval > bval+EPS_SIGNO)
                if m!='similitud_coseno':
                    gana_ambos = (hval < aval-EPS_SIGNO and hval < bval-EPS_SIGNO)
                cf[f'hermanos_{m}'] = hval
                cf[f'control_a_{m}'] = aval
                cf[f'control_b_{m}'] = bval
                cf[f'control_promedio_{m}'] = controles
                cf[f'contraste_favorable_{m}'] = favorable
                cf[f'gana_ambos_{m}'] = int(gana_ambos)
                f_by_metric[m].append((en,hval,aval,bval,favorable,int(gana_ambos)))
            contrastes_frases.append(cf)
        for alerta in N_ALERTAS:
            salida[f'n_enunciados_alguna_{alerta}'] = sum(
                r[f'alguna_{alerta}'] for r in contrastes_frases if r['pareja_familiar_referencia']==ref)
        for m in METRICAS:
            validos = f_by_metric[m]
            n=len(validos)
            hs_vals=[r[1] for r in validos]
            ca_vals=[r[2] for r in validos]
            cb_vals=[r[3] for r in validos]
            deltas=[r[4] for r in validos]
            controles=[(a+b)/2 for a,b in zip(ca_vals,cb_vals)]
            favor=statistics.fmean(deltas) if n else None
            raw=(favor if m=='similitud_coseno' else -favor) if n else None
            pref=m
            salida[f'n_enunciados_comunes_{pref}']=n
            salida[f'media_hermanos_{pref}']=statistics.fmean(hs_vals) if n else None
            salida[f'media_control_a_{pref}']=statistics.fmean(ca_vals) if n else None
            salida[f'media_control_b_{pref}']=statistics.fmean(cb_vals) if n else None
            salida[f'media_controles_{pref}']=statistics.fmean(controles) if n else None
            salida[f'contraste_crudo_hermanos_menos_controles_{pref}']=raw
            salida[f'contraste_favorable_{pref}']=favor
            if n:
                hh, aa, bb = map(statistics.fmean,(hs_vals,ca_vals,cb_vals))
                orden = (1 if m=='similitud_coseno' else -1)
                salida[f'ganan_ambos_global_{pref}']=int(
                    orden*(hh-aa)>EPS_SIGNO and orden*(hh-bb)>EPS_SIGNO)
                salida[f'margen_minimo_a_controles_{pref}']=min(
                    orden*(hh-aa), orden*(hh-bb))
            else:
                salida[f'ganan_ambos_global_{pref}']=None
                salida[f'margen_minimo_a_controles_{pref}']=None
            salida[f'n_enunciados_favorables_{pref}']=sum(signo(x)>0 for x in deltas)
            salida[f'n_enunciados_contrarios_{pref}']=sum(signo(x)<0 for x in deltas)
            salida[f'n_enunciados_empate_{pref}']=sum(signo(x)==0 for x in deltas)
            salida[f'n_enunciados_gana_ambos_{pref}']=sum(r[5] for r in validos)
            # El analisis leave-one-out muestra cuanto cambia el contraste
            # cuando quitamos una frase, sin estimar incertidumbre de la poblacion
            # Por eso estos rangos no son intervalos de confianza ni pruebas de significancia
            loo = [statistics.fmean(deltas[:i]+deltas[i+1:])
                   for i in range(n)] if n>1 else []
            signo_principal=signo(favor)
            estables = all(signo(x)==signo_principal for x in loo) if loo else None
            robustez.append({
                'pareja_familiar_referencia':ref, 'metrica':m,
                'n_enunciados_comunes':n, 'contraste_favorable':favor,
                'n_enunciados_favorables':salida[f'n_enunciados_favorables_{pref}'],
                'n_enunciados_contrarios':salida[f'n_enunciados_contrarios_{pref}'],
                'n_enunciados_empate':salida[f'n_enunciados_empate_{pref}'],
                'n_enunciados_gana_ambos':salida[f'n_enunciados_gana_ambos_{pref}'],
                'n_loo':len(loo),
                'loo_min':min(loo) if loo else None,
                'loo_max':max(loo) if loo else None,
                'loo_signo_estable':int(estables) if estables is not None else None,
                'n_loo_favorables':sum(signo(x)>0 for x in loo),
                'n_loo_contrarios':sum(signo(x)<0 for x in loo),
                'n_loo_empate':sum(signo(x)==0 for x in loo),
            })
            if m!='similitud_coseno':
                paso6 = indices_auditoria_trios[(ref,)]
                exigir(entero(paso6[f'n_enunciados_comunes_{m}'],ref)==n,
                       f'{ref}/{m}: cobertura Paso6 distinta')
                igual(paso6[f'media_hermanos_{m}'],salida[f'media_hermanos_{m}'],f'{ref}/{m}/h')
                igual(paso6[f'media_controles_{m}'],salida[f'media_controles_{m}'],f'{ref}/{m}/c')
                igual(paso6[f'contraste_hermanos_menos_controles_{m}'],raw,f'{ref}/{m}/raw')
        trios.append(salida)

    exigir(len(contrastes_frases)==59 and len(trios)==6 and len(robustez)==66,
           'Cantidad de contrastes integrados incorrecta')
    for fila in trios:
        exigir(fila['n_enunciados_comunes_similitud_coseno']==fila['n_enunciados_diseno'],
               'Coseno sin frases completas')
        exigir(fila['n_enunciados_comunes_delta_f0_st']==fila['n_enunciados_diseno'],
               'F0 mediana sin frases completas')
    return frases, parejas, contrastes_frases, trios, robustez


# Organizamos los nombres y el orden de las columnas finales
def campos_salida():
    frases = BASE_CAMPOS+['enunciado','n_puntuaciones','similitud_coseno']
    parejas = BASE_CAMPOS+['n_enunciados','n_puntuaciones','similitud_coseno']
    for m in METRICAS_ACUSTICAS:
        frases += [m,f'n_{m}']
        parejas += [m,f'n_enunciados_{m}']
    frases += N_ALERTAS
    parejas += N_ALERTAS_PAREJAS
    ident=['pareja_familiar_referencia','enunciado','id_hermanos','id_control_a','id_control_b']
    pref=['pareja_familiar_referencia','participante_hermano_a','participante_hermano_b',
          'participante_control','n_enunciados_diseno']
    ident += [f'{prefix}_{a}' for a in N_ALERTAS for prefix in ('total','alguna')]
    pref += [f'n_enunciados_alguna_{a}' for a in N_ALERTAS]
    for m in METRICAS:
        ident += [f'{tipo}_{m}' for tipo in ('hermanos','control_a','control_b',
                  'control_promedio','contraste_favorable','gana_ambos')]
        pref += [f'{tipo}_{m}' for tipo in ('n_enunciados_comunes','media_hermanos',
                 'media_control_a','media_control_b','media_controles',
                 'contraste_crudo_hermanos_menos_controles','contraste_favorable',
                 'n_enunciados_favorables','n_enunciados_contrarios',
                 'n_enunciados_empate','n_enunciados_gana_ambos',
                 'ganan_ambos_global','margen_minimo_a_controles')]
    loo=['pareja_familiar_referencia','metrica','n_enunciados_comunes',
         'contraste_favorable','n_enunciados_favorables','n_enunciados_contrarios',
         'n_enunciados_empate','n_enunciados_gana_ambos','n_loo','loo_min','loo_max',
         'loo_signo_estable','n_loo_favorables','n_loo_contrarios','n_loo_empate']
    return [frases, parejas, ident, pref, loo]


# Preparamos las filas con un formato numerico consistente
def bytes_csv(rows, columns):
    buff = io.StringIO(newline='')
    writer=csv.DictWriter(buff,fieldnames=columns,extrasaction='raise',lineterminator='\n')
    writer.writeheader()
    for r in rows:
        data={}
        for key in columns:
            exigir(key in r, f'Falta campo de salida: {key}')
            val=r[key]
            data[key] = '' if val is None else format(val,'.12g') if isinstance(val,float) else val
        writer.writerow(data)
    return buff.getvalue().encode('utf-8')


# Escribimos las tablas en temporales y no reemplazamos archivos existentes
def guardar(resultados, cfg):
    existentes=[v for v in SALIDAS.values() if (RES/v).exists()]
    exigir(not existentes, f'Se prohibe sobrescribir: {existentes}')
    columns=campos_salida()
    contenido={
        name: bytes_csv(rows, campos)
        for name,rows,campos in zip(('enunciados','parejas','frases_trios','trios','robustez'),
                                    resultados,columns)
    }
    contenido['config']=(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    temporales=[]; instalados=[]
    try:
        for name,fname in SALIDAS.items():
            ruta=RES/fname
            tmp=RES/(fname+'.tmp')
            with tmp.open('xb') as f:
                f.write(contenido[name]);f.flush();os.fsync(f.fileno())
            temporales.append((tmp,ruta))
        for tmp,ruta in temporales:
            # El enlace duro falla si ya existe el destino y asi evita sobrescribirlo
            os.link(tmp,ruta)
            instalados.append(ruta)
        for tmp,_ in temporales:
            tmp.unlink()
    except BaseException:
        for tmp,_ in temporales:
            tmp.unlink(missing_ok=True)
        for ruta in instalados:
            ruta.unlink(missing_ok=True)
        raise


# Esta funcion revisa entradas, calcula y guarda solo si se solicita una ejecucion normal
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--solo-verificar',action='store_true',help='Calcular sin escribir archivos')
    args=ap.parse_args()
    tablas,cfg6=descargar_y_validar()
    resultados=calcular(tablas)
    print('=== PASO 7: INTEGRACION ECAPA Y ACUSTICA ===')
    print('Entrada: 857 comparaciones de cada modalidad; 326 frases; 33 parejas')
    print('Salida: %d frases, %d parejas, %d contrastes frase-trio, %d trios, %d filas de robustez' %
          tuple(len(x) for x in resultados))
    print('Convencion: contraste_favorable > 0 indica mayor similitud entre hermanos que promedio de controles')
    print('Cobertura tonal filtrada por trio:',
          [r['n_enunciados_comunes_delta_rango_f0_st_sin_alertas'] for r in resultados[3]])
    if args.solo_verificar:
        print('VERIFICACION CORRECTA: no se escribieron archivos del Paso 7.')
        return
    config={
        'paso':7, 'descripcion':'Integracion descriptiva, no inferencial, sobre trios prefijados',
        'fecha_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version.split()[0], 'sha256_script':sha(Path(__file__)),
        'sha256_entradas':{k:sha(RES/v) for k,v in ENTRADAS.items()},
        'sha256_procedencia_paso6':sha(RES/ENTRADAS['acustica_config']),
        'cantidad_entradas':{k:len(v) for k,v in tablas.items()},
        'cantidad_salidas':dict(zip(('enunciados','parejas','frases_trios','trios','robustez'),
                                    [len(v) for v in resultados])),
        'familias_prefijadas':FAMILIAS,'metricas':METRICAS,
        'convencion':'ECAPA favorable=H-media(CA,CB); acustica favorable=media(CA,CB)-H',
        'contraste_crudo':'En ECAPA es favorable; en acustica su opuesto.',
        'agregacion':'Una media por frase a partir de tomas; para trio promediar diferencias frase a frase sobre las MISMAS frases validas para las tres parejas.',
        'robustez':'Por metrica y trio, leave-one-enunciado-out descriptivo; sin significancia estadistica.',
        'regla_empates_epsilon':EPS_SIGNO,
        'advertencias':[
            'El modelo ECAPA mide similitud de representaciones para verificacion de hablante, no parentesco ni similitud humana.',
            'Seis trios con controles reutilizados; no son observaciones familiares plenamente independientes.',
            'Tomas de voz y frases no representan nuevos participantes.',
            'Variabilidad F0 filtrada secundaria, con cobertura desigual; umbrales de 6 semitonos no universales.',
            'Velocidad de lectura total y duracion total contienen informacion matematica redundante.',
            'No se generaron p-valores, umbrales de identificacion ni clasificador de parentesco.',
        ],
    }
    guardar(resultados,config)
    for name,rows in zip(('enunciados','parejas','frases_trios','trios','robustez'),resultados):
        print(f'GUARDADO: {SALIDAS[name]} ({len(rows)} filas)')
    print(f'GUARDADO: {SALIDAS["config"]}')
    print('PASO 7 GENERADO: pendiente auditoria independiente con datos reales e interpretacion.')


if __name__=='__main__':
    main()
