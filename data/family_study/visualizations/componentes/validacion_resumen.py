# Validacion de tablas utilizadas en las figuras
#
# Se comprueban los identificadores, los conteos y los resultados
# de las comparaciones entre hermanos y controles
# Los datos se leen sin modificarlos ni completar valores faltantes
# Una tabla incompleta o inconsistente detiene la generacion de figuras
"""Comprueba la coherencia de los CSV sin modificar los resultados originales"""
from __future__ import annotations
import csv
import hashlib
import math
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = (
    'similitud_coseno', 'delta_f0_st', 'delta_rango_f0_st',
    'delta_rango_f0_st_sin_alertas', 'delta_duracion_total_s',
    'delta_duracion_total_pct', 'delta_habla_estimada_s',
    'delta_habla_estimada_pct', 'delta_fraccion_habla_pp',
    'delta_velocidad_total_pal_s', 'delta_velocidad_articulacion_pal_s',
)
REFERENCES = (
    'F01_P01__F01_P02', 'F02_P04__F02_P06', 'F03_P08__F03_P09',
    'F04_P10__F04_P11', 'F05_P12__F05_P13', 'F06_P14__F06_P15',
)
SOURCE_FILES = (
    'contrastes_integrados_trios.csv', 'contrastes_integrados_enunciados.csv',
    'robustez_integrada.csv', 'integracion_paso7_config.json',
)
EPS = 1e-9
TOL = 2e-8  # Se usa una tolerancia compatible con el redondeo de los CSV


# Se detiene la validacion cuando una condicion no se cumple
def insist(condition, message):
    if not condition:
        raise ValueError(message)


# Se obtiene la huella SHA-256 de los archivos de entrada
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


# Se lee un numero y se comprueba que sea valido
def num(row, key, context, allow_empty=False):
    insist(key in row, f'{context}: columna ausente {key}')
    value = row[key]
    if allow_empty and value == '':
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{context}: valor numérico inválido en {key}: {value!r}') from None
    insist(math.isfinite(n), f'{context}: valor no finito en {key}')
    return n


# Se comprueba que un conteo sea un numero entero
def integer(row, key, context):
    val = num(row, key, context)
    insist(val.is_integer(), f'{context}: se esperaba entero en {key}')
    return int(val)


# Se comparan dos valores teniendo en cuenta el redondeo
def same(x, y, ctx):
    insist(math.isclose(x, y, rel_tol=TOL, abs_tol=TOL),
           f'{ctx}: valor={x:.12g}, cálculo={y:.12g}')


# Se identifica si un contraste es positivo, negativo o cercano a cero
def sign(x):
    return (1 if x > EPS else (-1 if x < -EPS else 0))


# Se leen las tablas CSV y se revisan sus columnas y cantidad de filas
def csvfile(path, mandatory, expected):
    insist(path.is_file(), f'Falta archivo: {path}')
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        rd = csv.DictReader(stream)
        insist(rd.fieldnames and len(set(rd.fieldnames)) == len(rd.fieldnames),
               f'Cabeceras duplicadas o vacías: {path.name}')
        insist(set(mandatory).issubset(rd.fieldnames),
               f'{path.name}: faltan {sorted(set(mandatory)-set(rd.fieldnames))}')
        rows = list(rd)
    insist(len(rows) == expected, f'{path.name}: {len(rows)} filas; se esperan {expected}')
    insist(all(None not in r and all(v is not None for v in r.values()) for r in rows),
           f'{path.name}: filas corruptas, más columnas o faltantes')
    return rows


# Se organizan las filas por identificador sin permitir duplicados
def checked_table(rows, key_names, label):
    mapping = {}
    for r in rows:
        key = tuple(r[k] for k in key_names)
        insist(key not in mapping, f'{label}: clave repetida {key}')
        mapping[key] = r
    return mapping


# Se verifica la coherencia entre las familias, las frases y los contrastes
def load_and_check(base):
    """Comprueba conteos, identificadores y contrastes antes de dibujar las figuras"""
    res = Path(base).resolve() / 'results'
    insist(res.is_dir(), f'Falta directorio privado results/: {res}')
    import json
    config_path = res / SOURCE_FILES[3]
    insist(config_path.is_file(), f'Falta configuración: {config_path}')
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    insist(cfg.get('paso') == 7, 'La configuración no corresponde al Paso 7')
    trios = csvfile(res / SOURCE_FILES[0], ['pareja_familiar_referencia', 'n_enunciados_diseno'], 6)
    frases = csvfile(res / SOURCE_FILES[1], ['pareja_familiar_referencia', 'enunciado'], 59)
    robust = csvfile(res / SOURCE_FILES[2], ['pareja_familiar_referencia', 'metrica'], 66)
    tri_index = checked_table(trios, ['pareja_familiar_referencia'], 'Tríos')
    phrase_index = checked_table(frases, ['pareja_familiar_referencia','enunciado'], 'Enunciados')
    robust_index = checked_table(robust, ['pareja_familiar_referencia','metrica'], 'Robustez')
    insist(set(x[0] for x in tri_index) == set(REFERENCES), 'Tríos distintos del protocolo congelado')
    insist(set(robust_index) == {(ref,m) for ref in REFERENCES for m in METRICS},
           'Cobertura incompleta o inesperada en robustez')
    expect_phrases = {(ref, f'E{i:02d}') for ref in REFERENCES for i in range(1,11)
                      if not (ref == REFERENCES[3] and i == 8)}
    insist(set(phrase_index) == expect_phrases, 'Enunciados distintos del protocolo (E08 de F04)')
    coverage = []
    for ref in REFERENCES:
        t = tri_index[(ref,)]
        n_design = integer(t,'n_enunciados_diseno',ref)
        insist(n_design == (9 if ref == REFERENCES[3] else 10), f'{ref}: diseño inesperado')
        for m in METRICS:
            context = f'{ref}/{m}'
            ro = robust_index[(ref,m)]
            fs = [phrase_index[(ref,f'E{i:02d}')] for i in range(1,11)
                  if not (ref==REFERENCES[3] and i==8)]
            complete = []
            for f in fs:
                context_f = f'{context}/{f["enunciado"]}'
                h = num(f,'hermanos_'+m,context_f,True)
                ca = num(f,'control_a_'+m,context_f,True)
                cb = num(f,'control_b_'+m,context_f,True)
                delta = num(f,'contraste_favorable_'+m,context_f,True)
                win = f['gana_ambos_'+m]
                if delta is None:
                    insist(win == '', f'{context_f}: gana_ambos sin contraste')
                    continue
                insist(None not in (h,ca,cb),f'{context_f}: contraste con valores incompletos')
                insist(-1 <= h <= 1 and -1 <= ca <= 1 and -1 <= cb <= 1
                       if m == 'similitud_coseno' else min(h,ca,cb) >= 0,
                       f'{context_f}: valores fuera de rango')
                expected = h-(ca+cb)/2 if m=='similitud_coseno' else (ca+cb)/2-h
                same(delta,expected,context_f+'/signo')
                winning = (h > max(ca,cb)+EPS) if m=='similitud_coseno' else (h < min(ca,cb)-EPS)
                insist(integer(f,'gana_ambos_'+m,context_f)==int(winning),
                       f'{context_f}: gana_ambos incorrecto')
                complete.append((h,ca,cb,delta,int(winning)))
            n = len(complete)
            insist(n == integer(t,'n_enunciados_comunes_'+m,context)
                   == integer(ro,'n_enunciados_comunes',context),f'{context}: n no coincide')
            insist(n >= 2, f'{context}: cobertura insuficiente para LOO')
            means = [statistics.fmean(x[k] for x in complete) for k in range(4)]
            for k,field in enumerate(('media_hermanos_', 'media_control_a_',
                                       'media_control_b_', 'contraste_favorable_')):
                same(num(t,field+m,context),means[k],context+'/'+field)
            same(num(t,'media_controles_'+m,context),(means[1]+means[2])/2,context+'/media_controles')
            same(num(ro,'contraste_favorable',context),means[3],context+'/robustez')
            fav, contra, emp = (sum(sign(x[3])==s for x in complete) for s in (1,-1,0))
            wins = sum(x[4] for x in complete)
            counts = [('n_enunciados_favorables_', fav), ('n_enunciados_contrarios_', contra),
                      ('n_enunciados_empate_', emp), ('n_enunciados_gana_ambos_', wins)]
            for field,expected in counts:
                insist(integer(t,field+m,context) == expected, f'{context}: {field} no coincide')
            for field, expected in [('n_enunciados_favorables',fav),('n_enunciados_contrarios',contra),
                                    ('n_enunciados_empate',emp),('n_enunciados_gana_ambos',wins)]:
                insist(integer(ro,field,context) == expected, f'{context}: {field} no coincide')
            order = 1 if m == 'similitud_coseno' else -1
            both = int(order*(means[0]-means[1])>EPS and order*(means[0]-means[2])>EPS)
            insist(integer(t,'ganan_ambos_global_'+m,context)==both, f'{context}: ganan_ambos_global distinto')
            values = [x[3] for x in complete]
            loo = [statistics.fmean(values[:i]+values[i+1:]) for i in range(n)]
            assert len(loo)==n-0
            same(num(ro,'loo_min',context),min(loo),context+'/loo_min')
            same(num(ro,'loo_max',context),max(loo),context+'/loo_max')
            insist(integer(ro,'n_loo',context)==n, f'{context}: n_loo incorrecto')
            insist(integer(ro,'loo_signo_estable',context)==int(all(sign(x)==sign(means[3]) for x in loo)),
                   f'{context}: loo_signo_estable incorrecto')
            for s,field in [(1,'n_loo_favorables'),(-1,'n_loo_contrarios'),(0,'n_loo_empate')]:
                insist(integer(ro,field,context)==sum(sign(x)==s for x in loo), f'{context}: {field} incorrecto')
        coverage.append(integer(t,'n_enunciados_comunes_delta_rango_f0_st_sin_alertas',ref))
    insist(coverage == [9,4,5,5,6,6], f'Cobertura tonal filtrada inesperada: {coverage}')
    files = {name:sha(res/name) for name in SOURCE_FILES}
    return dict(trios=[tri_index[(r,)] for r in REFERENCES],
                frases=frases, robust=robust_index,
                source_hashes=files, source_dir=str(res),
                counts={'trios':len(trios),'frases':len(frases),'robustez':len(robust)},
                metric_coverage=coverage)
