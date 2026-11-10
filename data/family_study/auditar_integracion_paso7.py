#!/usr/bin/env python3
"""Comprueba de forma independiente los resultados integrados del estudio

Recalcula las medias, los contrastes familiares y el analisis leave-one-out
Usa tablas privadas de resultados, pero no abre audios ni archivos de participantes
No depende de las funciones del programa que produjo esas tablas
"""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'results'
TRIOS = [
    ('F01_P01','F01_P02','F02_P07'),('F02_P04','F02_P06','F03_P08'),
    ('F03_P08','F03_P09','F02_P07'),('F04_P10','F04_P11','F02_P03'),
    ('F05_P12','F05_P13','F02_P05'),('F06_P14','F06_P15','F03_P09'),
]
EPS = 1e-9

# Leemos el CSV que vamos a contrastar con nuestros propios calculos
def load(name):
    with (ROOT / name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

# Comprobamos que la celda guardada coincida con el valor que obtuvimos
def verify(real, expected, where):
    if expected is None:
        assert real in ('',None), (where, real, expected)
    else:
        assert real not in ('',None) and math.isclose(float(real),expected,rel_tol=1e-8,abs_tol=1e-8), (where,real,expected)

# Si no hay datos, dejamos la media sin valor en lugar de usar cero
def avg(vals):
    return sum(vals)/len(vals) if vals else None

# Clasificamos la direccion del contraste usando una tolerancia pequena
def sgn(v):
    return 1 if v>EPS else (-1 if v<-EPS else 0)

# Usamos identificadores para encontrar cada fila y detectar duplicados
def unique(rows,keys):
    out={}
    for r in rows:
        k=tuple(r[x] for x in keys)
        assert k not in out,k
        out[k]=r
    return out

c = json.loads((ROOT/'integracion_paso7_config.json').read_text(encoding='utf8'))
SOURCE_FILES = {
    'protocolo':'protocolo_comparaciones.csv',
    'ecapa_individual':'ecapa_comparaciones.csv',
    'ecapa_enunciado':'ecapa_resumen_enunciados.csv',
    'ecapa_pareja':'ecapa_resumen_parejas.csv',
    'acustica_individual':'acustica_comparaciones.csv',
    'acustica_enunciado':'acustica_resumen_enunciados.csv',
    'acustica_pareja':'acustica_resumen_parejas.csv',
    'acustica_trios':'acustica_contrastes_trios.csv',
    'acustica_config':'acustica_comparaciones_config.json',
}
for role,fname in SOURCE_FILES.items():
    assert c['sha256_entradas'][role]==hashlib.sha256((ROOT/fname).read_bytes()).hexdigest(),fname
assert c['cantidad_salidas'] == {
    'enunciados':326,'parejas':33,'frases_trios':59,'trios':6,'robustez':66}
met=['similitud_coseno']+[x for x in c['metricas'] if x!='similitud_coseno']
assert len(met)==11 and len(set(met))==11

# Comenzamos recalculando los promedios de cada enunciado a partir de las tomas
raw_ec=load('ecapa_comparaciones.csv')
raw_ac=load('acustica_comparaciones.csv')
frases_in=load('resumen_integrado_enunciados.csv')
parejas_in=load('resumen_integrado_parejas.csv')
frases_trios=load('contrastes_integrados_enunciados.csv')
trios_in=load('contrastes_integrados_trios.csv')
loo_in=load('robustez_integrada.csv')
assert [len(x) for x in (raw_ec,raw_ac,frases_in,parejas_in,frases_trios,trios_in,loo_in)] == [857,857,326,33,59,6,66]
raw_key=('id_comparacion','enunciado','toma_a','toma_b')
ec=unique(raw_ec,raw_key); ac=unique(raw_ac,raw_key)
assert set(ec)==set(ac)

by_phrase=defaultdict(list)
for key,e in ec.items():
    a=ac[key]
    for col in ('grupo','pareja_familiar_referencia','participante_a','participante_b','archivo_a','archivo_b'):
        assert e[col]==a[col], (key,col)
    by_phrase[key[:2]].append((e,a))
assert len(by_phrase)==326

frase_calc={}
for key, rows in by_phrase.items():
    s={}
    s['similitud_coseno']=avg([float(ec_row['similitud_coseno']) for ec_row,_ in rows])
    for metric in met[1:]:
        valid=[float(ac_row[metric]) for _,ac_row in rows if ac_row[metric]!='']
        s[metric]=(avg(valid) if len(valid)==len(rows) or not metric.endswith('_sin_alertas') else None)
    for flag, source in [('f0','f0_alertada'),('dispersion_f0','dispersion_f0_alertada'),('vad','vad_alertada')]:
        s['n_'+flag+'_alertadas']=sum(int(row[source]) for _,row in rows)
    frase_calc[key]=s

phrase_new=unique(frases_in,('id_comparacion','enunciado'))
assert set(phrase_new)==set(frase_calc)
for key,data in frase_calc.items():
    output=phrase_new[key]
    for metric in met:
        verify(output[metric],data[metric], (key,metric))
    for attr in ('n_f0_alertadas','n_dispersion_f0_alertadas','n_vad_alertadas'):
        assert int(output[attr])==data[attr]
print('AUDITORIA 1/4: 857 puntuaciones + 326 frases recalculadas OK')

# Despues calculamos los promedios por pareja con los datos de sus enunciados
by_pair=defaultdict(list)
for (pid,e),scores in frase_calc.items():
    by_pair[pid].append(scores)
pair_new=unique(parejas_in,('id_comparacion',))
assert len(by_pair)==len(pair_new)==33
for (pid,),r in pair_new.items():
    frases=by_pair[pid]
    assert int(r['n_enunciados'])==len(frases)
    for metric in met:
        values=[a[metric] for a in frases if a[metric] is not None]
        verify(r[metric],avg(values),(pid,metric))
        if metric!='similitud_coseno':
            assert int(r['n_enunciados_'+metric])==len(values)
print('AUDITORIA 2/4: 33 medias de parejas recalculadas OK')

# Por ultimo revisamos los contrastes familiares y su sensibilidad al quitar frases
by_ref=defaultdict(list)
for p in parejas_in:
    if p['grupo']!='misma_persona':
        by_ref[p['pareja_familiar_referencia']].append(p)
trio_new=unique(trios_in,('pareja_familiar_referencia',))
frase_trio_new=unique(frases_trios,('pareja_familiar_referencia','enunciado'))
loo_new=unique(loo_in,('pareja_familiar_referencia','metrica'))
assert len(trio_new)==6 and len(frase_trio_new)==59 and len(loo_new)==66

for h1,h2,c in TRIOS:
    ref=h1+'__'+h2
    rows=by_ref[ref]
    assert len(rows)==3
    h=next(x for x in rows if x['grupo']=='hermanos')
    a=next(x for x in rows if x['grupo']=='control' and {x['participante_a'],x['participante_b']}=={h1,c})
    b=next(x for x in rows if x['grupo']=='control' and {x['participante_a'],x['participante_b']}=={h2,c})
    assert {h['participante_a'],h['participante_b']}=={h1,h2}
    tid={'h':h['id_comparacion'],'a':a['id_comparacion'],'b':b['id_comparacion']}
    ens={en for pid,en in frase_calc if pid==tid['h']}
    assert ens==({f'E{i:02d}' for i in range(1,11)} - ({'E08'} if h2=='F04_P11' else set()))
    trio=trio_new[(ref,)]
    assert int(trio['n_enunciados_diseno'])==len(ens)
    for metric in met:
        records=[]
        for en in ens:
            parts=[frase_calc[(tid[x],en)][metric] for x in ('h','a','b')]
            out=frase_trio_new[(ref,en)]
            for key, v in zip(('hermanos','control_a','control_b'),parts):
                verify(out[f'{key}_{metric}'],v,(ref,en,key,metric))
            hval,aval,bval=parts
            if None in parts:
                verify(out[f'contraste_favorable_{metric}'],None,(ref,en,metric))
                continue
            multiplier=1 if metric=='similitud_coseno' else -1
            delta=multiplier*(hval-(aval+bval)/2)
            both=int(multiplier*(hval-aval)>EPS and multiplier*(hval-bval)>EPS)
            verify(out[f'contraste_favorable_{metric}'],delta,(ref,en,metric))
            assert int(out[f'gana_ambos_{metric}'])==both
            records.append((hval,aval,bval,delta,both))
        n=len(records)
        assert int(trio[f'n_enunciados_comunes_{metric}'])==n
        assert len(records)>=2, (ref,metric)
        hm=avg([r[0] for r in records]); am=avg([r[1] for r in records]); bm=avg([r[2] for r in records])
        delta=avg([r[3] for r in records]); multiplier=1 if metric=='similitud_coseno' else -1
        expected={
            'media_hermanos':hm,'media_control_a':am,'media_control_b':bm,
            'media_controles':(am+bm)/2,
            'contraste_favorable':delta,
            'contraste_crudo_hermanos_menos_controles':multiplier*delta,
            'margen_minimo_a_controles':min(multiplier*(hm-am),multiplier*(hm-bm))}
        for attr,val in expected.items():
            verify(trio[f'{attr}_{metric}'],val,(ref,metric,attr))
        assert int(trio[f'ganan_ambos_global_{metric}'])==int(
            multiplier*(hm-am)>EPS and multiplier*(hm-bm)>EPS)
        assert int(trio[f'n_enunciados_favorables_{metric}'])==sum(sgn(r[3])>0 for r in records)
        assert int(trio[f'n_enunciados_contrarios_{metric}'])==sum(sgn(r[3])<0 for r in records)
        assert int(trio[f'n_enunciados_empate_{metric}'])==sum(sgn(r[3])==0 for r in records)
        assert int(trio[f'n_enunciados_gana_ambos_{metric}'])==sum(r[4] for r in records)
        robust=loo_new[(ref,metric)]
        leave=[(sum(r[3] for r in records)-r[3])/(n-1) for r in records]
        verify(robust['contraste_favorable'],delta,(ref,metric,'robust'))
        verify(robust['loo_min'],min(leave),(ref,metric,'min'))
        verify(robust['loo_max'],max(leave),(ref,metric,'max'))
        assert int(robust['n_loo'])==n
        assert int(robust['n_loo_favorables'])==sum(sgn(x)>0 for x in leave)
        assert int(robust['n_loo_contrarios'])==sum(sgn(x)<0 for x in leave)
        assert int(robust['n_loo_empate'])==sum(sgn(x)==0 for x in leave)
        assert int(robust['loo_signo_estable'])==int(all(sgn(x)==sgn(delta) for x in leave))
print('AUDITORIA 3/4: 59 contrastes por frase y 6 por familia OK')
print('AUDITORIA 4/4: 66 pruebas leave-one-enunciado-out OK')
print('PASO 7: AUDITORIA INDEPENDIENTE SUPERADA')
