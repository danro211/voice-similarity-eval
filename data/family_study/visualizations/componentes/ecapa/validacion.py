# Verificacion de los datos utilizados en las figuras ECAPA
#
# Se comparan las puntuaciones individuales con los promedios
# por enunciado, por pareja y los contrastes familiares
# Las tablas deben coincidir en identificadores, valores y cantidades
"""Revisa la coherencia de las tablas ECAPA antes de utilizarlas en los graficos"""
from __future__ import annotations
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from componentes.validacion_resumen import (
    REFERENCES, EPS, csvfile, insist, integer, load_and_check, num, same,
    sha, checked_table,
)

FILES83 = ('ecapa_comparaciones.csv','ecapa_resumen_enunciados.csv',
           'ecapa_resumen_parejas.csv','protocolo_comparaciones.csv')
GROUPS = ('misma_persona','hermanos','control')
COUNTS = {'misma_persona': (149,149,15), 'hermanos': (236,59,6), 'control': (472,118,12)}


# Se identifica una pareja y el enunciado que se esta comparando
def _key(row):
    return row['id_comparacion'], row['enunciado']


# Se cargan los resultados ECAPA y se revisa su coherencia
def load83(base):
    base=Path(base).resolve()
    baseline=load_and_check(base)
    res=base/'results'
    raw=csvfile(res/FILES83[0],
        ('id_comparacion','grupo','pareja_familiar_referencia','enunciado','toma_a',
         'toma_b','archivo_a','archivo_b','participante_a','participante_b','similitud_coseno'),857)
    phrases=csvfile(res/FILES83[1],
        ('id_comparacion','grupo','pareja_familiar_referencia','enunciado',
         'participante_a','participante_b','n_puntuaciones','media_coseno'),326)
    pairs=csvfile(res/FILES83[2],
        ('id_comparacion','grupo','pareja_familiar_referencia','participante_a',
         'participante_b','n_enunciados','n_puntuaciones','media_coseno'),33)
    protocol=csvfile(res/FILES83[3],
        ('id_comparacion','grupo','pareja_familiar_referencia','participante_a',
         'participante_b','n_enunciados'),33)
    pair_idx=checked_table(pairs, ['id_comparacion'],'resumen de parejas')
    proto_idx=checked_table(protocol, ['id_comparacion'],'protocolo')
    ph_idx=checked_table(phrases, ['id_comparacion','enunciado'],'resumen de frases')
    insist(set(pair_idx)==set(proto_idx),'parejas distintas del protocolo')

    for g, (nr,np,npar) in COUNTS.items():
        for name, table, expected in (('comparaciones',raw,nr),('frases',phrases,np),('parejas',pairs,npar)):
            insist(sum(r['grupo']==g for r in table)==expected,
                   f'{name}/{g}: tamaño inesperado')
    insist(set(r['grupo'] for r in raw+phrases+pairs) == set(GROUPS),'grupos inesperados')
    members=defaultdict(list)
    for q in raw:
        ident=q['id_comparacion']; en=q['enunciado']; p=pair_idx.get((ident,))
        insist(p is not None,f'comparación huérfana {ident}')
        insist(en in [f'E{i:02d}' for i in range(1,11)],f'frase inválida {ident}/{en}')
        for col in ('grupo','pareja_familiar_referencia','participante_a','participante_b'):
            insist(q[col]==p[col],f'{ident}/{en}: {col} no coincide')
        score=num(q,'similitud_coseno',ident+'/'+en)
        insist(-1<=score<=1,f'{ident}/{en}: coseno fuera [-1,1]')
        insist(q['archivo_a']!=q['archivo_b'],f'{ident}/{en}: mismo archivo comparado')
        insist(q['toma_a'] in ('T01','T02') and q['toma_b'] in ('T01','T02'),
               f'{ident}/{en}: toma inesperada')
        members[(ident,en)].append(q)
    insist(set(members)==set(ph_idx), 'combinaciones por frase incompletas o sobrantes')
    by_pair=defaultdict(list)
    for (ident,en), rr in members.items():
        par=pair_idx[(ident,)]; phrase=ph_idx[(ident,en)]
        for col in ('grupo','pareja_familiar_referencia','participante_a','participante_b'):
            insist(phrase[col]==par[col],f'{ident}/{en}: metadato diferente')
        expected=1 if par['grupo']=='misma_persona' else 4
        insist(len(rr)==expected,f'{ident}/{en}: {len(rr)} tomas, se esperaban {expected}')
        if expected==1:
            insist((rr[0]['toma_a'],rr[0]['toma_b'])==('T01','T02'),
                   f'{ident}/{en}: tomas propia persona incorrectas')
        else:
            combos={(r['toma_a'],r['toma_b']) for r in rr}
            insist(combos=={('T01','T01'),('T01','T02'),('T02','T01'),('T02','T02')},
                   f'{ident}/{en}: combinaciones de tomas inesperadas')
        insist(integer(phrase,'n_puntuaciones',ident)==expected,'n_puntuaciones inconsistente')
        same(num(phrase,'media_coseno',ident),statistics.fmean(num(x,'similitud_coseno',ident) for x in rr),
             f'{ident}/{en}/media_coseno')
        by_pair[ident].append(phrase)
    insist(len(by_pair)==33,'faltan resúmenes de pares')
    for (ident,),par in pair_idx.items():
        proto=proto_idx[(ident,)]; lista=by_pair[ident]
        for col in ('grupo','pareja_familiar_referencia','participante_a','participante_b'):
            insist(proto[col]==par[col],f'{ident}: protocolo {col} distinto')
        n=integer(par,'n_enunciados',ident)
        insist(n==integer(proto,'n_enunciados',ident)==len(lista),f'{ident}: frases vs protocolo')
        expected=1 if par['grupo']=='misma_persona' else 4
        insist(integer(par,'n_puntuaciones',ident)==n*expected,f'{ident}: n puntuaciones')
        expected_ph={f'E{i:02d}' for i in range(1,11)}
        if par['pareja_familiar_referencia']==REFERENCES[3] or 'F04_P11' in (par['participante_a'],par['participante_b']):
            expected_ph.remove('E08')
        insist({r['enunciado'] for r in lista}==expected_ph,f'{ident}: diseño de frases distinto')
        same(num(par,'media_coseno',ident),
             statistics.fmean(num(r,'media_coseno',ident) for r in lista),
             f'{ident}/media_pareja')
    # Los contrastes de las tres parejas se comparan con las tablas integradas
    for t in baseline['trios']:
        ref=t['pareja_familiar_referencia']
        group=[p for p in pairs if p['pareja_familiar_referencia']==ref]
        insist(Counter(p['grupo'] for p in group)=={'hermanos':1,'control':2},
               f'{ref}: deben existir H, CA y CB')
        h=next(p for p in group if p['grupo']=='hermanos')
        a,b=t['participante_hermano_a'],t['participante_hermano_b']
        insist((h['participante_a'],h['participante_b'])==(a,b),f'{ref}: orden de hermanos')
        ca=next((p for p in group if p['grupo']=='control' and p['participante_a']==a),None)
        cb=next((p for p in group if p['grupo']=='control' and p['participante_a']==b),None)
        insist(ca is not None and cb is not None,f'{ref}: control A/B ausente')
        insist(ca['participante_b']==cb['participante_b']==t['participante_control'],
               f'{ref}: controlador de referencia no coincide')
        for r,col in ((h,'media_hermanos_similitud_coseno'),
                      (ca,'media_control_a_similitud_coseno'),
                      (cb,'media_control_b_similitud_coseno')):
            same(num(r,'media_coseno',ref), num(t,col,ref),f'{ref}/{col}')
        # Se comprueban los resultados de cada frase y sus valores medios
        for f in (x for x in baseline['frases'] if x['pareja_familiar_referencia']==ref):
            en=f['enunciado']
            for r,col in ((h,'hermanos_similitud_coseno'),
                          (ca,'control_a_similitud_coseno'),
                          (cb,'control_b_similitud_coseno')):
                insist(f['id_'+('hermanos' if r is h else 'control_a' if r is ca else 'control_b')]==r['id_comparacion'],
                       f'{ref}/{en}: identidad de comparación')
                same(num(ph_idx[(r['id_comparacion'],en)],'media_coseno',ref),num(f,col,ref),
                     f'{ref}/{en}/{col}')
    sources={name:sha(res/name) for name in FILES83}
    sources.update(baseline['source_hashes'])
    return dict(baseline=baseline,raw=raw,phrases=phrases,pairs=pairs,protocol=protocol,
                by_pair=by_pair,source_hashes=sources,source_dir=str(res),
                counts={'puntuaciones':len(raw),'resumen_enunciados':len(phrases),
                        'resumen_parejas':len(pairs),'trios':6,'frases_trios':59})
