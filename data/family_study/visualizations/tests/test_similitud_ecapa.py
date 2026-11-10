#!/usr/bin/env python3
# Pruebas automaticas de las figuras ECAPA
#
# Se crean tablas ficticias en una carpeta temporal
# Las pruebas no necesitan grabaciones ni datos de los participantes
# Se comprueban los conteos, las validaciones y la generacion de figuras
"""Comprueba las figuras ECAPA con datos inventados y carpetas temporales"""
from __future__ import annotations
import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from collections import defaultdict

HERE=Path(__file__).resolve().parent
VIS=HERE.parents[0]
sys.path.insert(0,str(VIS/'tests'))
from test_resumen_familiar import fixture, write
GEN=VIS/'generadores'/'generar_similitud_ecapa.py'
REFS=[('F01_P01','F01_P02','F02_P07'),('F02_P04','F02_P06','F03_P08'),
      ('F03_P08','F03_P09','F02_P07'),('F04_P10','F04_P11','F02_P03'),
      ('F05_P12','F05_P13','F02_P05'),('F06_P14','F06_P15','F03_P09')]
IDS=[f'F{f:02d}_P{p:02d}' for f, ps in enumerate(((1,2),(3,4,5,6,7),(8,9),(10,11),(12,13),(14,15)),1) for p in ps]


# Se leen los archivos CSV preparados para las pruebas
def read(path):
    with path.open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


# Se construyen las tablas ficticias que utilizan los generadores
def fixture83(base):
    fixture(base)
    r=base/'results'
    integ=read(r/'contrastes_integrados_enunciados.csv')
    mapping={(f['pareja_familiar_referencia'],f['enunciado']):f for f in integ}
    trios=read(r/'contrastes_integrados_trios.csv')
    for row,(a,b,c) in zip(trios,REFS):
        row.update(participante_hermano_a=a,participante_hermano_b=b,participante_control=c)
    write(r/'contrastes_integrados_trios.csv',trios)
    raw,summary,pairs,protocol=[],[],[],[]

    def add(group,a,b,ref,ens,vals):
        ident=f'{group}_{len(pairs)}'
        p= {'id_comparacion':ident,'grupo':group,
            'pareja_familiar_referencia':ref,'participante_a':a,'participante_b':b,
            'n_enunciados':len(ens),'n_puntuaciones':len(ens)*(1 if group=='misma_persona' else 4)}
        averages=[]
        for en,v in zip(ens,vals):
            tweaks=[0.] if group=='misma_persona' else [-.018,-.006,.006,.018]
            combos=[('T01','T02')] if group=='misma_persona' else [
                ('T01','T01'),('T01','T02'),('T02','T01'),('T02','T02')]
            for k,(ta,tb) in enumerate(combos):
                raw.append({'id_comparacion':ident,'grupo':group,'pareja_familiar_referencia':ref,
                    'participante_a':a,'participante_b':b,'enunciado':en,
                    'toma_a':ta,'toma_b':tb,
                    'archivo_a':f'{a}_{en}_{ta}.wav','archivo_b':f'{b}_{en}_{tb}.wav',
                    'similitud_coseno':v+tweaks[k]})
            summary.append({'id_comparacion':ident,'grupo':group,'pareja_familiar_referencia':ref,
                'participante_a':a,'participante_b':b,'enunciado':en,
                'n_puntuaciones':len(combos),'media_coseno':v})
            averages.append(v)
        pairs.append({**p,'media_coseno':statistics.fmean(averages)})
        protocol.append({**p,'enunciados':';'.join(ens)})
        return ident

    for i,pid in enumerate(IDS):
        ens=[f'E{j:02d}' for j in range(1,11) if not (pid=='F04_P11' and j==8)]
        add('misma_persona',pid,pid,'',ens,[.71+.002*i+.0001*j for j in range(len(ens))])
    for a,b,c in REFS:
        ref=f'{a}__{b}'
        ens=[f'E{j:02d}' for j in range(1,11) if not (a=='F04_P10' and j==8)]
        coll=[mapping[(ref,en)] for en in ens]
        hi=add('hermanos',a,b,ref,ens,[float(x['hermanos_similitud_coseno']) for x in coll])
        ai=add('control',a,c,ref,ens,[float(x['control_a_similitud_coseno']) for x in coll])
        bi=add('control',b,c,ref,ens,[float(x['control_b_similitud_coseno']) for x in coll])
        for e in ens:
            mapping[(ref,e)].update({'id_hermanos':hi,'id_control_a':ai,'id_control_b':bi})
    write(r/'ecapa_comparaciones.csv',raw)
    write(r/'ecapa_resumen_enunciados.csv',summary)
    write(r/'ecapa_resumen_parejas.csv',pairs)
    write(r/'protocolo_comparaciones.csv',protocol)
    write(r/'contrastes_integrados_enunciados.csv',integ)
    assert (len(raw),len(summary),len(pairs))==(857,326,33)


# Se comprueba la exportacion y el rechazo de datos inconsistentes
class TestEcapa83(unittest.TestCase):
    def execute(self,base,modo,*options,success=True):
        p=subprocess.run([sys.executable,'-B',str(GEN),'--base',str(base),
                 '--modo',modo,*options],capture_output=True,text=True)
        self.assertEqual(p.returncode==0,success,p.stdout+'\n'+p.stderr)
        return p

    def test_full_pipeline_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);fixture83(base)
            self.execute(base,'verificar')
            self.assertFalse((base/'visualizations').exists())
            self.execute(base,'probar-render')
            self.assertFalse((base/'visualizations').exists())
            self.execute(base,'generar','--run-id','synthetic_83')
            out=base/'visualizations'/'exports'/'synthetic_83'
            self.assertEqual(len(list(out.iterdir())),13)
            manifest=json.loads((out/'manifest_8_3.json').read_text())
            self.assertEqual(manifest['run_id'],'synthetic_83')
            self.assertEqual([x['id'] for x in manifest['figuras']],['V01','V02','V03','V04'])
            self.assertEqual(manifest['esquemas']['puntuaciones'],857)
            self.assertEqual(manifest['esquemas']['resumen_enunciados'],326)
            self.assertEqual(manifest['esquemas']['resumen_parejas'],33)
            self.assertEqual(next(x for x in manifest['figuras'] if x['id']=='V03')['data_snapshot']['rows'],59)
            self.execute(base,'generar','--run-id','synthetic_83',success=False)
            self.assertEqual(len(list(out.iterdir())),13)

    def test_corrupted_cosine_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);fixture83(base)
            path=base/'results'/'ecapa_comparaciones.csv'
            rr=read(path);rr[0]['similitud_coseno']='0.001'
            write(path,rr)
            self.execute(base,'verificar',success=False)

    def test_missing_comparison_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);fixture83(base)
            path=base/'results'/'ecapa_resumen_enunciados.csv'
            rr=read(path);rr[0]['enunciado']='E10'
            write(path,rr)
            self.execute(base,'verificar',success=False)

if __name__=='__main__':unittest.main(verbosity=2)
