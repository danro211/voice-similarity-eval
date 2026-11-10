#!/usr/bin/env python3
# Pruebas automaticas de las figuras V05, V06 y V15
#
# Se preparan tablas con valores inventados para comprobar el codigo
# No se necesitan grabaciones ni informacion de los participantes
# Se revisan las relaciones numericas y la generacion de las figuras
"""Prueba las figuras V05, V06 y V15 mediante tablas ficticias, sin utilizar voces reales"""
from __future__ import annotations
import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GEN=ROOT/'generadores'/'generar_resumen_familiar.py'
REFS=(
 'F01_P01__F01_P02','F02_P04__F02_P06','F03_P08__F03_P09',
 'F04_P10__F04_P11','F05_P12__F05_P13','F06_P14__F06_P15'
)
METRICAS=(
 'similitud_coseno','delta_f0_st','delta_rango_f0_st','delta_rango_f0_st_sin_alertas',
 'delta_duracion_total_s','delta_duracion_total_pct','delta_habla_estimada_s',
 'delta_habla_estimada_pct','delta_fraccion_habla_pp',
 'delta_velocidad_total_pal_s','delta_velocidad_articulacion_pal_s'
)


# Se escribe un CSV ficticio con las columnas esperadas
def write(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


# Se construyen ejemplos de familias, frases y contrastes
def fixture(root):
    r=root/'results';r.mkdir(parents=True)
    trios, frases, robustez=[],[],[]
    coverage=[9,4,5,5,6,6]
    for idx,ref in enumerate(REFS):
        n_design=9 if idx==3 else 10
        tf={'pareja_familiar_referencia':ref,'n_enunciados_diseno':n_design}
        phrases=[{'pareja_familiar_referencia':ref,'enunciado':f'E{i:02d}'}
                 for i in range(1,11) if not (idx==3 and i==8)]
        for midx,m in enumerate(METRICAS):
            n=coverage[idx] if m.endswith('sin_alertas') else n_design
            complete=[]
            for i,p in enumerate(phrases):
                ok=(i<n)
                # Los valores ficticios mantienen una relacion consistente entre parejas
                delta=.07+idx*.006+(i%4-1.5)*.003 if idx%2==0 else -.055+(i%3)*.006
                if m=='similitud_coseno':
                    h=.36+idx*.025+(i%3)*.012; ca=h-delta-.018;cb=h-delta+.018
                else:
                    h=2.2+midx*.04+idx*.07; ca=h+delta+.025;cb=h+delta-.025
                if ok:
                    p.update({f'hermanos_{m}':h,f'control_a_{m}':ca,
                              f'control_b_{m}':cb,
                              f'contraste_favorable_{m}':delta,
                              f'gana_ambos_{m}':int(delta>.025)})
                    complete.append((h,ca,cb,delta,int(delta>.025)))
                else:
                    p.update({f'hermanos_{m}':'',f'control_a_{m}':'',
                              f'control_b_{m}':'',f'contraste_favorable_{m}':'',
                              f'gana_ambos_{m}':''})
            hs=sum(r[0] for r in complete)/n
            ca=sum(r[1] for r in complete)/n
            cb=sum(r[2] for r in complete)/n
            deltas=[r[3] for r in complete]
            contrast=sum(deltas)/n
            fav=sum(x>1e-9 for x in deltas)
            contra=sum(x< -1e-9 for x in deltas)
            emp=n-fav-contra
            wins=sum(r[4] for r in complete)
            loo=[sum(deltas[:i]+deltas[i+1:])/(n-1) for i in range(n)]
            sg=lambda x: (x>1e-9)-(x< -1e-9)
            st=int(all(sg(x)==sg(contrast) for x in loo))
            order=1 if m=='similitud_coseno' else -1
            tf.update({
                f'n_enunciados_comunes_{m}':n,
                f'media_hermanos_{m}':hs,
                f'media_control_a_{m}':ca,
                f'media_control_b_{m}':cb,
                f'media_controles_{m}':(ca+cb)/2,
                f'contraste_favorable_{m}':contrast,
                f'n_enunciados_favorables_{m}':fav,
                f'n_enunciados_contrarios_{m}':contra,
                f'n_enunciados_empate_{m}':emp,
                f'n_enunciados_gana_ambos_{m}':wins,
                f'ganan_ambos_global_{m}':int(order*(hs-ca)>1e-9 and order*(hs-cb)>1e-9),
            })
            robustez.append({
              'pareja_familiar_referencia':ref,'metrica':m,
              'n_enunciados_comunes':n,'contraste_favorable':contrast,
              'n_enunciados_favorables':fav,'n_enunciados_contrarios':contra,
              'n_enunciados_empate':emp,'n_enunciados_gana_ambos':wins,
              'n_loo':n,'loo_min':min(loo),'loo_max':max(loo),
              'loo_signo_estable':st,
              'n_loo_favorables':sum(sg(x)>0 for x in loo),
              'n_loo_contrarios':sum(sg(x)<0 for x in loo),
              'n_loo_empate':sum(sg(x)==0 for x in loo),
            })
        trios.append(tf)
        frases.extend(phrases)
    write(r/'contrastes_integrados_trios.csv',trios)
    write(r/'contrastes_integrados_enunciados.csv',frases)
    write(r/'robustez_integrada.csv',robustez)
    (r/'integracion_paso7_config.json').write_text(json.dumps({'paso':7,'origen':'SOLO DATOS SINTETICOS'}))


# Se prueba el renderizado y se rechazan las entradas inconsistentes
class TestPilotos(unittest.TestCase):
    def run_gen(self, base, mode, *extra, good=True):
        p=subprocess.run([sys.executable,'-B',str(GEN),'--base',str(base),
                          '--modo',mode,*extra],capture_output=True,text=True)
        self.assertEqual(p.returncode==0,good,f'{p.stdout}\n{p.stderr}')
        return p

    def test_synthetic_full_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);fixture(base)
            self.run_gen(base,'verificar')
            self.assertFalse((base/'visualizations').exists())
            self.run_gen(base,'probar-render')
            self.assertFalse((base/'visualizations').exists())
            self.run_gen(base,'generar','--run-id','synthetic_001')
            out=base/'visualizations'/'exports'/'synthetic_001'
            files=list(out.iterdir())
            self.assertEqual(len(files),10)  # Se esperan tres figuras en tres formatos y un manifiesto
            manifest=json.loads((out/'manifest_8_2.json').read_text())
            self.assertEqual(len(manifest['figuras']),3)
            self.assertEqual(manifest['run_id'],'synthetic_001')
            self.assertEqual(manifest['modo'],'generar')
            self.assertEqual(manifest['version'],'0.2.0')
            self.assertEqual(manifest['estilo'],'atlas_v0.2')
            v15=next(x for x in manifest['figuras'] if x['id']=='V15')
            self.assertEqual(len(v15['data_snapshot']['stable']),6)
            self.assertEqual(len(v15['data_snapshot']['groups']),5)
            self.assertEqual(manifest['esquemas'],{'trios':6,'frases':59,'robustez':66})
            self.run_gen(base,'generar','--run-id','synthetic_001',good=False)
            self.assertEqual(len(list(out.iterdir())),10)

    def test_falsified_data_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);fixture(base)
            p=base/'results'/'contrastes_integrados_trios.csv'
            with p.open(newline='') as stream:
                rd=csv.DictReader(stream);data=list(rd)
            data[0]['contraste_favorable_similitud_coseno']='0.987654'
            write(p,data)
            r=self.run_gen(base,'verificar',good=False)
            self.assertIn('contraste_favorable_',r.stderr)
            self.assertFalse((base/'visualizations').exists())

    def test_missing_phrase_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);fixture(base)
            p=base/'results'/'contrastes_integrados_enunciados.csv'
            with p.open(newline='') as stream:
                rd=csv.DictReader(stream);data=list(rd)
            data[0]['enunciado']='E08'  # Se repite E08 para comprobar que se detectan enunciados duplicados
            write(p,data)
            self.run_gen(base,'verificar',good=False)


if __name__=='__main__':
    unittest.main(verbosity=2)
