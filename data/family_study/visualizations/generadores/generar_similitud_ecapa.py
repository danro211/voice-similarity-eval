#!/usr/bin/env python3
# Generacion de las figuras ECAPA V01 a V04
#
# Se utilizan las tablas de similitud que ya fueron calculadas
# La figura V01 se utiliza en el informe y las otras permiten
# examinar las parejas y los enunciados con mayor detalle
# La verificacion y la prueba de renderizado no exportan figuras
"""Genera las figuras ECAPA a partir de resultados ya calculados, sin modificar los CSV originales"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
# Se cargan los modulos de graficacion sin modificar los programas anteriores
VISUAL=ROOT.parent
sys.path.insert(0,str(VISUAL))
from componentes.estilo import apply
from componentes.validacion_resumen import sha
from componentes.ecapa.validacion import load83
from componentes.ecapa.graficos import FIGURES83, RENDERERS83


# Se seleccionan las figuras, se validan los CSV y se prepara la exportacion
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=VISUAL.parent)
    p.add_argument('--modo',choices=('verificar','probar-render','generar'),default='verificar')
    p.add_argument('--figura',choices=['all']+list(RENDERERS83),default='all')
    p.add_argument('--run-id',default='')
    args=p.parse_args()
    if args.run_id and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.run_id):
        p.error('El run-id solo admite ASCII, números, _ y - (1–64 caracteres)')
    base=args.base.expanduser().resolve()
    try:
        ds=load83(base)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        p.error(f'Entradas no verificadas: {exc}')
    names=list(RENDERERS83) if args.figura=='all' else [args.figura]
    print('=== IE-0499 · PASO 8.3 · ECAPA ===')
    print('Versión módulo: 0.1.1')
    print('Modo:',args.modo)
    print('Datos:',ds['source_dir'])
    print('Validaciones: 857 puntuaciones / 326 enunciados / 33 parejas / 6 tríos / 59 frases')
    print('Figuras:',', '.join(names))
    if args.modo=='verificar':
        print('VALIDACIÓN CORRECTA. Sin escritura de archivos.')
        return
    import matplotlib
    apply()
    import matplotlib.pyplot as plt
    if args.modo=='probar-render':
        for k in names:
            fig,spec=RENDERERS83[k](ds)
            try:
                with io.BytesIO() as buf:
                    fig.savefig(buf,format='png',dpi=115,facecolor='white',bbox_inches=None)
                    if buf.tell()<7000:
                        raise ValueError(f'Figura insuficiente: {k}')
            finally:
                plt.close(fig)
            print('RENDER OK:',k)
        print('No se exportaron archivos.')
        return
    exportroot=base/'visualizations'/'exports'
    runid=args.run_id or datetime.now(timezone.utc).strftime('p8_3_%Y%m%dT%H%M%SZ')
    final=exportroot/runid
    exportroot.mkdir(parents=True,exist_ok=True)
    try:
        final.mkdir(mode=0o700,exist_ok=False)
    except FileExistsError:
        p.error('Ya existe --run-id; jamás se sobrescribe una ejecución anterior')
    try:
        entries=[]
        for k in names:
            fig,spec=RENDERERS83[k](ds)
            try:
                files={}
                for ext in ('pdf','svg','png'):
                    target=final/f'{k}_{k.lower()}_ecapa.{ext}'
                    fig.savefig(target,format=ext,dpi=300,facecolor='white',bbox_inches=None)
                    if ext=='png':
                        from PIL import Image
                        with Image.open(target) as img:img.verify()
                    files[target.name]=sha(target)
            finally:
                plt.close(fig)
            entries.append({'id':k,'titulo':FIGURES83[k]['title'],
                            'fuentes':FIGURES83[k]['sources'],
                            'sha256_output':files,'data_snapshot':spec})
            print('EXPORTADO:',k,'PDF + SVG + PNG')
        manifest={
            'proyecto':'UCR IE-0499 · 8.3 · ECAPA',
            'version':'0.1.1','atlas_base':'v0.2 congelado',
            'run_id':runid,'modo':'generar','utc':datetime.now(timezone.utc).isoformat(),
            'python':sys.version.split()[0],'matplotlib':matplotlib.__version__,
            'esquemas':ds['counts'],
            'input_sha256':ds['source_hashes'],
            'codigo_sha256':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},
            'atlas_base_codigo_sha256':{str(p.relative_to(VISUAL)):sha(p) for p in sorted((VISUAL/'componentes').rglob('*.py'))},
            'figuras':entries,
            'advertencia':'Descriptivo; frases/tomas/hablantes no son observaciones independientes; sin intervalos inferenciales. Datos privados.'}
        target=final/'manifest_8_3.json'
        with target.open('x',encoding='utf-8') as out:
            json.dump(manifest,out,indent=2,ensure_ascii=False,allow_nan=False)
            out.write('\n')
        print('MANIFIESTO:',target)
        print('SALIDA:',final)
        print('PENDIENTE: inspección visual de imágenes reales por el usuario.')
    except BaseException:
        shutil.rmtree(final)
        raise

if __name__=='__main__':main()
