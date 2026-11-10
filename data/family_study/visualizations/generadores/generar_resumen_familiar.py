#!/usr/bin/env python3
# Generacion de las figuras V05, V06 y V15
#
# Se leen los resultados integrados y se comprueba su consistencia
# El modo probar-render dibuja las figuras sin guardarlas
# El modo generar exporta PDF, SVG y PNG junto con un manifiesto
# Las exportaciones anteriores se conservan sin sobrescribirlas
"""Genera V05, V06 y V15 a partir de los resultados integrados del estudio"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))
from componentes.validacion_resumen import load_and_check, sha
from componentes.estilo import apply
from componentes.graficos_resumen import RENDERERS, FIGURES
from componentes import __version__


# Se muestra un error explicativo cuando falla una validacion
def fail(msg):
    raise SystemExit('ERROR: '+str(msg))


# Se reciben los argumentos, se revisan los datos y se dibujan las figuras
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=SRC.parent,help='Carpeta privada data/family_study/')
    p.add_argument('--modo',choices=['verificar','probar-render','generar'],default='verificar')
    p.add_argument('--figura',choices=['all']+list(RENDERERS),default='all')
    p.add_argument('--run-id',default='',help='ID irrepetible: solo letras ASCII, números, guion, guion bajo')
    args=p.parse_args()
    if args.run_id and not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.run_id):
        fail('run-id inválido')
    base=args.base.expanduser().resolve()
    if not (base/'results').is_dir():
        fail(f'No se encuentra results/ dentro de {base} (usa --base)')
    try:
        data=load_and_check(base)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        fail(e)
    names=list(RENDERERS) if args.figura=='all' else [args.figura]
    print('=== PASO 8.2: INFRAESTRUCTURA VISUAL ===')
    print('Modo:',args.modo)
    print('Entradas:',str(data['source_dir']))
    print('Comprobaciones: 6 tríos, 59 frases, 66 robustez; 11 métricas validadas')
    print('Cobertura F0 filtrada:',data['metric_coverage'])
    print('Pilotos:',', '.join(names))
    if args.modo == 'verificar':
        print('VERIFICACION CORRECTA: cero archivos creados por este programa.')
        return
    import io
    import matplotlib
    apply()
    import matplotlib.pyplot as plt
    images=[]; prov=[]
    if args.modo=='probar-render':
        for name in names:
            fig,spec=RENDERERS[name](data)
            with io.BytesIO() as buffer:
                fig.savefig(buffer,format='png',dpi=130,facecolor='white')
                assert buffer.tell()>5000
            plt.close(fig)
            print('RENDER OK:',name)
        print('PRUEBA SIN EXPORTACION: ninguna figura se escribió en disco por el programa.')
        return
    # La exportacion utiliza una carpeta nueva para evitar sobrescrituras
    exportroot=base/'visualizations'/'exports'
    runid=args.run_id or datetime.now(timezone.utc).strftime('p8_2_%Y%m%dT%H%M%SZ')
    final=exportroot/runid
    exportroot.mkdir(parents=True,exist_ok=True)
    try:
        final.mkdir(mode=0o700,exist_ok=False)
    except FileExistsError:
        fail('Ya existe esta ejecución; elige otro --run-id. No se sobrescribió nada.')
    try:
        for name in names:
            fig,spec=RENDERERS[name](data)
            try:
                filehashes={}
                for ext in ('pdf','svg','png'):
                    target=final/f'{name}_{name.lower()}_piloto.{ext}'
                    fig.savefig(target,format=ext,dpi=300,facecolor='white',bbox_inches=None)
                    filehashes[target.name]=sha(target)
                    if ext=='png':
                        from PIL import Image
                        with Image.open(target) as img:
                            img.verify()
                images.append(dict(id=name,title=FIGURES[name]['title'],
                    sources=FIGURES[name]['inputs'],sha256_output=filehashes,
                    data_snapshot=spec))
                print('EXPORTADO:',name,'PDF + SVG + PNG (300 dpi)')
            finally:
                plt.close(fig)
        code_files=sorted(x for x in (SRC/'componentes').rglob('*.py') if '__pycache__' not in x.parts)
        meta={
          'proyecto':'UCR IE-0499 / Proyecto Eléctrico / Etapa 8.2',
          'estado':'pilotos; verificar visualmente antes de aprobar',
          'version':__version__,'run_id':runid,'modo':'generar','estilo':'atlas_v0.2','exportacion_dpi_png':300,'utc':datetime.now(timezone.utc).isoformat(),
          'python':sys.version.split()[0],
          'librerias':{'matplotlib':matplotlib.__version__},
          'esquemas':data['counts'],
          'input_sha256':data['source_hashes'],
          'codigo_sha256':{str(x.relative_to(SRC)):sha(x) for x in code_files},
          'figuras':images,'advertencia':'No inferencial: LOO no equivale a IC; archivos privados.'
        }
        with (final/'manifest_8_2.json').open('x',encoding='utf-8') as out:
            json.dump(meta,out,indent=2,ensure_ascii=False,allow_nan=False)
            out.write('\n')
        print('MANIFIESTO:',final/'manifest_8_2.json')
        print('SALIDA:',final)
        print('FIGURAS PILOTO GENERADAS: pendientes de inspección visual en la VM.')
    except BaseException:
        # Solo se limpia la salida incompleta de la ejecucion actual
        shutil.rmtree(final)
        raise

if __name__=='__main__':
    main()
