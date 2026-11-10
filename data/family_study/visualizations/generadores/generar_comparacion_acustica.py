#!/usr/bin/env python3
# Generacion de las figuras acusticas V08 y V09
#
# V08 presenta las diferencias de tono entre hermanos y controles
# V09 compara la duracion y la velocidad aproximada de articulacion
# En ambas figuras una distancia menor representa mayor semejanza
# Se utilizan los resultados integrados sin volver a calcular las medidas
"""Genera V08 y V09 con las distancias acusticas y temporales previamente calculadas"""
from __future__ import annotations
import argparse, hashlib, io, json, math, re, shutil, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
VISUAL=ROOT.parent
sys.path.insert(0,str(VISUAL))
from componentes.validacion_resumen import load_and_check, num, integer, same, sha, REFERENCES, insist
from componentes.estilo import apply, base_title, P

METRICS = [
 ('delta_f0_st','Distancia de F0 mediana', 'Diferencia tonal absoluta (semitonos)', 'TONO'),
 ('delta_rango_f0_st_sin_alertas','Diferencia de variabilidad F0 filtrada', 'Diferencia de rango tonal (semitonos)', 'TONO'),
 ('delta_duracion_total_pct','Diferencia relativa de duración', 'Diferencia de duración (%)', 'RITMO'),
 ('delta_velocidad_articulacion_pal_s','Diferencia de velocidad de articulación', 'Diferencia absoluta (palabras/s)', 'RITMO'),
]
FIGS = {'V08': METRICS[:2], 'V09': METRICS[2:]}

# Se extraen las medias y la cobertura de las medidas que se van a mostrar
def load(base):
    ds=load_and_check(base)
    rows=[]
    for i,t in enumerate(ds['trios']):
        ref=t['pareja_familiar_referencia']; row={'familia':f'F{i+1:02d}','ref':ref}
        for metric,*_ in METRICS:
            ctx=f'{ref}/{metric}'
            n=integer(t,'n_enunciados_comunes_'+metric,ctx)
            expected=ds['metric_coverage'][i] if metric=='delta_rango_f0_st_sin_alertas' else (9 if i==3 else 10)
            insist(n==expected, f'{ctx}: cobertura inesperada')
            h=num(t,'media_hermanos_'+metric,ctx)
            c=num(t,'media_controles_'+metric,ctx)
            a=num(t,'media_control_a_'+metric,ctx); b=num(t,'media_control_b_'+metric,ctx)
            same(c,(a+b)/2,ctx+'/control_promedio')
            same(c-h,num(t,'contraste_favorable_'+metric,ctx),ctx+'/contraste')
            insist(min(h,c,a,b)>=0, f'{ctx}: distancia negativa')
            row[metric]={'hermanos':h,'control':c,'n':n,'total':9 if i==3 else 10,
                         'ventaja':c-h}
        rows.append(row)
    return ds, rows


# Se dibujan las comparaciones entre hermanos y controles por familia
def draw(rows, which):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ms=FIGS[which]
    # Cada panel utiliza las mismas familias y convenciones de comparacion
    # Los titulos y detalles metodologicos se dejan para el pie en LaTeX
    fig=plt.figure(figsize=(11.3,5.5),facecolor='white')
    axes=fig.subplots(1,2,gridspec_kw={
        'left':.125,'right':.955,'bottom':.185,'top':.865,'wspace':.22
    })
    snapshot=[]
    for ax,j,(metric,heading,xlabel,_) in zip(axes,range(2),ms):
        values=[q[metric] for q in rows]
        mx=max(max(r['hermanos'],r['control']) for r in values)
        limit=max(1.0, mx*1.23) if metric=='delta_rango_f0_st_sin_alertas' else (mx*1.23 if mx>1e-10 else 1.0)
        for i,(f,item) in enumerate(zip(rows,values)):
            y=5-i
            if i%2==0:ax.axhspan(y-.41,y+.41,color=P['alternate'],zorder=0)
            h=item['hermanos']; c=item['control']
            ax.plot([h,c],[y,y],color=P['grid'],linewidth=2.3,zorder=1)
            ax.scatter(h,y,s=90,facecolor=P['good'],edgecolor='white',linewidth=1.3,zorder=3)
            ax.scatter(c,y,s=90,facecolor='white',edgecolor=P['bad'],linewidth=1.9,marker='D',zorder=4)
            if metric=='delta_rango_f0_st_sin_alertas':
                ax.text(limit*.95,y+.23,f"{item['n']}/{item['total']}",
                        fontsize=8.5,color=P['muted'],ha='right',va='center')
        ax.set_yticks(list(range(5,-1,-1)))
        ax.set_yticklabels([f['familia'] for f in rows],fontsize=11,weight='bold')
        if j==1:ax.tick_params(axis='y',length=0,labelleft=False)
        ax.set_xlim(0,limit)
        ax.set_ylim(-.6,5.65)
        ax.set_xlabel(xlabel,fontsize=10.3,labelpad=12)
        ax.grid(axis='x',color=P['grid'],linewidth=.9)
        ax.set_axisbelow(True)
        ax.tick_params(axis='y',length=0)
        ax.set_title(heading,fontsize=11.7,fontweight='bold',color=P['muted'],pad=22)
        snapshot.append({'metrica':metric,'valores':values,'limite_x':limit})
    # El circulo identifica hermanos y el rombo el promedio de controles
    # El segmento permite apreciar la diferencia entre ambos resultados
    return fig, {'figura':which,'familias':6,'metricas':snapshot}


# Se validan las tablas y se generan figuras y manifiestos de procedencia
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=VISUAL.parent)
    p.add_argument('--modo',choices=('verificar','probar-render','generar'),default='verificar')
    p.add_argument('--run-id',default='')
    args=p.parse_args()
    if args.run_id and not re.fullmatch('[A-Za-z0-9_-]{1,64}',args.run_id):p.error('run-id inválido')
    try:ds,rows=load(args.base.resolve())
    except (OSError,ValueError,KeyError,AssertionError) as exc:p.error(f'Validación falló: {exc}')
    print('=== IE-0499 · PASO 8.4 ACÚSTICA REDUCIDO ===')
    print('Modo:',args.modo)
    print('Seis familias; cuatro métricas; 35/59 unidades de F0 filtrada.')
    if args.modo=='verificar':
        print('VALIDACIÓN CORRECTA: solo lectura de CSV del Paso 7.')
        return
    apply()
    import matplotlib.pyplot as plt
    if args.modo=='probar-render':
        for name in FIGS:
            fig,_=draw(rows,name)
            try:
                with io.BytesIO() as b:
                    fig.savefig(b,dpi=100,format='png',facecolor='white',bbox_inches=None)
                    insist(b.tell()>12000,f'{name}: imagen insuficiente')
            finally:plt.close(fig)
            print('RENDER OK:',name)
        print('Ningún archivo exportado.')
        return
    if not args.run_id:p.error('Indique --run-id único para generar')
    final=args.base.resolve()/'visualizations'/'exports'/args.run_id
    final.parent.mkdir(parents=True,exist_ok=True)
    try:final.mkdir(mode=0o700,exist_ok=False)
    except FileExistsError:p.error('La salida ya existe y no se sobrescribe')
    try:
        entries=[]
        for name in FIGS:
            fig,spec=draw(rows,name)
            try:
                files={}
                for ext in ['pdf','svg','png']:
                    f=final/f'{name}_{name.lower()}_acustica.{ext}'
                    fig.savefig(f,format=ext,dpi=300,facecolor='white',bbox_inches=None)
                    if ext=='png':
                        from PIL import Image
                        with Image.open(f) as img:img.verify()
                    files[f.name]=sha(f)
            finally:plt.close(fig)
            entries.append({'id':name,'archivos':files,'data_snapshot':spec})
            print('EXPORTADO:',name,'PDF + SVG + PNG')
        manifest={'proyecto':'IE0499 Paso 8.4 acústica — alcance reducido','version':'0.1.1',
                  'run_id':args.run_id,'fecha_utc':datetime.now(timezone.utc).isoformat(),
                  'datos_origen':ds['source_dir'],'input_sha256':ds['source_hashes'],
                  'codigo_sha256':{x.name:sha(x) for x in ROOT.glob('*.py')},
                  'figuras':entries,'notas':'Descriptivo. Medidas correlacionadas, controles prefijados, filtros de calidad y muestra pequeña; no hay inferencia causal.'}
        with (final/'manifest_8_4.json').open('x',encoding='utf-8') as f:
            json.dump(manifest,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
        print('SALIDA:',final)
        print('MANIFIESTO:',final/'manifest_8_4.json')
    except BaseException:
        shutil.rmtree(final)
        raise

if __name__=='__main__':main()
