# Representaciones graficas de las comparaciones ECAPA
#
# V01 muestra distribuciones y V02 compara los resultados por pareja
# V03 presenta los contrastes por enunciado y V04 indica
# con que frecuencia las comparaciones favorecen a los hermanos
"""Contiene las funciones de dibujo de V01, V02, V03 y V04 a partir de tablas ECAPA"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from componentes.estilo import P, base_title, note, format_signed
from componentes.validacion_resumen import REFERENCES, num, integer

LABELS={r:f'F{i:02d}' for i,r in enumerate(REFERENCES,1)}
GROUPS=[('misma_persona','Misma persona',P['blue']),('hermanos','Hermanos',P['good']),('control','Controles',P['bad'])]
FIGURES83={
 'V01':{'title':'Distribuciones de similitud ECAPA por nivel de análisis',
        'sources':['ecapa_comparaciones.csv','ecapa_resumen_enunciados.csv']},
 'V02':{'title':'Similitud media de las 33 parejas de comparación',
        'sources':['ecapa_resumen_parejas.csv','ecapa_resumen_enunciados.csv','protocolo_comparaciones.csv']},
 'V03':{'title':'Variación del contraste ECAPA entre enunciados',
        'sources':['contrastes_integrados_enunciados.csv','contrastes_integrados_trios.csv']},
 'V04':{'title':'Consistencia descriptiva por enunciado',
        'sources':['contrastes_integrados_trios.csv','contrastes_integrados_enunciados.csv']},
}


# Se muestran por separado las comparaciones entre tomas y sus promedios
# Las distintas tomas de una persona no se tratan como participantes nuevos
def v01(ds):
    """Muestra las puntuaciones de cada toma y las medias por enunciado en paneles separados"""
    # Se crean dos paneles con una escala de similitud compartida
    fig=plt.figure(figsize=(11.3,5.5),facecolor='white')
    raw_values=np.array([num(r,'similitud_coseno','V01/raw') for r in ds['raw']])
    # Los dos paneles utilizan los mismos limites horizontales
    lo=max(-1, float(np.floor((min(0,raw_values.min())-.04)*10)/10))
    hi=min(1, float(np.ceil((max(0,raw_values.max())+.04)*10)/10))
    if hi-lo<.4:
        lo=max(-1,lo-.1);hi=min(1,hi+.1)
    axes=[fig.add_axes((.180,.185,.345,.680)),fig.add_axes((.590,.185,.345,.680))]
    summaries=[]
    for k,(ax, table, caption) in enumerate(zip(axes,[ds['raw'],ds['phrases']],
                ['PUNTUACIONES ENTRE TOMAS','MEDIAS POR ENUNCIADO'])):
        ax.set_ylim(-.58,2.58); ax.set_xlim(lo,hi)
        ax.set_yticks([2,1,0],['Misma persona','Hermanos','Controles'] if k==0 else ['', '', ''],fontsize=10.2)
        ax.tick_params(axis='y',length=0,pad=8)
        ax.set_xlabel('Similitud coseno (adimensional)',fontsize=9.5,labelpad=10)
        ax.grid(axis='x',color=P['grid'],lw=.65,zorder=0)
        ax.set_axisbelow(True)
        ax.text(0,1.09,caption,transform=ax.transAxes,color=P['muted'],fontsize=10.0,weight='bold')
        for i,(group,_,color) in enumerate(GROUPS):
            y=2-i
            vals=np.array([num(r,'similitud_coseno' if k==0 else 'media_coseno',group)
                           for r in table if r['grupo']==group]); vals.sort()
            # Los puntos se desplazan ligeramente para evitar que se oculten entre si
            jitter=.13*np.sin(np.arange(len(vals))*2.399963229728653)
            ax.scatter(vals,y+jitter,s=12,alpha=.22,color=color,linewidths=0,zorder=2)
            q10,q25,med,q75,q90=np.percentile(vals,[10,25,50,75,90])
            ax.plot([q10,q90],[y,y],color=color,lw=1.6,zorder=3)
            ax.plot([q25,q75],[y,y],color=color,lw=7,solid_capstyle='round',zorder=4)
            ax.scatter([med],[y],s=70,marker='o',edgecolor='white',lw=1.3,color=color,zorder=5)
            summaries.append({'nivel':'tomas' if k==0 else 'enunciados','grupo':group,
                              'n':len(vals),'mediana':float(med), 'q25':float(q25),
                              'q75':float(q75),'p10':float(q10),'p90':float(q90)})
    # El pie de figura explica las bandas y el numero de observaciones
    # Las observaciones de varias tomas pueden compartir participantes
    return fig,dict(id='V01',rows={'raw':len(ds['raw']),'phrases':len(ds['phrases'])},
                    axis_bounds=[lo,hi],summaries=summaries)


# Se agrupan las parejas segun el tipo de comparacion
def v02(ds):
    """Organiza los resultados de las 33 parejas en grupos verticales

    Los encabezados ocupan filas propias para evitar confundirlos con los resultados
    """
    pairs=ds['pairs']; phrases=ds['by_pair']
    seq=[]
    own=sorted((r for r in pairs if r['grupo']=='misma_persona'),key=lambda r:r['participante_a'])
    siblings=sorted((r for r in pairs if r['grupo']=='hermanos'),key=lambda r:REFERENCES.index(r['pareja_familiar_referencia']))
    controls=sorted((r for r in pairs if r['grupo']=='control'),
      key=lambda r:(REFERENCES.index(r['pareja_familiar_referencia']),r['participante_a']))
    for group, rows in [('misma_persona',own),('hermanos',siblings),('control',controls)]:
        for r in rows:
            if group=='misma_persona': label=r['participante_a'].replace('_',' · ')
            elif group=='hermanos': label=LABELS[r['pareja_familiar_referencia']]+' · H'
            else:
                ref=r['pareja_familiar_referencia']
                pr=next(x for x in ds['baseline']['trios'] if x['pareja_familiar_referencia']==ref)
                kind='CA' if r['participante_a']==pr['participante_hermano_a'] else 'CB'
                label=LABELS[ref]+' · '+kind
            values=np.array([num(p,'media_coseno','V02') for p in phrases[r['id_comparacion']]])
            seq.append((group,label,float(np.mean(values)),float(values.min()),
                        float(values.max()),len(values),r['id_comparacion']))
    assert len(seq)==33
    fig=plt.figure(figsize=(12.9,14.1),facecolor='white')
    base_title(fig,'Resumen individual · V02','Las 33 comparaciones del protocolo',
               'Media ECAPA por pareja; rango observado entre enunciados')
    ax=fig.add_axes((.232,.191,.685,.589))
    minimum=min(x[3] for x in seq); maximum=max(x[4] for x in seq)
    low=max(-1,float(np.floor((min(minimum,0)-.04)*10)/10))
    high=min(1,float(np.ceil((maximum+.04)*10)/10))
    ax.set_xlim(low,high);ax.set_ylim(-.8,35.8)
    ax.grid(axis='x',color=P['grid'],lw=.7,zorder=0);ax.set_axisbelow(True)
    ax.axvline(0,color=P['control'],lw=.7,zorder=1)

    headers={35:('MISMA PERSONA · 15',P['blue']),
             19:('HERMANOS · 6',P['good']),
             12:('CONTROLES · 12',P['bad'])}
    for yy,(name,color) in headers.items():
        ax.axhspan(yy-.48,yy+.48,color=P['alternate'],zorder=0)
        ax.text(.02,yy,name,transform=ax.get_yaxis_transform(),ha='left',va='center',
                color=color,fontsize=9.1,weight='bold',zorder=5)

    ypos=list(range(34,19,-1))+list(range(18,12,-1))+list(range(11,-1,-1))
    assert len(ypos)==len(seq)==33
    for i,((group,label,avg,mn,mx,n,_id),yy) in enumerate(zip(seq,ypos)):
        color=next(c for g,_,c in GROUPS if group==g)
        if i%2==0: ax.axhspan(yy-.46,yy+.46,color=P['alternate'],zorder=0)
        ax.plot([mn,mx],[yy,yy],color=color,alpha=.70,lw=2.4,zorder=3)
        ax.scatter([avg],[yy],color=color,s=51,edgecolor='white',lw=.6,zorder=4)
        ax.text(1.015,yy,f'{avg:.3f} ({n})',transform=ax.get_yaxis_transform(),
                va='center',ha='left',fontsize=7.45,color=P['ink'],clip_on=False)
    ax.set_yticks(ypos,[item[1] for item in seq],fontsize=8.2)
    ax.tick_params(axis='y',length=0,pad=8)
    ax.set_xlabel('Similitud coseno (adimensional)',fontsize=10,labelpad=12)
    fig.text(.232,.117,'Punto: media de enunciados    Segmento: mínimo y máximo descriptivos    (n): frases disponibles',
             fontsize=8.7,color=P['ink'])
    note(fig,'Se conserva el orden del protocolo. Los rangos NO son intervalos de confianza.',
         y=.070,fontsize=9.0)
    return fig,dict(id='V02',rows=33,axis_bounds=[low,high],
                    groups={'misma_persona':15,'hermanos':6,'control':12},
                    means={x[6]:round(x[2],9) for x in seq})

# Se muestra un contraste por cada enunciado disponible en la familia
def v03(ds):
    """Dibuja los contrastes de similitud para los enunciados de las seis familias"""
    entries=defaultdict(dict)
    for f in ds['baseline']['frases']:
        entries[f['pareja_familiar_referencia']][int(f['enunciado'][1:])]=num(
            f,'contraste_favorable_similitud_coseno',f['pareja_familiar_referencia'])
    vals=[v for d in entries.values() for v in d.values()]
    low=float(np.floor((min(vals+[0])-.04)*20)/20)
    high=float(np.ceil((max(vals+[0])+.04)*20)/20)
    fig=plt.figure(figsize=(13.5,10.7),facecolor='white')
    base_title(fig,'Variación entre enunciados · V03','Contraste ECAPA en cada enunciado',
               'Comparación de hermanos frente al promedio de sus controles prefijados')
    axes=[];snap=[]
    for idx,ref in enumerate(REFERENCES):
        row,col=divmod(idx,2)
        ax=fig.add_axes((.085+col*.47,.60-row*.21,.394,.153))
        d=entries[ref]; ys=np.array([d.get(e,np.nan) for e in range(1,11)])
        n=len(d)
        ax.axhline(0,color=P['ink'],lw=1,zorder=2)
        ax.grid(axis='y',color=P['grid'],lw=.7,zorder=0)
        # Los enunciados se muestran como categorias y no como una serie temporal
        for k,v in enumerate(ys):
            if np.isfinite(v):
                c=P['good'] if v>0 else P['bad'] if v<0 else P['control']
                ax.scatter([k+1],[v],color=c,s=46,edgecolor='white',lw=.55,zorder=4)
        mean=np.nanmean(ys)
        ax.axhline(mean,color=P['blue'],ls='--',lw=1.3,zorder=3)
        ax.set_ylim(low,high);ax.set_xlim(.5,10.5)
        ax.set_xticks(range(1,11),[f'E{j:02d}' for j in range(1,11)],fontsize=7.7)
        ax.tick_params(axis='x',length=0,pad=5)
        ax.tick_params(axis='y',labelsize=8)
        ax.set_title(f'F{idx+1:02d}  ·  {n} frases     media {format_signed(mean,3)}',
                     color=P['ink'],fontsize=10.6,loc='left',weight='bold',pad=9)
        if col==0: ax.set_ylabel('Contraste',fontsize=9,labelpad=8)
        if row==2:ax.set_xlabel('Enunciado',fontsize=9,labelpad=8)
        axes.append(ax)
        snap.append(dict(ref=ref,n=n,mean=float(mean),by_phrase={f'E{e:02d}':float(v) for e,v in d.items()}))
    fig.text(.085,.132,'● Valor por frase: turquesa favorable a hermanos / ocre favorable a controles      - - Media del conjunto',
             color=P['ink'],fontsize=8.75)
    note(fig,'Enunciados discretos, no una evolución temporal. Misma escala vertical; F04 carece de E08 (no equivale a cero).',
         y=.096,fontsize=8.9)
    return fig,dict(id='V03',rows=sum(len(v) for v in entries.values()),
                    y_bounds=[low,high],series=snap)


# Se cuenta cuantas frases favorecen a los hermanos frente a los controles
def v04(ds):
    """Cuenta los enunciados favorables a los hermanos y los compara con los controles"""
    ts=ds['baseline']['trios']
    fig=plt.figure(figsize=(12.5,7.9),facecolor='white')
    base_title(fig,'Consistencia descriptiva · V04','En cuántas frases favorece a los hermanos',
               'Contrastes ECAPA positivos, negativos o nulos para cada conjunto')
    ax=fig.add_axes((.155,.27,.625,.485))
    ax.set_xlim(0,10); ax.set_ylim(-.65,5.65)
    ax.set_yticks(list(range(6))[::-1],[f'F{i:02d}' for i in range(1,7)],fontsize=11,weight='bold')
    ax.tick_params(axis='y',length=0,pad=12)
    ax.set_xticks(range(11)); ax.grid(axis='x',color=P['grid'],lw=.7,zorder=0)
    ax.set_xlabel('Número de enunciados',fontsize=10,labelpad=12)
    results=[]
    for i,t in enumerate(ts):
        n=integer(t,'n_enunciados_comunes_similitud_coseno',f'V04/F{i+1:02d}')
        pos=integer(t,'n_enunciados_favorables_similitud_coseno',f'V04/F{i+1:02d}')
        neg=integer(t,'n_enunciados_contrarios_similitud_coseno',f'V04/F{i+1:02d}')
        tie=integer(t,'n_enunciados_empate_similitud_coseno',f'V04/F{i+1:02d}')
        both=integer(t,'n_enunciados_gana_ambos_similitud_coseno',f'V04/F{i+1:02d}')
        assert pos+neg+tie==n
        y=5-i
        if i%2==0:ax.axhspan(y-.46,y+.46,color=P['alternate'],zorder=0)
        for start,count,color in ((0,pos,P['good']),(pos,neg,P['bad']),(pos+neg,tie,P['control'])):
            if count:
                ax.barh(y,count,left=start,height=.49,color=color,edgecolor='white',lw=1.1,zorder=3)
                if count>=2:
                    ax.text(start+count/2,y,str(count),ha='center',va='center',color='white',fontsize=10,weight='bold',zorder=4)
        results.append(dict(ref=t['pareja_familiar_referencia'],n=n,positivas=pos,negativas=neg,empates=tie,supera_ambos=both))
        ax.text(1.11,y,f'{both}/{n}',transform=ax.get_yaxis_transform(),va='center',ha='center',fontsize=11.2,
                 color=P['ink'],weight='bold',clip_on=False)
    fig.text(.815,.762,'SUPERA AMBOS',fontsize=9.4,color=P['muted'],weight='bold')
    fig.text(.817,.738,'CONTROLES',fontsize=9.4,color=P['muted'],weight='bold')
    fig.text(.080,.153,'■',fontsize=12,color=P['good'])
    fig.text(.100,.153,'Favorece hermanos',fontsize=9.4,color=P['ink'])
    fig.text(.270,.153,'■',fontsize=12,color=P['bad'])
    fig.text(.290,.153,'Favorece controles',fontsize=9.4,color=P['ink'])
    fig.text(.460,.153,'■',fontsize=12,color=P['control'])
    fig.text(.480,.153,'Empate numérico',fontsize=9.4,color=P['ink'])
    note(fig,'Superar el promedio NO implica superar ambos controles. F04 tiene 9 frases; las otras familias, 10.',
         y=.103,fontsize=9)
    return fig,dict(id='V04',rows=6,counts=results)

RENDERERS83={'V01':v01,'V02':v02,'V03':v03,'V04':v04}
