# Representaciones graficas V05, V06 y V15 del informe
#
# Las funciones reciben resultados calculados y previamente validados
# Solo organizan su representacion grafica sin recalcular similitudes
# V05 muestra contrastes y V06 compara los hermanos con sus controles
# V15 resume la direccion de las once medidas sin combinarlas en un valor
"""Dibuja las figuras V05, V06 y V15 con resultados ya calculados

Los graficos muestran contrastes y direcciones por medida sin construir una
puntuacion conjunta de semejanza vocal"""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from .estilo import P, base_title, note, format_signed
from .validacion_resumen import REFERENCES, METRICS, num, integer, sign

LABELS = {ref: f'F{i:02d}' for i, ref in enumerate(REFERENCES, 1)}
FIGURES = {
 'V05': {'title':'Contrastes ECAPA y sensibilidad LOO', 'inputs':['contrastes_integrados_trios.csv','robustez_integrada.csv']},
 'V06': {'title':'Similitud ECAPA: hermanos y controles', 'inputs':['contrastes_integrados_trios.csv']},
 'V15': {'title':'Concordancia descriptiva de once medidas', 'inputs':['contrastes_integrados_trios.csv','robustez_integrada.csv']},
}


# Se recuperan las familias en el mismo orden utilizado en el estudio
def _names():
    return [LABELS[r] for r in REFERENCES]


# Se alternan fondos suaves para facilitar la lectura entre filas
def _bands(ax, y, width=None):
    for i, yy in enumerate(y):
        if i % 2 == 0:
            ax.axhspan(yy - .47, yy + .47, color=P['alternate'], zorder=0)


# Se muestra el contraste ECAPA y su cambio al excluir cada enunciado
# Los extremos representan una comprobacion de sensibilidad, no confianza estadistica
def v05(dataset):
    refs=REFERENCES; trios=dataset['trios']; robust=dataset['robust']
    vals=[num(t,'contraste_favorable_similitud_coseno',r) for t,r in zip(trios,refs)]
    low=[num(robust[(r,'similitud_coseno')],'loo_min',r) for r in refs]
    high=[num(robust[(r,'similitud_coseno')],'loo_max',r) for r in refs]
    n=[integer(t,'n_enunciados_diseno',r) for t,r in zip(trios,refs)]
    unstable=[low[i] <= 0 <= high[i] for i in range(6)]
    # Si el rango atraviesa cero, el sentido del contraste depende de la frase excluida
    # El titulo y las explicaciones principales quedan fuera del dibujo
    # Se mantienen los mismos valores y escalas de los datos
    fig=plt.figure(figsize=(11.3,5.6),facecolor='white')
    ax=fig.add_axes((.135,.185,.575,.725))
    right=fig.add_axes((.748,.185,.210,.725),sharey=ax)
    y=np.arange(6)[::-1]
    ax.set_xlim(-.14,.39);ax.set_ylim(-.65,5.65)
    _bands(ax,y);_bands(right,y)
    ax.axvline(0,color=P['ink'],lw=1.05,zorder=2)
    ax.grid(axis='x',color=P['grid'],linewidth=.7,zorder=0)
    for i, yy in enumerate(y):
        c=P['ambiguous'] if unstable[i] else P['good'] if vals[i]>0 else P['bad']
        ax.plot([low[i],high[i]],[yy,yy],color=c,lw=3.4,solid_capstyle='round',zorder=3)
        ax.plot([low[i],low[i]],[yy-.12,yy+.12],color=c,lw=1.4,zorder=3)
        ax.plot([high[i],high[i]],[yy-.12,yy+.12],color=c,lw=1.4,zorder=3)
        ax.scatter([vals[i]],[yy],s=100,facecolor='white' if unstable[i] else c,
                   edgecolor=c if unstable[i] else 'white',lw=2.0 if unstable[i] else 1.2,zorder=4)
        # Se destaca el cruce de cero cuando el signo no es estable
        if unstable[i]:
            ax.plot([0,0],[yy-.19, yy+.19],color=P['ambiguous'],lw=1.3,zorder=3)
    ax.set_yticks(y,_names(),weight='bold',fontsize=11)
    ax.tick_params(axis='y',length=0,pad=11)
    ax.set_xticks([-.1,0,.1,.2,.3])
    ax.set_xlabel('Contraste de similitud coseno (adimensional)',fontsize=10,labelpad=12)
    right.set_xlim(0,1);right.set_ylim(-.65,5.65);right.axis('off')
    fig.text(.748,.931,'CONTRASTE',fontsize=8.5,color=P['muted'],weight='bold')
    fig.text(.865,.931,'FRASES',fontsize=8.5,color=P['muted'],weight='bold')
    for i,yy in enumerate(y):
        right.text(.015,yy,format_signed(vals[i],4),fontsize=10.3,va='center',weight='bold',color=P['ink'])
        right.text(.59,yy,f'{n[i]}/{n[i]}',fontsize=10,va='center',color=P['muted'])
    # La interpretacion del rango y del signo se indica en el pie de figura
    # Los puntos huecos identifican las familias con un signo inestable
    return fig, dict(id='V05',rows=6,metric='similitud_coseno',
                     values=[round(x,9) for x in vals],
                     loo_bounds=[[round(a,9),round(b,9)] for a,b in zip(low,high)],
                     range_crosses_zero=unstable)


# Se comparan los hermanos con cada una de las dos parejas de control
def v06(dataset):
    refs=REFERENCES; trios=dataset['trios']
    h=[num(t,'media_hermanos_similitud_coseno',r) for t,r in zip(trios,refs)]
    ca=[num(t,'media_control_a_similitud_coseno',r) for t,r in zip(trios,refs)]
    cb=[num(t,'media_control_b_similitud_coseno',r) for t,r in zip(trios,refs)]
    n=[integer(t,'n_enunciados_diseno',r) for t,r in zip(trios,refs)]
    # Se deja el titulo explicativo para el pie de figura del informe
    fig=plt.figure(figsize=(11.3,5.6),facecolor='white')
    ax=fig.add_axes((.135,.185,.585,.725))
    right=fig.add_axes((.760,.185,.196,.725),sharey=ax)
    y=np.arange(6)[::-1]
    ax.set_xlim(0,.72);ax.set_ylim(-.6,5.6)
    _bands(ax,y);_bands(right,y)
    for i,yy in enumerate(y):
        ax.scatter([h[i]],[yy],s=109,marker='o',facecolor=P['blue'],edgecolor=P['ink'],lw=.75,zorder=5)
        ax.scatter([ca[i]],[yy+.18],s=61,marker='s',facecolor=P['white'],edgecolor=P['control'],lw=1.8,zorder=4)
        ax.scatter([cb[i]],[yy-.18],s=68,marker='^',facecolor=P['white'],edgecolor=P['control'],lw=1.8,zorder=4)
    ax.grid(axis='x',color=P['grid'],lw=.7,zorder=0)
    ax.set_yticks(y,_names(),weight='bold',fontsize=11);ax.tick_params(axis='y',length=0,pad=11)
    ax.set_xticks(np.arange(0,.71,.1));ax.set_xlabel('Similitud coseno (adimensional)',fontsize=10,labelpad=12)
    right.set_xlim(0,1);right.set_ylim(-.6,5.6);right.axis('off')
    fig.text(.760,.931,'H / CA / CB',fontsize=8.5,color=P['muted'],weight='bold')
    for i,yy in enumerate(y):
        right.text(.01,yy+.19,f'H   {h[i]:.4f}',fontsize=9.2,color=P['blue'],weight='bold',va='center')
        right.text(.01,yy,f'CA {ca[i]:.4f}',fontsize=9.2,color=P['muted'],va='center')
        right.text(.01,yy-.19,f'CB {cb[i]:.4f}',fontsize=9.2,color=P['muted'],va='center')
    # Circulo para hermanos, cuadrado para el primer control y triangulo para el segundo
    # El pie de figura explica las tres marcas utilizadas
    return fig,dict(id='V06',rows=6,metric='similitud_coseno',values={'H':h,'CA':ca,'CB':cb},coverage=n)


SHORT={
 'similitud_coseno':'ECAPA',
 'delta_f0_st':'F0\nmediana',
 'delta_rango_f0_st':'Variab.\nF0',
 'delta_rango_f0_st_sin_alertas':'Variab.\nfiltrada',
 'delta_duracion_total_s':'Duración\nsegundos',
 'delta_duracion_total_pct':'Duración\nrelativa',
 'delta_habla_estimada_s':'Habla\nsegundos',
 'delta_habla_estimada_pct':'Habla\nrelativa',
 'delta_fraccion_habla_pp':'Fracción\nvoz',
 'delta_velocidad_total_pal_s':'Velocidad\ntotal',
 'delta_velocidad_articulacion_pal_s':'Velocidad\narticulac.',
}
GROUPS=[('ECAPA',0,0),('TONO',1,3),('DURACIÓN',4,5),('VOZ ESTIMADA',6,8),('VELOCIDAD',9,10)]


# Se muestra a que grupo favorece cada medida en las seis familias
# El relleno indica si la direccion cambia al excluir una frase
def v15(dataset):
    """Representa el sentido de once medidas y su estabilidad por familia

    El circulo representa resultados favorables a los hermanos y el rombo a los controles
    El relleno muestra si el signo se mantiene al excluir cada enunciado
    Solo el contraste ECAPA se presenta con un valor numerico en la matriz
    """
    refs=REFERENCES; trios=dataset['trios']; rob=dataset['robust']
    # Se organiza la matriz en formato ancho para mantener las etiquetas legibles
    # Las unidades distintas impiden comparar directamente la magnitud de las medidas
    fig=plt.figure(figsize=(13.6,6.25),facecolor='white')
    ax=fig.add_axes((.090,.245,.858,.655))
    nc=len(METRICS); ax.set_xlim(-.55,nc-.47); ax.set_ylim(5.55,-1.76)
    ax.set_xticks([]);ax.set_yticks(range(6),_names(),fontsize=11,weight='bold')
    ax.tick_params(axis='y',length=0,pad=12);ax.set_frame_on(False)
    # Se alterna el fondo de las filas para seguir cada familia
    for i in range(6):
        if i%2==0:
            ax.add_patch(patches.Rectangle((-.50,i-.46),nc-.01,.92,
                                           facecolor=P['alternate'],edgecolor='none',zorder=0))
    # Solo se anota el valor numerico del contraste ECAPA
    ax.add_patch(patches.Rectangle((-.47,-.48),.94,6.02,facecolor=P['blue_light'],alpha=.35,
                                   edgecolor='none',zorder=0.4))
    # Se agrupan los encabezados para distinguir los tipos de medidas
    for title,begin,end in GROUPS:
        xstart=begin-.43; xend=end+.43
        ax.add_patch(patches.FancyBboxPatch((xstart,-1.72),xend-xstart,.37,
                     boxstyle='round,pad=0.0,rounding_size=.055',
                     facecolor=P['neutral'],edgecolor='none',zorder=1))
        ax.text((begin+end)/2,-1.535,title,ha='center',va='center',fontsize=9.1,
                weight='bold',color=P['muted'],zorder=2)
    for j,m in enumerate(METRICS):
        ax.text(j,-.96,SHORT[m],ha='center',va='center',fontsize=8.55,
                color=P['ink'],linespacing=1.32)
    # Se utilizan separadores discretos entre los grupos de medidas
    for _,_,end in GROUPS[:-1]:
        ax.plot([end+.5,end+.5],[-.40,5.48],color=P['grid'],lw=1.3,zorder=1)
    arr=[]; stable_matrix=[]; raw_contrasts=[]
    for i,ref in enumerate(refs):
        signs=[];stable_row=[];contrasts=[]
        for j,m in enumerate(METRICS):
            v=num(trios[i],'contraste_favorable_'+m,ref)
            r=rob[(ref,m)]
            sg=sign(v); stable=integer(r,'loo_signo_estable',ref+'/'+m)==1
            signs.append(sg);stable_row.append(stable);contrasts.append(v)
            if sg==1:
                ax.scatter([j],[i-.06],s=137,marker='o',
                    facecolor=P['good'] if stable else 'white',edgecolor=P['good'],
                    lw=1.9 if not stable else .5,zorder=4)
            elif sg==-1:
                ax.scatter([j],[i-.06],s=122,marker='D',
                    facecolor=P['bad'] if stable else 'white',edgecolor=P['bad'],
                    lw=1.9 if not stable else .5,zorder=4)
            else:
                ax.scatter([j],[i-.06],s=125,marker='s',
                    facecolor='white',edgecolor=P['control'],lw=1.7,zorder=4)
            # No se muestran numeros de contraste en medidas con unidades diferentes
            if j==0:
                ax.text(j,i+.28,format_signed(v,4),ha='center',va='center',
                    fontsize=8.4,color=P['ink'],zorder=5)
            # Se indica cuantas frases superaron el filtrado de calidad de F0
            if m.endswith('_sin_alertas'):
                cover=integer(r,'n_enunciados_comunes',ref)
                total=integer(trios[i],'n_enunciados_diseno',ref)
                ax.text(j,i+.28,f'{cover}/{total}',ha='center',va='center',
                    fontsize=8.1,color=P['muted'],zorder=5)
        arr.append(signs);stable_matrix.append(stable_row);raw_contrasts.append(contrasts)
    # La leyenda distingue la direccion y la estabilidad mediante formas y relleno
    leg_y=.138
    fig.text(.085,leg_y,'●',fontsize=15,color=P['good'],va='center')
    fig.text(.109,leg_y,'Favorece hermanos',fontsize=9.5,color=P['ink'],va='center')
    fig.text(.315,leg_y,'◆',fontsize=13,color=P['bad'],va='center')
    fig.text(.337,leg_y,'Favorece controles',fontsize=9.5,color=P['ink'],va='center')
    fig.text(.54,leg_y,'●',fontsize=13,color=P['ink'],va='center')
    fig.text(.558,leg_y,'Relleno: LOO estable',fontsize=9.4,color=P['ink'],va='center')
    fig.text(.758,leg_y,'○',fontsize=14,color=P['ambiguous'],va='center')
    fig.text(.778,leg_y,'Hueco: signo variable',fontsize=9.2,color=P['ink'],va='center')
    # La cobertura y las limitaciones se explican en el pie del informe
    return fig,dict(id='V15',rows=6,columns=nc,metric_order=list(METRICS),
                    signs=arr,stable=stable_matrix,
                    ecapa_contrast=[round(row[0],9) for row in raw_contrasts],
                    groups=[dict(label=a,start=b,end=c) for a,b,c in GROUPS])

RENDERERS={'V05':v05,'V06':v06,'V15':v15}
