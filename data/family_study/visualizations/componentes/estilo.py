# Configuracion visual compartida por las figuras
#
# Se definen colores, tipografia y parametros graficos en un solo lugar
# Estos ajustes no modifican las medidas que vienen de los CSV
# Se priorizan fuentes serif compatibles con el documento en LaTeX
# STIXGeneral se utiliza si las otras fuentes no estan disponibles
"""Define las fuentes, los colores y el formato compartido por las figuras"""
from __future__ import annotations
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.ft2font import FT2Font


# Se busca una fuente que incluya todos los simbolos de las figuras
def fuente_academica():
    """Selecciona una fuente serif que permita mostrar todos los caracteres

    Primero se buscan fuentes compatibles con el informe en LaTeX
    Si no estan disponibles se utiliza STIXGeneral, incluida con Matplotlib
    """
    # Se comprueba la disponibilidad de caracteres especiales en las fuentes
    caracteres = "·ÓÚáéíñóú—−■◆○●"
    for familia in ("CMU Serif", "Latin Modern Roman", "STIXGeneral"):
        try:
            regular = findfont(FontProperties(family=familia), fallback_to_default=False)
            negrita = findfont(FontProperties(family=familia, weight="bold"), fallback_to_default=False)
            for ruta in (regular, negrita):
                cobertura = FT2Font(ruta).get_charmap()
                if not all(ord(c) in cobertura for c in caracteres):
                    break
            else:
                return familia
        except (OSError, RuntimeError, ValueError):
            continue
    # STIXGeneral se incluye normalmente con Matplotlib
    return "STIXGeneral"

P = {
    'ink':'#213547', 'muted':'#607182', 'background':'#FFFFFF',
    'blue':'#245F9A', 'blue_light':'#DBE9F4',
    'ambiguous':'#62798C', 'control':'#8B98A6', 'control_b':'#C3CDD5',
    'good':'#167A78', 'good_light':'#DBF0EB',
    'bad':'#A56220', 'bad_light':'#F6E7D5',
    'grid':'#DEE6EC', 'neutral':'#EEF2F5', 'white':'#FFFFFF',
    'alternate':'#F7F9FB',
}


# Se configuran las opciones de Matplotlib antes de dibujar
def apply():
    mpl.use('Agg',force=True)
    familia = fuente_academica()
    mpl.rcParams.update({
        'font.family':familia, 'font.size':10,
        'mathtext.fontset':'stix' if familia == 'STIXGeneral' else 'cm',
        'font.weight':'normal', 'axes.labelcolor':P['ink'],
        'text.color':P['ink'], 'axes.edgecolor':P['grid'],
        'xtick.color':P['muted'], 'ytick.color':P['ink'],
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.spines.left':False, 'axes.titleweight':'bold',
        'savefig.facecolor':'white','figure.facecolor':'white',
        'pdf.fonttype':42, 'ps.fonttype':42,'svg.fonttype':'none',
        'axes.unicode_minus':True,
    })


# Se colocan encabezados cuando una plantilla los necesita
def base_title(fig, eyebrow, title, sub):
    fig.text(.055,.954,eyebrow.upper(),fontsize=9.2,color=P['good'],weight='bold',ha='left')
    fig.text(.055,.895,title,fontsize=17.2,color=P['ink'],weight='bold',ha='left')
    fig.text(.055,.849,sub,fontsize=10.1,color=P['muted'],ha='left')
    fig.add_artist(Rectangle((.055,.822),.89,.003,transform=fig.transFigure,facecolor=P['grid'],edgecolor='none'))


# Se agrega una anotacion breve dentro de la figura
def note(fig,text,y=.065,fontsize=9):
    fig.text(.055,y,text,fontsize=fontsize,color=P['muted'],va='top',ha='left')


# Se muestra el signo de los contrastes junto con su valor
def format_signed(v, nd=3):
    return f'{v:+.{nd}f}'.replace('-', '−')
