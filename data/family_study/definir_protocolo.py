# DEFINICION DE LAS COMPARACIONES DEL ESTUDIO
#
# Este programa organiza las comparaciones que vamos a realizar con las voces
# Comprueba los participantes y las grabaciones disponibles, identifica las
# parejas de hermanos y asigna una persona de control a cada pareja
# Todas las comparaciones de un mismo grupo usan las mismas frases
# Al final guarda una tabla con las parejas y las tomas que se compararan
# Aqui no se calcula ninguna similitud ni se modifica el audio
#
# Ejecutar desde la raiz del repositorio: python data/family_study/definir_protocolo.py
from pathlib import Path
from collections import defaultdict, Counter
import csv
import re

base = Path("data/family_study")

# Usamos la codificacion de cada CSV para leer los datos sin alterar sus caracteres
# Los participantes usan UTF-8 y las relaciones familiares usan cp1252
def leer(nombre, codificacion):
    with (base / "metadata" / nombre).open(
        encoding=codificacion, newline=""
    ) as f:
        return list(csv.DictReader(f))

personas = {
    r["participant_id"]: r
    for r in leer("participants.csv", "utf-8-sig")
}

relaciones = {
    frozenset((r["participant_1"], r["participant_2"])): r
    for r in leer("relationships.csv", "cp1252")
}

assert len(personas) == 15, "Cantidad inesperada de participantes"

# Cada fila define dos hermanos, una persona de control y el tipo de parentesco
# Los controles se eligieron antes de calcular similitudes y no se cambian segun los resultados
trios = [
    ("F01_P01", "F01_P02", "F02_P07", "twins"),
    ("F02_P04", "F02_P06", "F03_P08", "siblings"),
    ("F03_P08", "F03_P09", "F02_P07", "siblings"),
    ("F04_P10", "F04_P11", "F02_P03", "twins"),
    ("F05_P12", "F05_P13", "F02_P05", "siblings"),
    ("F06_P14", "F06_P15", "F03_P09", "twins"),
]

# Contamos las grabaciones procesadas para detectar si falta o sobra algun WAV
archivos = sorted((base / "processed").glob("*.wav"))
assert len(archivos) == 298, "Cantidad inesperada de WAV"

muestras = defaultdict(set)
patron = re.compile(
    r"^(F\d{2}_P\d{2})_E(\d{2})_T(0[12])\.wav$"
)

for archivo in archivos:
    m = patron.fullmatch(archivo.name)
    assert m, f"Nombre incorrecto: {archivo.name}"

    p, e, t = m.group(1), int(m.group(2)), int(m.group(3))
    assert p in personas and 1 <= e <= 10
    assert (e, t) not in muestras[p]
    muestras[p].add((e, t))

validos = {}

# Solo usamos un enunciado si estan disponibles sus dos grabaciones
for p in personas:
    validos[p] = {
        e for e in range(1, 11)
        if {(e, 1), (e, 2)} <= muestras[p]
    }

    assert len(muestras[p]) == 2 * len(validos[p])

    esperado = set(range(1, 11))
    if p == "F04_P11":
        esperado.remove(8)

    assert validos[p] == esperado, f"Faltantes inesperados: {p}"

filas = []

# Esta funcion prepara los datos de una comparacion para guardarlos en el CSV
# Entre dos personas se hacen cuatro cruces de tomas, pero con la misma persona solo uno
def agregar(grupo, a, b, pareja, familia, control, tipo, enunciados):
    pa, pb = personas[a], personas[b]
    enunciados = sorted(enunciados)
    assert enunciados

    filas.append({
        "id_comparacion": f"{grupo}_{pareja or a}_{a}_{b}",
        "grupo": grupo,
        "pareja_familiar_referencia": pareja,
        "familia_estudiada": familia,
        "participante_a": a,
        "participante_b": b,
        "control_referencia": control,
        "parentesco": tipo,
        "sexo_a": pa["sex"],
        "sexo_b": pb["sex"],
        "edad_a": pa["age_years"],
        "edad_b": pb["age_years"],
        "diferencia_edad": abs(
            int(pa["age_years"]) - int(pb["age_years"])
        ),
        "n_enunciados": len(enunciados),
        "enunciados": ";".join(
            f"E{e:02d}" for e in enunciados
        ),
        "n_comparaciones_tomas": len(enunciados) * (
            1 if grupo == "misma_persona" else 4
        ),
    })

# Primero comparamos las dos tomas de cada persona para tener una referencia individual
for p in sorted(personas):
    agregar(
        "misma_persona", p, p, "",
        personas[p]["family_id"], "",
        "misma_identidad", validos[p]
    )

# Ahora armamos las comparaciones entre hermanos y con la persona de control
# Revisamos que los hermanos sean de una familia y el control de otra
for a, b, c, tipo in trios:
    assert len({a, b, c}) == 3
    assert all(p in personas for p in (a, b, c))

    fa, fb, fc = (
        personas[p]["family_id"] for p in (a, b, c)
    )

    assert fa == fb and fa != fc, "Control de familia incorrecta"

    assert len({
        personas[p]["sex"] for p in (a, b, c)
    }) == 1, "Sexo del control incompatible"

    relacion = relaciones.get(frozenset((a, b)))
    assert relacion is not None
    assert relacion["relationship"].strip().lower() == tipo
    assert relacion["biological_relation"].strip().lower() == "yes"

    assert all(
        frozenset((p, c)) not in relaciones
        for p in (a, b)
    ), "Parentesco documentado con el control"

    # Hermanos, hermano A-control y hermano B-control se comparan con las mismas frases
    comunes = validos[a] & validos[b] & validos[c]
    pareja = f"{a}__{b}"

    agregar("hermanos", a, b, pareja, fa, c, tipo, comunes)
    agregar(
        "control", a, c, pareja, fa, c,
        "sin_parentesco_conocido", comunes
    )
    agregar(
        "control", b, c, pareja, fa, c,
        "sin_parentesco_conocido", comunes
    )

conteos = Counter(f["grupo"] for f in filas)

assert conteos == {
    "misma_persona": 15,
    "hermanos": 6,
    "control": 12,
}

assert len({f["id_comparacion"] for f in filas}) == 33

esperados = {
    "misma_persona": 149,
    "hermanos": 236,
    "control": 472,
}

for grupo, cantidad in esperados.items():
    calculadas = sum(
        f["n_comparaciones_tomas"]
        for f in filas if f["grupo"] == grupo
    )
    assert calculadas == cantidad, f"Conteo incorrecto: {grupo}"

# El archivo se guarda solo si todavia no existe para no cambiar las parejas despues del analisis
salida = base / "results" / "protocolo_comparaciones.csv"
salida.parent.mkdir(parents=True, exist_ok=True)

if salida.exists():
    raise SystemExit(
        f"El protocolo ya existe y no se sobrescribirá: {salida}"
    )

with salida.open("w", encoding="utf-8", newline="") as f:
    escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
    escritor.writeheader()
    escritor.writerows(filas)

print("\n=== PROTOCOLO GENERADO ===")
print("Archivo:", salida)
print("Definiciones:", len(filas))
print("Misma persona:", conteos["misma_persona"])
print("Hermanos:", conteos["hermanos"])
print("Controles:", conteos["control"])
print("Comparaciones de tomas previstas:", sum(
    f["n_comparaciones_tomas"] for f in filas
))

print("\n=== PAREJAS Y CONTROLES ===")
for a, b, c, tipo in trios:
    n = len(validos[a] & validos[b] & validos[c])
    print(f"{a} - {b} | {tipo} | Control: {c} | Frases: {n}")
