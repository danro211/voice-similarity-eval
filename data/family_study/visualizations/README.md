# Visualizacion de resultados de similitud vocal

Este directorio contiene los programas que generan las figuras del estudio de
similitud vocal entre hermanos. El codigo lee las tablas del analisis existente;
no procesa ni publica grabaciones y no vuelve a calcular los embeddings.

## Estructura

- `generadores/`: tres programas ejecutables, agrupados por el tipo de figuras
- `componentes/`: funciones compartidas de graficacion, estilo y validacion
- `componentes/ecapa/`: graficos y validaciones especificos de ECAPA-TDNN
- `tests/`: pruebas automaticas que utilizan datos sinteticos

## Generadores

| Programa | Figuras | Contenido |
|---|---|---|
| `generadores/generar_resumen_familiar.py` | V05, V06, V15 | Contrastes y resumen global entre hermanos y controles |
| `generadores/generar_similitud_ecapa.py` | V01-V04 | Puntuaciones de similitud coseno ECAPA-TDNN |
| `generadores/generar_comparacion_acustica.py` | V08, V09 | Distancias de tono, duracion y articulacion |

Los generadores consultan los CSV de `data/family_study/results/` mediante
`--base`. Esta carpeta de resultados es **privada**, no debe publicarse.

Desde la raiz del repositorio y con acceso autorizado a los
resultados privados, se pueden ejecutar verificaciones sin generar archivos:

```bash
DATOS="/ruta/privada/data/family_study"
python data/family_study/visualizations/generadores/generar_resumen_familiar.py --base "$DATOS" --modo verificar
python data/family_study/visualizations/generadores/generar_similitud_ecapa.py --base "$DATOS" --modo verificar
python data/family_study/visualizations/generadores/generar_comparacion_acustica.py --base "$DATOS" --modo verificar
```

Se debe sustituir `DATOS` por la ubicacion local autorizada del estudio.
Los CSV privados no estan incluidos en el repositorio publico.

Para comprobar el dibujo sin exportaciones, se utiliza `--modo probar-render`.
La generacion definitiva utiliza `--modo generar` y, cuando corresponda,
`--run-id` unico. Los resultados historicos y sus manifiestos no se modifican.

Las seis pruebas automaticas se ejecutan con:

```bash
python -B -m unittest discover -s data/family_study/visualizations/tests -p 'test_*.py' -v
```

## Compatibilidad y procedencia

Las rutas se reorganizaron para facilitar la lectura del codigo. Se ajustaron
solo los importadores y las referencias a la ubicacion de los archivos; no se
modificaron los calculos de las figuras. Por esa razon, los hashes del codigo
y de nuevos manifiestos pueden diferir de los de las figuras historicas.

Los CSV, las figuras exportadas y los manifiestos se mantienen en el
repositorio de trabajo privado, fuera del codigo destinado a GitHub.
