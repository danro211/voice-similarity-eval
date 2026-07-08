import os
import glob
import torch
from TTS.api import TTS

# Forzamos a PyTorch a permitir la carga de configuraciones de XTTS
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

print("Cargando modelo XTTS v2 en GPU...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

texto_a_generar = "El Barcelona es el mejor equipo del mundo, sé que algún día serán campeones de Europa."

# Busca TODOS los archivos .wav en la carpeta
archivos_wav = glob.glob("data/raw/*.wav")
identidades_procesadas = set() # Aquí guardaremos los nombres de quienes ya clonamos

if not archivos_wav:
    print("No se encontraron archivos .wav en data/raw/")
else:
    for ruta_audio in archivos_wav:
        nombre_archivo = os.path.basename(ruta_audio)
        
        # 1. Filtro de seguridad: Ignorar los archivos sintéticos ya generados
        if "sintetico" in nombre_archivo:
            continue
            
        # 2. Extraer la identidad 
        identidad = nombre_archivo.split('_')[0]
        
        archivo_salida = f"data/raw/{identidad}_sintetico_xtts.wav"
        
        # 3. Control de redundancia
        if identidad in identidades_procesadas or os.path.exists(archivo_salida):
            # Si ya existe, lo saltamos para no gastar GPU ni sobreescribir
            identidades_procesadas.add(identidad)
            continue
            
        print(f"\nUsando '{nombre_archivo}' como referencia para generar la voz sintética de: {identidad}...")
        try:
            tts.tts_to_file(
                text=texto_a_generar,
                file_path=archivo_salida,
                speaker_wav=ruta_audio,
                language="es"
            )
            print(f"Guardado exitosamente: {archivo_salida}")
            # Marcamos a esta persona como procesada
            identidades_procesadas.add(identidad)
        except Exception as e:
            print(f"Error generando audio para {identidad}: {e}")

print("\n¡Proceso de clonación por lotes completado!")
