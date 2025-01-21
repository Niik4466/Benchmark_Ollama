import requests
import json
import argparse
import csv
import os
from modules.gpu_monitor import GpuMonitor

def query_ollama(prompt, model="llama2", port="127.0.0.1:11434"):
    """
    Envía una consulta directa a la API de Ollama y devuelve los tokens/s junto con el prompt.

    Args:
        prompt (str): La consulta que deseas enviar al modelo.
        model (str): El modelo a utilizar (por defecto, 'llama2').
        port (str): El puerto donde se ejecuta el servicio Ollama (por defecto, 127.0.0.1:11434).

    Returns:
        tuple: Prompt, tokens/s (o un mensaje de error si algo falla), objeto gpu_monitor con los datos de uso de gpu
    """
    # Definimos la url a consultar y los headers de la consulta
    url = f"http://{port}/api/generate"
    headers = {
        "Content-Type": "application/json",
    }

    # Definimos la consulta
    payload = {
        "model": model,
        "prompt": prompt
    }
    
    # Definimos nuestro objeto GpuMonitor para monitorear los recursos
    gpu_monitor = GpuMonitor(0.1)

    # Empezamos el monitoreo
    gpu_monitor.start()
    try:
        # Realizamos la consulta a la API. Importante considerar que se nos devuelve un stream de Jsons no uno unico
        response = requests.post(url, json=payload, headers=headers)
        # Terminamos de monitorear la gpu una vez esta lista la respuesta
        gpu_monitor.stop()

        response.raise_for_status()  # Lanza una excepción para errores HTTP
        # Iteramos por sobre la respuesta para obtener el ultimo Json que contiene los datos necesarios.
        for line in response.iter_lines(decode_unicode=True):
            if line.strip(): # Ignoramos las lineas vacias
                data = json.loads(line)
        
        # Extraer la duración de la evaluación y la cantidad de evaluaciones
        prompt_eval_duration = data.get("eval_duration", 0)  # En Nanosegundos
        prompt_eval_count = data.get("eval_count", 0) 
        
        # Si los valores están disponibles, calcular los tokens por segundo
        if prompt_eval_duration > 0 and prompt_eval_count > 0:
            # Calcular tokens/s
            tokens_per_second = prompt_eval_count / (prompt_eval_duration / 1e9)
            print("Respuesta completa:", response.text)
            return prompt, tokens_per_second, gpu_monitor
        else:
            return prompt, "No disponible", gpu_monitor

    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud: {e}")
        print("Respuesta completa:", response.text)
        return prompt, f"Error: {e}", gpu_monitor

# Configuración del parser de argumentos
parser = argparse.ArgumentParser(description="Script para consultar la API de Ollama y registrar tokens/s")

parser.add_argument(
    "--prompts", "-p",
    required=True,
    type=str,
    help="Lista de prompts separados por comas (ejemplo: prompt1,prompt2,prompt3)",
    dest="prompts"
)
parser.add_argument(
    "--port",
    required=False,
    type=str,
    help="Puerto a utilizar para conectarse con Ollama",
    default="127.0.0.1:11434",
    dest="port"
)
parser.add_argument(
    "--model", "-m",
    required=False,
    type=str,
    help="Modelo a utilizar (por defecto: llama2)",
    default="llama2",
    dest="model"
)
parser.add_argument(
    "-g",
    required=False,
    type=int,
    help="Cantidad de Gpu's utilizadas",
    dest="gpus"
)

args = parser.parse_args()

# Leer prompts y directorio de resultados
prompt_list = args.prompts.strip('[]').split(',')

# Obtener ruta del archivo de resultados desde la variable de entorno
result_path = os.environ.get("RESULT_PATH", ".")  # Por defecto, usar el directorio actual
output_file = os.path.join(result_path, "ollama_results.csv")

# Verificar si el archivo ya existe para agregar encabezados solo si es necesario
file_exists = os.path.isfile(output_file)

# Procesamos e modelo para obtener la cantidad de parametros y la quantizacion
#params = args.model.split(":")
#quantization = params[1].split("-")[2]
#model_name=params[0]
#params=params[1].split("-")[0]
model_name="llama2"
params = "7b"
quantization = "q4"

# Procesar prompts y registrar resultados
with open(output_file, mode='a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Escribir encabezados si el archivo no existía
    if not file_exists:
        writer.writerow(["Model"
                        , "Params"
                        , "Quantization"
                        , "Tokens/s"
                        , "Num_Gpus"
                        , "GPU_0_VRAM_usage_avg"
                        , "GPU_0_VRAM_usage_max"
                        , "GPU_0_Power_avg"
                        , "GPU_0_Power_max"
                        , "GPU_1_VRAM_usage_avg"
                        , "GPU_1_VRAM_usage_max"
                        , "GPU_1_Power_avg"
                        , "GPU_1_Power_max"
                        , "GPU_2_VRAM_usage_avg"
                        , "GPU_2_VRAM_usage_max"
                        , "GPU_2_Power_avg"
                        , "GPU_2_Power_max"
                        , "GPU_3_VRAM_usage_avg"
                        , "GPU_3_VRAM_usage_max"
                        , "GPU_3_Power_avg"
                        , "GPU_3_Power_max"
                        , "GPU_4_VRAM_usage_avg"
                        , "GPU_4_VRAM_usage_max"
                        , "GPU_4_Power_avg"
                        , "GPU_4_Power_max"
                        , "GPU_5_VRAM_usage_avg"
                        , "GPU_5_VRAM_usage_max"
                        , "GPU_5_Power_avg"
                        , "GPU_5_Power_max"
                        , "GPU_6_VRAM_usage_avg"
                        , "GPU_6_VRAM_usage_max"
                        , "GPU_6_Power_avg"
                        , "GPU_6_Power_max"])

    for prompt in prompt_list:
        print(f"Consultando con el prompt: {prompt.strip()}")
        prompt, tokens_per_second, gpu_monitor = query_ollama(prompt.strip(), model=args.model.strip(), port=args.port)
        gpu_stats = gpu_monitor.get_stats()
        sorted_gpu_stats = {key: gpu_stats[key] for key in sorted(gpu_stats.keys())}
        # Escribir en el archivo CSV
        writer.writerow([model_name, params, quantization, tokens_per_second, args.gpus] + list(sorted_gpu_stats.values()))
