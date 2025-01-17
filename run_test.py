import requests
import argparse
import csv
import os

def query_ollama(prompt, model="llama2", port="127.0.0.1:11434"):
    """
    Envía una consulta directa a la API de Ollama y devuelve los tokens/s junto con el prompt.

    Args:
        prompt (str): La consulta que deseas enviar al modelo.
        model (str): El modelo a utilizar (por defecto, 'llama2').
        port (str): El puerto donde se ejecuta el servicio Ollama (por defecto, 127.0.0.1:11434).

    Returns:
        tuple: Prompt, tokens/s (o un mensaje de error si algo falla).
    """
    url = f"http://{port}/api/query"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Lanza una excepción para errores HTTP
        data = response.json()
        
        # Extraer la duración de la evaluación y la cantidad de evaluaciones
        prompt_eval_duration = data.get("prompt_eval_duration", 0)  # En microsegundos
        prompt_eval_count = data.get("prompt_eval_count", 1)  # Asumimos al menos una evaluación
        
        # Si los valores están disponibles, calcular los tokens por segundo
        if prompt_eval_duration > 0 and prompt_eval_count > 0:
            # Convertir la duración de microsegundos a segundos
            tokens_per_second = (prompt_eval_count / (prompt_eval_duration / 1_000_000))
            return prompt, tokens_per_second
        else:
            return prompt, "No disponible"
    
    except requests.exceptions.RequestException as e:
        return prompt, f"Error: {e}"

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

args = parser.parse_args()

# Leer prompts y directorio de resultados
prompt_list = args.prompts.strip('[]').split(',')

# Obtener ruta del archivo de resultados desde la variable de entorno
result_path = os.environ.get("RESULT_PATH", ".")  # Por defecto, usar el directorio actual
output_file = os.path.join(result_path, "ollama_results.csv")

# Verificar si el archivo ya existe para agregar encabezados solo si es necesario
file_exists = os.path.isfile(output_file)

# Procesar prompts y registrar resultados
with open(output_file, mode='a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Escribir encabezados si el archivo no existía
    if not file_exists:
        writer.writerow(["Prompt", "Tokens/s"])
    
    for prompt in prompt_list:
        print(f"Consultando con el prompt: {prompt.strip()}")
        prompt, tokens_per_second = query_ollama(prompt.strip(), model=args.model, port=args.port)
        print(f"Tokens/s: {tokens_per_second}")
        print("-" * 40)
        # Escribir en el archivo CSV
        writer.writerow([prompt, tokens_per_second])
