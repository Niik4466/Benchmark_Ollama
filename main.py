import json, os, argparse, subprocess
from dotenv import load_dotenv


# -------------- SETEAMOS LAS FLAGS -------------------

parser = argparse.ArgumentParser(description="Benchmark de Ollama API en el NLHPC")

parser.add_argument(
    "--gpus", "-g",
    type=int,
    default=1,
    help="Numero de GPUs totales de la prueba, por defecto 1",
    dest="num_gpus"
)

parser.add_argument(
    "--rep", "-r",
    type=int,
    default=1,
    help="Cantidad de veces a ejecutar el experimento, por defecto 1",
    dest="num_rep"
)

parser.add_argument(
    "--partition", "-p",
    type=str,
    default="mi210",
    help="Particion a utilizar para correr las pruebas, por defecto mi210",
    dest="partition"
)

parser.add_argument(
    "--port",
    type=str,
    default="127.0.0.1:11434",
    help="Puerto desde donde la API de ollama escucha",
    dest="port"
)

args = parser.parse_args()

# -------------- CARGAMOS VARIABLES DE ENTORNO ------------------

# Cargar archivos desde un .env
load_dotenv()
models_json_path=os.getenv('MODELS_JSON_PATH')
prompts_json_path = os.getenv('PROMPTS_JSON_PATH')
result_path = os.getenv('RESULT_PATH')


# -------------- CARGAMOS DATOS DE LOS MODELOS ------------------

# Leer el archivo JSON
with open(models_json_path, 'r') as models_json_file:
    models_dict = json.load(models_json_file)  # Cargar el contenido como un diccionario
with open(prompts_json_path, 'r') as  promts_json_file:
    prompts_dict = json.load(promts_json_file)

# Acceder a la lista de modelos y prompts
models_name_list = models_dict.get('models', [])
model_weight_list = models_dict.get('weights', [])
prompts_list = prompts_dict.get('prompts', [])

# -------------- INFORMACION DE CONFIGURACION ---------------

print("Modelos:", models_name_list)
print("Model weights", model_weight_list)
print("Prompts:", prompts_list)
print("Gpus a utilizar: ", args.num_gpus)
print("Cantidad de repeticiones: ", args.num_rep)

# -------------- LANZAR PRUEBAS ----------------------------

# Funcion auxiliar para encontrar el job_id de un trabajo lanzado
def get_jobid(result):
        # Verificar si el comando se ejecutó correctamente
    if result.returncode == 0:
        # Extraer el JOB ID usando una expresión regular
        match = result.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = match.group(1)
        return job_id

job_id=-1

# Lanzamos las pruebas con SLURM
for r in range(args.num_rep):
    for g in range(args.num_gpus):
        for model,weight in zip(models_name_list, model_weight_list):
            result = subprocess.run([f"./sbatch_generator.sh -p {args.partition} --gpus={g} -m {model} --prompts={prompts_list} -w {weight} --job_id={job_id} --port={args.port}"], capture_output=True, text=True, check=True)
            job_id = get_jobid(result=result)
