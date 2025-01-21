#!/bin/bash

# Función para mostrar el uso del script
usage() {
    echo "Uso: $0 -p <particion> --prompts=[prompt1,prompt2,prompt3] --gpus=<num_gpus> -m <model_name>"
    echo "  -p <particion>       : Nombre de la partición para SBATCH"
    echo "  --prompts=<prompts>  : Lista de prompts separados por comas (ejemplo: prompt1,prompt2,prompt3)"
    echo "  --gpus=<num_gpus>    : Número de GPUs a asignar"
    echo "  -m <model_name>      : Nombre del modelo a usar"
    echo "  -w <model weight>    : Peso del modelo especificado con -m"
    echo "  --job_id=<job_id>    : Job id del anterior trabajo, si es -1 entonces el lanzamiento sera independiente"
    echo "  --port=<ip:port>     : Configuracion del puerto para OLLAMA_HOST"
    exit 1
}

# Inicializar variables
partition=""
prompts=""
gpus=1
model_name=""
model_weight=0

# Parsear los argumentos
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p) partition="$2"; shift ;;
        --prompts=*) prompts="${1#*=}" ;;
        --gpus=*) gpus="${1#*=}" ;;
        -m) model_name="$2"; shift ;;
        -w) model_weight="$2"; shift ;;
        --job_id=*) job_id="${1#*=}" ;;
        --port=*) port="${1#*=}" ;;
        -h|--help) usage ;;
	    *) echo "Error: Opción desconocida: $1"; usage ;;
    esac
    shift
done

# Validaciones
if [[ -z "$partition" || -z "$prompts" || -z "$model_name" ]]; then
    echo "Error: Faltan parámetros obligatorios."
    usage
fi

# Validar que el número de GPUs sea un entero positivo
if ! [[ "$gpus" =~ ^[0-9]+$ ]]; then
    echo "Error: El parámetro --gpus debe ser un número entero positivo."
    exit 1
fi

if ! [[ "$model_weight" =~ ^[0-9]+$ ]]; then
    echo "Error: El parámetro -w debe ser un número entero positivo."
    exit 1
fi

# Crear un script temporal para SBATCH
temp_script="$(pwd)/sbatch_script_$$.job"
cat <<EOF > "$temp_script"
#!/bin/bash
#SBATCH -J ollama_bench_${model_name}_${partition}
#SBATCH -p ${partition}
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=${model_weight}
#SBATCH --gres=gpu:${gpus}
#SBATCH -o logs/ollama_bench_${model_name}_%j.out.err
#SBATCH -e logs/ollama_bench_${model_name}_%j.out.err

# ----------------Modulos----------------------------
ml aocc gcc/14.2.0-zen4-y python/3.9.19-zen4-l
ml py-torch ollama
# ----------------Comandos--------------------------

export OLLAMA_HOST=${port}

~/ollama/bin/ollama serve &
sleep 3

# Cargar ambiente venv
source ollama_bench/bin/activate

# Ejecutar el script Python con los prompts
python run_test.py --prompts=${prompts} -m ${model_name} --port=${port} -g ${gpus}
EOF

# Enviar el script a la cola de trabajos
cat "$temp_script"
chmod 755 "$temp_script"

if [[ job_id != -1 ]]; then
    sbatch "$temp_script"
else
    sbatch --dependency=after:"$job_id" "$temp_script"
fi

# Limpiar
sleep 2
rm -f "$temp_script"
