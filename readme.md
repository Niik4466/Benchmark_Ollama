# Ollama Benchmark NLHPC

## Resumen

Ollama Benchmark NLHPC es un benchmark creado para testear el uso de ollama en entornos de supercomputacion que utilizan un gestor de tareas SLURM y modulos de software lmod.
Este proyecto se enfoca en la personalizacion y el monitoreo de los recursos utilizados en GPU y multi GPU tanto para AMD como NVIDIA

## Estructura del proyecto

- **main.py**: Programa principal a ejecutar. Toma los parametros de ejecucion, informa la configuracion a ejecutar y lanza varias instancias del programa sbatch, la gracia es que guardara los job_id para generar una dependencia de tareas, logrando que solamente un job utilice los recursos al mismo tiempo.
- **sbatch_generator.sh**: Script en bash que genera scripts sbatch de manera automatica para ejecutar, estos scripts ejecutan el servicio de ollama y el programa run_tests.py. El primer job es independiente y los siguientes tienen dependencia con el anterior.
- **run_tests.py**: Encargado de realizar las consultas al servicio de ollama. Se obtienen datos como tokens/s y utilizacion en VRAM de la gpu, a continuacion el detalle:
    - **Nombre del modelo**
    - **Cantidad de parametros del modelo**
    - **Quantizacion del modelo**
    - **Tokens por segundo**
    - **Numero de GPU's utilizadas**
    - **Uso de VRAM de la gpu**
    - **Uso energetico en mV de la gpu**
- **models.JSON**: Lista de modelos a ejecutar, se recomienda utilizar el modelo exacto de ollama en el formato `<modelo>:<parametros>-<cuantizacion>`, sacado desde https://ollama.com/search. Por ejemplo: `llama3.1:8b-instruct-q8_0`. En la seccion "weight" se debe ingresar el peso en MB del modelo a utilizar.
- **prompts.JSON**: Lista de prompts a ejecutar.

Una vez terminada la ejecucion de `main.py`. Los resultados en formato .csv quedan guardados en `$RESULT_PATH`

## Instalacion

Para Instalar y ejecutar el bench se requiere de un entorno virtual venv y las dependencias especificadas en el archivo `requirements.txt`

1. Instalar el entorno virtual
    ```bash
    python -m venv ollama_bench
    ```
2. Activar el entorno virtual
    ```bash
    source ollama_bench/bin/activate
    ```
3. Instalar las librerias de python
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Ejecucion

Para ejecutar el benchmark se deben seguir los siguientes pasos posteriores a la instalacion

1. Activar el entorno venv
    ```bash
    source ollama_bench/bin/activate
    ```
2. Ejecutar el programa main.py
    ```bash
    python main.py <params>
    ```

## Parametros

el programa `main.py` tiene varios parametros para personalizar la ejecucion del benchmark
Claro, aquí tienes la sección de parámetros completada para tu archivo `README.md`:

---

## Parámetros

El programa `main.py` incluye varios parámetros que permiten personalizar la ejecución del benchmark.

- **`--gpus`, `-g`**:  
  Especifica el número de GPUs totales que se utilizarán durante la prueba.  
  **Por defecto**: `1`.  
  **Ejemplo**: `--gpus 4`.

- **`--rep`, `-r`**:  
  Define la cantidad de repeticiones para ejecutar el experimento. Esto puede ser útil para obtener resultados más consistentes al promediar múltiples ejecuciones.  
  **Por defecto**: `1`.  
  **Ejemplo**: `--rep 10`.

- **`--partition`, `-p`**:  
  Determina la partición del cluster donde se realizarán las pruebas. Esto depende de la configuración del NLHPC.  
  **Por defecto**: `mi210`.  
  **Ejemplo**: `--partition v100`.

- **`--port`**:  
  Especifica el puerto desde donde la API de Ollama escucha.  
  **Por defecto**: `127.0.0.1:11434`
  **Ejemplo**: `--port 127.0.0.1:8080`.


## Variables de entorno

Aquí tienes la sección de variables de entorno para tu archivo `README.md`:

---

## Variables de Entorno

El programa `main.py` utiliza variables de entorno para configurar ciertos aspectos del benchmark. A continuación, se describen:

- **`MODELS_JSON_PATH`**:  
    Ruta al archivo JSON que contiene la definición de los modelos a evaluar. Este archivo debe incluir las especificaciones necesarias para los modelos que se utilizarán en el benchmark.
    **Por defecto**: `models.JSON`  
    **Ejemplo**: `/path/to/models.json`.

- **`PROMPTS_JSON_PATH`**:  
    Ruta al archivo JSON que contiene los prompts o entradas que se usarán durante las pruebas. Este archivo debe incluir los textos o configuraciones de entrada para el benchmark.
    **Por defecto**: `prompts.JSON`  
    **Ejemplo**: `/path/to/prompts.json`.

- **`RESULT_PATH`**:  
    Ruta al directorio donde se almacenarán los resultados generados por el benchmark. El programa guardará los datos en esta ubicación para análisis posterior.
    **Por defecto**: `.`  
    **Ejemplo**: `/path/to/results`.

- **`DEVICE_BACKEND`**:
    Indica con que software van a correr las pruebas `cuda/rocm`. El programa utilizara cierta libreria para medir el uso dependiendo del backend.
    **Por defecto**: `rocm`
    **Ejemplo**: `cuda`

- **`MODELS_DIR`**:
    Indica al servicio de ollama de que directorio obtendra y descargara los modelos a utilizar en el archivo `models.JSON`
    **Por defecto**: `~/.ollama/models`
    **Ejemplo**: `~/ollama_models`

#### Configuración temporal
Puedes exportar las variables directamente en la terminal antes de ejecutar el script:

```bash
export MODELS_JSON_PATH=/path/to/models.JSON
export PROMPTS_JSON_PATH=/path/to/prompts.JSON
export RESULT_PATH=/path/to/results
export DEVICE_BACKEND=rocm
export MODELS_DIR=/path/to/models
```