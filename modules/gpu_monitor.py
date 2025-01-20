from pyamdgpuinfo import *
import os
import threading
import time

class GpuMonitor():
    def __init__(self, interval=0.1):
        """
        Detecta las GPU's disponibles, asigna una lista de objetos gpu1, gpu2, gpuX

        Monitorea el uso de los recursos en intervalos regulares (por defecto 0.1 segundos)

        Args:
            interval (float): Intervalo entre mediciones
        """
        
        # VARS
        self.interval = interval
        gpus_ids = list(map(int, os.getenv('ROCR_VISIBLE_DEVICES').split(",")))
        self.gpus = list(map(get_gpu, gpus_ids))
        self.utilisation = [[] for _ in range(len(self.gpus))]
        self.vram_usage = [[] for _ in range(len(self.gpus))]
        self.power = [[] for _ in range(len(self.gpus))]
        self.running = False
        self.thread = None

    def start(self):
        """Inicia el monitoreo en un hilo separado"""
        if (self.running):
            print("El monitoreo ya ha empezado")
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
    
    def stop(self):
        """Detiene el Hilo de monitoreo"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor(self):
        # Comenzamos el polling para utilizar el metodo query_utilisation()
        for i in range((self.gpus)): self.gpus[i].start_query_utilisation_polling()
        # Comenzamos a registrar las metricas una vez el consumo de energia sea mayor que 47 mW
        start = False
        while self.running:
            if start:
                # Registramos las metricas para cada gpu
                for id_gpu in range(len(self.gpus)):
                    self.utilisation[id_gpu].append((self.gpus[id_gpu].query_utilisation()))
                    self.vram_usage[id_gpu].append(self.gpus[id_gpu].query_vram_usage())
                    self.power.append(self.gpus[id_gpu].query_power())
            else:
                start = any(gpu.query_power() > 41 for gpu in self.gpus)
            time.sleep(self.interval)
        for i in range(len(self.gpus)): self.gpus[i].stop_query_utilisation_polling()

    def get_stats(self):
        """
        Obtiene estadísticas agregadas (promedio, máximo) de las métricas recopiladas en _monitor.
        """
        stats = {}
        for idx in range(len(self.gpus)):
            stats[f"gpu_{idx}_utilisation_avg"] = sum(self.utilisation[idx]) / len(self.utilisation[idx]) if self.utilisation[idx] else None
            stats[f"gpu_{idx}_utilisation_max"] = max(self.utilisation[idx]) if self.utilisation[idx] else None
            stats[f"gpu_{idx}_vram_usage_avg"] = sum(self.vram_usage[idx]) / len(self.vram_usage[idx]) if self.vram_usage[idx] else None
            stats[f"gpu_{idx}_vram_usage_max"] = max(self.vram_usage[idx]) if self.vram_usage[idx] else None
            stats[f"gpu_{idx}_power_avg"] = sum(self.power[idx]) / len(self.power[idx]) if self.power[idx] else None
            stats[f"gpu_{idx}_power_max"] = max(self.power[idx]) if self.power[idx] else None
        return stats