from pynvml import *
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

        nvmlInit()

        # VARS
        self.interval = interval
        self.gpus = [nvmlDeviceGetHandleByIndex(i) for i in range(nvmlDeviceGetCount())]
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
        nvmlShutdown()
    
    def _monitor(self):
        # Comenzamos a registrar las metricas una vez el consumo de energia sea mayor que 41 mW
        start = False
        while self.running:
            if start:
                # Registramos las metricas para cada gpu
                for idx, gpu in enumerate(self.gpus):
                    self.vram_usage[idx].append(nvmlDeviceGetMemoryInfo(gpu).used)
                    self.power[idx].append(nvmlDeviceGetPowerUsage(gpu))
            else:
                start = any(nvmlDeviceGetPowerUsage(gpu) > 41 for gpu in self.gpus)
            time.sleep(self.interval)

    def get_stats(self):
        """
        Obtiene estadísticas agregadas (promedio, máximo) de las métricas recopiladas en _monitor.
        """
        stats = {}
        for idx in range(len(self.gpus)):
            stats[f"gpu_{idx}_vram_usage_avg"] = sum(self.vram_usage[idx]) / len(self.vram_usage[idx]) if self.vram_usage[idx] else None
            stats[f"gpu_{idx}_vram_usage_max"] = max(self.vram_usage[idx]) if self.vram_usage[idx] else None
            stats[f"gpu_{idx}_power_avg"] = sum(self.power[idx]) / len(self.power[idx]) if self.power[idx] else None
            stats[f"gpu_{idx}_power_max"] = max(self.power[idx]) if self.power[idx] else None
        return stats