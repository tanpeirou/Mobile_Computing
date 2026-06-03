import psutil
import time
import os

class ResourceMonitor:
    def start(self):
        self.process = psutil.Process(os.getpid())
        self.start_mem = self.process.memory_info().rss
        self.start_time = time.time()

    def stop(self):
        end_mem = self.process.memory_info().rss
        elapsed = time.time() - self.start_time
        return {
            "latency": elapsed,
            "memory_mb": (end_mem - self.start_mem) / (1024 * 1024)
        }
