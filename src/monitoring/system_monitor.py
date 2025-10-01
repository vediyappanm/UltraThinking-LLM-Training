"""System resource monitoring"""
import psutil
import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_used_gb: float
    disk_usage_percent: float
    timestamp: float


class SystemMonitor:
    """Monitor system resources during training"""
    
    def __init__(self, log_interval: int = 60):
        """
        Args:
            log_interval: Seconds between system metric logs
        """
        self.log_interval = log_interval
        self.last_log_time = 0
        self.metrics_history = []
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=memory.percent,
            memory_available_gb=memory.available / 1e9,
            memory_used_gb=memory.used / 1e9,
            disk_usage_percent=disk.percent,
            timestamp=time.time()
        )
    
    def check_and_log(self, force: bool = False) -> Optional[SystemMetrics]:
        """Check system metrics and log if interval elapsed"""
        current_time = time.time()
        
        if force or (current_time - self.last_log_time) >= self.log_interval:
            metrics = self.get_current_metrics()
            self.metrics_history.append(metrics)
            self.last_log_time = current_time
            
            logger.info(
                f"System Metrics | "
                f"CPU: {metrics.cpu_percent:.1f}% | "
                f"RAM: {metrics.memory_used_gb:.1f}/{metrics.memory_used_gb + metrics.memory_available_gb:.1f}GB "
                f"({metrics.memory_percent:.1f}%) | "
                f"Disk: {metrics.disk_usage_percent:.1f}%"
            )
            
            # Warn if resources are high
            if metrics.memory_percent > 90:
                logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
            
            if metrics.cpu_percent > 95:
                logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            
            if metrics.disk_usage_percent > 90:
                logger.warning(f"High disk usage: {metrics.disk_usage_percent:.1f}%")
            
            return metrics
        
        return None
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of system metrics"""
        if not self.metrics_history:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        mem_values = [m.memory_percent for m in self.metrics_history]
        
        return {
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': sum(mem_values) / len(mem_values),
            'max_memory_percent': max(mem_values),
            'max_memory_used_gb': max(m.memory_used_gb for m in self.metrics_history)
        }
