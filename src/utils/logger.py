import os
from datetime import datetime
from pathlib import Path

class Logger:
    _instance = None
    _initialized = False

    def __new__(cls, log_dir=None):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_dir=None):
        if not Logger._initialized:
            # Por defecto, escribe en logs/ relativo a la raiz del repo
            base = Path(__file__).resolve().parent.parent
            self.log_dir = Path(log_dir) if log_dir else base / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d")
            self.log_file = self.log_dir / f"log_{timestamp}.txt"
            Logger._initialized = True

    @staticmethod
    def print(message, level="INFO"):
        """
        Método estático para imprimir y guardar mensaje con timestamp
        Args:
            message: mensaje a loggear
            level: nivel del log (INFO, WARNING, ERROR, etc)
        """
        if Logger._instance is None:
            Logger()
        
        message = str(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"
        print(log_message)
        with open(Logger._instance.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

if __name__ == "__main__":
    Logger.print("Iniciando entrenamiento...")
    Logger.print("¡Advertencia!", level="WARNING")
    Logger.print("Error encontrado", level="ERROR")
    
    # Todas las instancias son la misma
    logger1 = Logger()
    logger2 = Logger()
    print(f"¿Misma instancia?: {logger1 is logger2}") 
