from typing import Any
from abc import ABC, abstractmethod
import mlflow
import mlflow.keras
import pandas as pd
import numpy as np
import tensorflow as tf

class LearningModel(ABC):
    """
    Interfaz abstracta para cualquier modelo de aprendizaje automático.
    Define los métodos esenciales que todo modelo debe implementar.
    """
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str) -> Any:
        pass

class MLPModel(LearningModel):
    """
    Implementación de un Perceptrón Multicapa (MLP) usando TensorFlow/Keras.
    """
    def __init__(self, model: tf.keras.Model):
        self._model = model
        print("MLPModel inicializado.")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Entrena el modelo y registra el experimento con MLflow.
        """
        print("Iniciando entrenamiento del modelo...")
        # Iniciar un nuevo run de MLflow
        with mlflow.start_run() as run:
            print(f"MLflow Run ID: {run.info.run_id}")
            
            # Registrar parámetros del entrenamiento
            mlflow.log_params(kwargs.get('params', {}))
            
            # Entrenar el modelo
            history = self._model.fit(
                X_train,
                y_train,
                epochs=kwargs.get('epochs', 10),
                batch_size=kwargs.get('batch_size', 32),
                validation_split=kwargs.get('validation_split', 0.2),
                verbose=0 # Para no saturar la consola
            )
            
            # Registrar métricas
            final_metrics = {f"final_{k}": v[-1] for k, v in history.history.items()}
            mlflow.log_metrics(final_metrics)
            print(f"Entrenamiento finalizado. Métricas finales: {final_metrics}")
            
            # Registrar el modelo
            mlflow.keras.log_model(self._model, "model")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones sobre nuevos datos.
        """
        return self._model.predict(X)

    def save(self, path: str):
        """
        Guarda el modelo en la ruta especificada.
        """
        self._model.save(path)
        print(f"Modelo guardado en: {path}")

    @classmethod
    def load(cls, path: str) -> 'MLPModel':
        """
        Método de clase para cargar un modelo desde una ruta.
        """
        loaded_model = tf.keras.models.load_model(path)
        print(f"Modelo cargado desde: {path}")
        return cls(loaded_model)

