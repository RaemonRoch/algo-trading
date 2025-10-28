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
    


class CNNModel(LearningModel):
    """
    Implementación de una Red Neuronal Convolucional (CNN) usando TensorFlow/Keras.
    """
    def __init__(self, model: tf.keras.Model):
        """
        Inicializa el wrapper del modelo CNN.

        Args:
            model (tf.keras.Model): Un modelo de CNN de Keras compilado.
        """
        self._model = model
        print("CNNModel inicializado.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        """
        Entrena el modelo CNN y registra el experimento con MLflow.

        Args:
            X_train (np.ndarray): Datos de entrenamiento. Para una CNN, esto suele
                                  ser un array de imágenes con forma (n_samples, height, width, channels).
            y_train (np.ndarray): Etiquetas de entrenamiento.
            **kwargs: Argumentos adicionales para el método `fit` de Keras y para MLflow.
                      Ej: epochs, batch_size, validation_split, params (para mlflow).
        """
        print("Iniciando entrenamiento del modelo CNN...")
        # Iniciar un nuevo run de MLflow
        with mlflow.start_run() as run:
            print(f"MLflow Run ID: {run.info.run_id}")
            
            # Registrar parámetros del entrenamiento
            # Se espera un diccionario 'params' dentro de kwargs
            mlflow.log_params(kwargs.get('params', {}))
            
            # Entrenar el modelo
            history = self._model.fit(
                X_train,
                y_train,
                epochs=kwargs.get('epochs', 10),
                batch_size=kwargs.get('batch_size', 32),
                validation_split=kwargs.get('validation_split', 0.2),
                verbose=kwargs.get('verbose', 1) # Verbose 1 es común para ver el progreso
            )
            
            # Registrar métricas al final del entrenamiento
            # Se toman los valores de la última época
            final_metrics = {f"final_{k}": v[-1] for k, v in history.history.items()}
            mlflow.log_metrics(final_metrics)
            print(f"Entrenamiento finalizado. Métricas finales: {final_metrics}")
            
            # Registrar el modelo en MLflow
            mlflow.keras.log_model(self._model, "cnn-model")
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones sobre nuevos datos.

        Args:
            X (np.ndarray): Datos para la predicción, típicamente un lote de imágenes.

        Returns:
            np.ndarray: Las predicciones del modelo.
        """
        print("Realizando predicciones...")
        return self._model.predict(X)

    def save(self, path: str):
        """
        Guarda el modelo Keras en la ruta especificada.

        Args:
            path (str): Ruta del archivo o directorio donde se guardará el modelo.
        """
        self._model.save(path)
        print(f"Modelo CNN guardado en: {path}")

    @classmethod
    def load(cls, path: str) -> 'CNNModel':
        """
        Método de clase para cargar un modelo Keras desde una ruta y devolver
        una instancia de CNNModel.

        Args:
            path (str): Ruta desde donde cargar el modelo.

        Returns:
            CNNModel: Una nueva instancia de la clase con el modelo cargado.
        """
        loaded_model = tf.keras.models.load_model(path)
        print(f"Modelo CNN cargado desde: {path}")
        return cls(loaded_model)
