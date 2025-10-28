from typing import List
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from src.core.market import Exchange
from src.core.models import LearningModel


class Strategy(ABC):
    """
    Interfaz abstracta para una estrategia de trading.
    Define el método esencial para obtener señales de trading.
    """
    @abstractmethod
    def get_signals(self, features: pd.DataFrame) -> np.ndarray:
        """
        Genera señales de trading a partir de un DataFrame de características.
        Señales: 1 (comprar/long), 0 (vender/short), -1 (mantener/neutral).
        """
        pass

class MLStrategy(Strategy):
    """
    Estrategia basada en un modelo de Machine Learning para generar señales.
    """
    def __init__(self, model: LearningModel,
                 tp:List[float],
                 sl:List[float]):
        self.model = model
        self.signal_threshold = 0.5 # Umbral para decidir entre long (1) o short (0)
        self.sl = sl
        self.tp = tp
        print("MLStrategy inicializada con un modelo.")

    def get_signals(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predice la probabilidad y la convierte en una señal binaria.
        """
        probabilities = self.model.predict(features)
        # Convertir probabilidades a señales (1 para long, 0 para short)
        signals = (probabilities > self.signal_threshold).astype(int)
        return signals

    def check_and_retrain(self, new_data: pd.DataFrame, target_column: str):
        """
        Verifica si hay deriva de datos y reentrena el modelo si es necesario.
        NOTA: La detección de deriva de datos real (e.g., con KS) es compleja.
        Aquí se simula una lógica simple para el reentrenamiento.
        """
        # Lógica de detección de deriva de datos (simplificada)
        # En un caso real, usarías una librería como evidently.ai o alibi-detect.
        needs_retraining = self._detect_data_drift(new_data)
        
        if needs_retraining:
            print("Deriva de datos detectada. Reentrenando el modelo...")
            X_new = new_data.drop(columns=[target_column])
            y_new = new_data[target_column]
            
            # Aquí pasarías los hiperparámetros de reentrenamiento
            self.model.fit(X_new, y_new, epochs=5, batch_size=64, params={'retraining': True})
        else:
            print("No se detectó deriva de datos significativa.")
            
    def _detect_data_drift(self, new_data) -> bool:
        return np.random.choice([True, False], p=[0.1, 0.9]) # 10% de probabilidad de reentrenar


class CompoundStrategy(Strategy):
    """
    Una estrategia que combina las señales de múltiples estrategias.
    """
    def __init__(self, strategies: List[Strategy], aggregation_logic: str = 'unanimous'):
        self.strategies = strategies
        self.aggregation_logic = aggregation_logic # 'unanimous', 'majority_vote'
        print(f"CompoundStrategy inicializada con {len(strategies)} estrategias y lógica de '{aggregation_logic}'.")

    def get_signals(self, features: pd.DataFrame) -> np.ndarray:
        """
        Genera una señal final basada en la agregación de señales de las sub-estrategias.
        """
        all_signals = np.array([s.get_signals(features) for s in self.strategies])
        
        # Lógica de agregación
        if self.aggregation_logic == 'unanimous':
            # La señal es 1 si TODAS son 1
            long_signal = np.all(all_signals == 1, axis=0)
            # La señal es 0 si TODAS son 0
            short_signal = np.all(all_signals == 0, axis=0)
            
            # Por defecto, la señal es neutral (-1)
            final_signal = np.full(long_signal.shape, -1)
            final_signal[long_signal] = 1
            final_signal[short_signal] = 0
            return final_signal
        
        elif self.aggregation_logic == 'majority_vote':
            # Calcula la media y redondea para obtener el voto mayoritario
            # (1 si > 0.5, 0 si < 0.5, y podría ser ambiguo en 0.5)
            mean_signal = np.mean(all_signals, axis=0)
            return np.round(mean_signal).astype(int)
        
        else:
            raise ValueError("Lógica de agregación no soportada.")


class StrategyOrchestrator:
    """
    Orquesta la ejecución de una o más estrategias en el tiempo,
    interactuando con el Exchange para colocar órdenes.
    """
    def __init__(self, strategy: Strategy, exchange: Exchange):
        self.strategy = strategy
        self.exchange = exchange
        self.order_map = {1: 'long', 0: 'short'}
        self.order_size=1000
        self.tpl = strategy.tp[0]
        self.tps = strategy.tp[1]
        self.sll = strategy.sl[0]
        self.sls = strategy.sl[1]
        print("StrategyOrchestrator listo para operar.")

    def run(self, historical_data: pd.DataFrame, features_columns: List[str]):
        """
        Ejecuta la simulación de trading iterando sobre datos históricos.
        """
        print("\n--- Iniciando simulación de trading ---")
        for index, row in historical_data.iterrows():
            # 1. Preparar las características para el modelo en este paso de tiempo
            # Usamos .to_frame().T para convertir la fila (Series) en un DataFrame de una sola fila
            current_features = row[features_columns].to_frame().T
            
            # 2. Obtener la señal de la estrategia
            signal_array = self.strategy.get_signals(current_features)
            signal = signal_array.flatten()[0] # Extraer la señal escalar
            
            # 3. Mapear la señal a una orden y ejecutar
            order_type = self.order_map.get(signal)

            take_profit = self.tpl if order_type == 'long' else self.tps
            stop_loss = self.sll if order_type == 'long' else self.sls
        
            
            if order_type:
                self.exchange.place_order(type=order_type,
                                          timestamp=index,
                                          quantity=self.order_size,
                                          take_profit = take_profit,
                                          stop_loss=stop_loss
                                          )
            else:
                # Si la señal es -1 (neutral) o no reconocida, no hacemos nada.
                print(f"[{index}] Señal neutral. Manteniendo posición actual.")
        
        print("--- Simulación finalizada ---")


