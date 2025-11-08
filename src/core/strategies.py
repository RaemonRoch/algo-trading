from typing import List
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from pykalman import KalmanFilter

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
class MarketNeutralStrategy(Strategy):
    """
    Implementa una estrategia de trading de pares (pairs trading) market-neutral
    utilizando un Filtro de Kalman para estimar el 'dynamic hedge ratio'.

    Esta estrategia ES STATEFUL (mantiene estado). Mantiene el estado del 
    Filtro de Kalman y un historial del spread internamente.
    """
    def __init__(self, 
                 asset1_col: str, 
                 asset2_col: str,
                 rolling_window: int = 30,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 tp: list = None,  # Heredado para compatibilidad
                 sl: list = None): # Heredado para compatibilidad
        
        print("MarketNeutralStrategy inicializada.")
        self.asset1_col = asset1_col
        self.asset2_col = asset2_col
        self.rolling_window = rolling_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.spread_history = []
        self.hedge_ratio_history = [] # Para guardar el historial de betas
        self.z_score_history = []
        
        # Para 'tp' y 'sl' (para compatibilidad con el Orchestrator)
        # Nota: Esta estrategia usa 'exit_threshold' para cerrar, no TP/SL.
        self.tp = tp or [0.05, 0.05] 
        self.sl = sl or [-0.02, -0.02]

        # --- Configuración del Filtro de Kalman (KF1) para el Hedge Ratio ---
        # El objetivo es modelar: P1 = beta_0 + beta_1 * P2 + error
        # El ESTADO del filtro es: [beta_0 (intercept), beta_1 (slope)]
        
        # Matriz de Transición (F): Asumimos un random walk para los betas.
        # [beta_0, beta_1]_t = [1, 0] * [beta_0, beta_1]_{t-1} + error
        #                       [0, 1]
        transition_matrix = np.eye(2) 
        
        # Covarianza de Transición (Q): Cuánto ruido/cambio esperamos en los betas.
        transition_covariance = np.eye(2) * 1e-5 # Pequeño cambio

        # Covarianza de Observación (R): Cuánto ruido hay en la medición (P1).
        observation_covariance = 1.0 

        # Estado inicial (media y covarianza)
        initial_state_mean = np.array([0.0, 1.0]) # Empezamos con [intercept=0, slope=1]
        initial_state_covariance = np.eye(2) * 1.0

        self.kf_hedge = KalmanFilter(
            transition_matrices=transition_matrix,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            n_dim_obs=1,  # Observamos 1 variable: P1
            n_dim_state=2 # El estado tiene 2 variables: beta_0, beta_1
        )
        
        # Guardamos el estado actual para la siguiente iteración
        self.current_state_mean = initial_state_mean
        self.current_state_covariance = initial_state_covariance

    def get_signals(self, features: pd.DataFrame) -> np.ndarray:
        """
        Actualiza el filtro de Kalman y genera una señal de trading.
        
        'features' debe ser un DataFrame de 1 fila con los precios actuales.
        
        Señales generadas:
         1: Comprar Spread (Long P1, Short P2)
         0: Vender Spread (Short P1, Long P2)
        -1: Mantener / Neutral (No hacer nada)
        -2: Cerrar Posición (Reversión a la media)
        """
        
        # 1. Extraer precios del DataFrame (que es 1 fila)
        try:
            # P1 es la variable dependiente (Y)
            p1 = features[self.asset1_col].values[0] 
            # P2 es la variable independiente (X)
            p2 = features[self.asset2_col].values[0]
        except (KeyError, IndexError):
            print("Error: 'features' no contiene las columnas de assets esperadas.")
            return np.array([-1]) # Señal neutral

        # 2. --- Actualización del Filtro de Kalman (Hedge Ratio) ---
        
        # Matriz de Observación (H): Cambia en cada paso del tiempo
        # P1_t = [1, P2_t] * [beta_0, beta_1]_t + error_medicion
        observation_matrix = np.array([[1, p2]])
        
        # Medición (Z)
        observation = np.array([p1])

        # Actualizar el filtro (paso de "filter" + "update")
        # Esto calcula el nuevo estado [beta_0, beta_1] para el tiempo 't'
        self.current_state_mean, self.current_state_covariance = \
            self.kf_hedge.filter_update(
                self.current_state_mean,
                self.current_state_covariance,
                observation=observation,
                observation_matrix=observation_matrix
            )
        
        # 3. --- Generación de Señal (Z-Score) ---
        intercept, slope = self.current_state_mean
        
        # Calcular el spread (residual) actual usando el hedge ratio dinámico
        spread = p1 - (intercept + slope * p2)
        
        # Guardar historial del spread para el Z-score
        self.spread_history.append(spread)
        if len(self.spread_history) > self.rolling_window:
            self.spread_history.pop(0) # Mantener el tamaño de la ventana
        
        # Necesitamos suficientes datos para el Z-score
        if len(self.spread_history) < self.rolling_window:
            return np.array([-1]) # Neutral

        # Calcular Z-score
        mean_spread = np.mean(self.spread_history)
        std_spread = np.std(self.spread_history)
        
        if std_spread < 1e-6: # Evitar división por cero
            return np.array([-1]) # Neutral
                
        z_score = (spread - mean_spread) / std_spread
        
        # 4. --- Lógica de Decisión (Señal) ---
        signal = -1 # Neutral por defecto

        if z_score < -self.entry_threshold:
            signal = 1  # Long Spread (Comprar P1, Vender P2)
        elif z_score > self.entry_threshold:
            signal = 0  # Short Spread (Vender P1, Comprar P2)
        elif abs(z_score) < self.exit_threshold:
            # Si la señal estaba en 1 o 0, esta señal -2 indica "Cerrar"
            signal = -2 
        
        # Retornamos un array de numpy como lo espera el Orchestrator
        return np.array([signal])        

class StrategyOrchestrator:
    """
    Orquesta la ejecución de una estrategia de trading de pares,
    interactuando con el Exchange para colocar órdenes de dos activos.
    ES STATEFUL.
    """
    def __init__(self, 
                 strategy: MarketNeutralStrategy, # ¡Específico!
                 exchange: Exchange,
                 capital_allocation_pct: float = 0.4): # 40% por "pata"
        
        if not isinstance(strategy, MarketNeutralStrategy):
            raise TypeError("StrategyOrchestrator requiere una MarketNeutralStrategy.")
            
        self.strategy = strategy
        self.exchange = exchange
        
        # Usamos el 40% del requerimiento "40% for each asset" como 
        # la asignación de capital para la PATA PRIMARIA (Asset 1).
        # La Pata 2 (Asset 2) se dimensionará por el hedge_ratio.
        self.capital_allocation_pct = capital_allocation_pct
        
        # Estado de la posición
        self.position_status = 'neutral' # 'neutral', 'long_spread', 'short_spread'
        
        # Nombres de las columnas de los assets
        self.asset1 = strategy.asset1_col
        self.asset2 = strategy.asset2_col
        
        print("StrategyOrchestrator listo para operar (modo Market-Neutral).")

    def run(self, historical_data: pd.DataFrame, features_columns: List[str]):
        """
        Ejecuta la simulación de trading iterando sobre datos históricos.
        """
        print("\n--- Iniciando simulación de trading (Market-Neutral) ---")
        
        # Asegurarnos que las columnas de los assets están en los features
        if self.asset1 not in features_columns or self.asset2 not in features_columns:
            raise ValueError("Las columnas de assets de la estrategia no están en features_columns.")

        # Iteramos sobre el índice (timestamp) y la fila (row)
        for timestamp, row in historical_data.iterrows():
            
            # 1. Preparar las características (precios actuales)
            current_features = row[features_columns].to_frame().T
            current_features.index = [timestamp] # Asegurar que el índice sea el correcto
            
            # 2. Obtener la señal de la estrategia
            # NOTA: ¡Este paso TAMBIÉN actualiza el Filtro de Kalman dentro de la estrategia!
            signal_array = self.strategy.get_signals(current_features)
            signal = signal_array.flatten()[0] # Señal: 1, 0, -1, -2
            
            # 3. Lógica de Ejecución (Stateful)
            
            # --- LÓGICA DE APERTURA ---
            if self.position_status == 'neutral':
                if signal == 1: # Abrir Long Spread
                    self._open_position(timestamp, 'long_spread')
                elif signal == 0: # Abrir Short Spread
                    self._open_position(timestamp, 'short_spread')
            
            # --- LÓGICA DE CIERRE ---
            # Si la estrategia nos dice "Cerrar" (signal -2)
            # Y *actualmente* estamos en una posición...
            elif (self.position_status != 'neutral' and signal == -2):
                self._close_position(timestamp)
            
            # Si la señal es -1 (mantener) o una señal de entrada (0 o 1)
            # cuando ya estamos en posición, no hacemos nada.
            
            # 4. Actualizar el estado del exchange (cálculo de PnL, borrow costs, etc.)
            self.exchange.update_exchange_status(timestamp)

        print("--- Simulación finalizada ---")

    def _open_position(self, timestamp: any, trade_type: str):
        """Abre una nueva posición de spread (long o short)."""
        
        # 1. Obtener precios y hedge ratio
        try:
            price1 = self.exchange.get_last_price(timestamp, self.asset1)
            price2 = self.exchange.get_last_price(timestamp, self.asset2)
        except ValueError:
            print(f"[{timestamp}] No se encontraron datos de precio. Omitiendo trade.")
            return

        # Obtenemos el hedge ratio (slope) MÁS RECIENTE del filtro de Kalman
        hedge_ratio = self.strategy.current_state_mean[1] # beta_1 (slope)

        # 2. Calcular tamaño de la posición (Sizing)
        # Usamos el 40% del *balance actual* para la Pata 1
        capital_for_leg1 = self.exchange.balance * self.capital_allocation_pct
        
        # Unidades para la Pata 1
        qty1_units = capital_for_leg1 / price1
        
        # Unidades para la Pata 2 (¡EL DYNAMIC HEDGE!)
        qty2_units = qty1_units * hedge_ratio
        
        # 3. Ejecutar trades
        if trade_type == 'long_spread':
            # Señal 1: Comprar P1 (long), Vender P2 (short)
            self.exchange.execute_trade(timestamp, self.asset1, 'buy', qty1_units)
            self.exchange.execute_trade(timestamp, self.asset2, 'sell', qty2_units)
            self.position_status = 'long_spread'
        
        elif trade_type == 'short_spread':
            # Señal 0: Vender P1 (short), Comprar P2 (long)
            self.exchange.execute_trade(timestamp, self.asset1, 'sell', qty1_units)
            self.exchange.execute_trade(timestamp, self.asset2, 'buy', qty2_units)
            self.position_status = 'short_spread'

    def _close_position(self, timestamp: any):
        """Cierra cualquier posición de spread abierta."""
        
        # Obtenemos las posiciones actuales en unidades
        pos1_units = self.exchange.positions[self.asset1]
        pos2_units = self.exchange.positions[self.asset2]
        
        if pos1_units == 0 and pos2_units == 0:
            return # No hay nada que cerrar

        # Para cerrar, hacemos la operación opuesta con la cantidad exacta que tenemos
        
        # Cerrar Pata 1
        trade_type_1 = 'sell' if pos1_units > 0 else 'buy'
        self.exchange.execute_trade(timestamp, self.asset1, trade_type_1, abs(pos1_units))
        
        # Cerrar Pata 2
        trade_type_2 = 'sell' if pos2_units > 0 else 'buy'
        self.exchange.execute_trade(timestamp, self.asset2, trade_type_2, abs(pos2_units))

        self.position_status = 'neutral'