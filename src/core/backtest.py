import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import copy

from src.core.strategies import StrategyOrchestrator
from src.core.market import Exchange


class BackTesting:
    """
    Clase para ejecutar un backtest de una estrategia de trading usando K-Fold cross-validation.
    """
    def __init__(self, 
                 historical_data: pd.DataFrame,
                 orchestrator: StrategyOrchestrator,
                 exchange_init: Exchange):
        """
        Inicializador de la clase BackTesting.
        
        Args:
            historical_data (pd.DataFrame): DataFrame completo con los datos históricos.
            orchestrator (StrategyOrchestrator): Objeto que ejecuta la lógica de la estrategia.
            exchange_init (Exchange): Una instancia inicial del Exchange (con el capital inicial).
                                      Se usará como plantilla para cada fold.
        """
        self.historical_data = historical_data
        self.exchange_init_template = exchange_init
        self.orchestrator = orchestrator
        self.performance_per_fold = {}

    def run_backtest(self, 
                     n_splits: int = 5,
                     features_columns: list = None):
        """
        Ejecuta el backtest particionando la data usando KFold.
        
        Args:
            n_splits (int): El número de folds a crear (por defecto 5).
            features_columns (list): Lista de nombres de las columnas a usar como features.
        """
        # Usamos KFold de sklearn. shuffle=False es CRUCIAL para series de tiempo.
        kf = KFold(n_splits=n_splits, shuffle=False)
        
        print(f"🚀 Iniciando backtest con {n_splits} folds...")
        
        # enumerate nos da un índice para cada fold (0, 1, 2, ...)
        for fold_idx, (train_index, test_index) in enumerate(kf.split(self.historical_data)):
            
            # 1. Obtenemos los datos para este fold específico
            fold_data = self.historical_data.iloc[test_index].copy()
            
            # 2. Reiniciamos el estado del exchange para asegurar que cada fold comience igual.
            # Usamos deepcopy para crear un objeto completamente nuevo y no una referencia.
            self.orchestrator.exchange = copy.deepcopy(self.exchange_init_template)
            
            print(f"   -> Ejecutando Fold {fold_idx + 1}/{n_splits} con {len(fold_data)} registros...")
            
            # 3. Ejecutamos la estrategia en los datos del fold
            self.orchestrator.run(historical_data=fold_data,
                                  features_columns=features_columns)
            
            # 4. Guardamos el resultado (DataFrame del performance) en nuestro diccionario
            self.performance_per_fold[fold_idx] = self.orchestrator.exchange.historical_balance
        
        print("✅ Backtest completado.")

    def plot_performance_per_fold(self):
        """
        Crea un gráfico con n subplots (uno por cada fold) mostrando el rendimiento
        del portafolio a lo largo del tiempo.
        """
        if not self.performance_per_fold:
            print("No hay datos para graficar. Por favor, ejecuta `run_backtest()` primero.")
            return
        
        num_folds = len(self.performance_per_fold)
        # Creamos una figura con un subplot por cada fold
        fig, axes = plt.subplots(nrows=num_folds, ncols=1, figsize=(12, 3 * num_folds), sharex=True)
        
        fig.suptitle('Rendimiento del Portafolio por Fold', fontsize=16, y=0.95)

        for fold_num, performance_df in self.performance_per_fold.items():
            ax = axes[fold_num]
            ax.plot(performance_df['timestamp'], performance_df['portfolio_value'], label=f'Fold {fold_num + 1}')
            ax.set_ylabel('Valor del Portafolio')
            ax.set_title(f'Fold {fold_num + 1}')
            ax.grid(True, linestyle='--', alpha=0.6)
            
        axes[-1].set_xlabel('Timestamp')
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Ajusta para que el título no se superponga
        plt.show()