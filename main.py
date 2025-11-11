import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint 
from src.core.metrics import generate_performance_report

from src.core.market import Exchange
from src.core.strategies import MarketNeutralStrategy, StrategyOrchestrator

def run_strategy_backtest():
    """
    Función principal para ejecutar el backtest.
    """
    
    # --- 1. Cargar y Preparar Datos ---
    print("Cargando datos reales de precios...")
    data = pd.read_csv("data/prices.csv")
    
    try:
        # 1. Convertir la columna a datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        # 2. Establecer como índice
        data = data.set_index('timestamp')
        
    except KeyError:
        print("Error: No se encontró la columna 'timestamp'. Revisa tu CSV.")
        return
    # -------------------------

    asset1_col = 'GOOGL'
    asset2_col = 'MSFT'
    features_cols = [asset1_col, asset2_col]
    
    # --- 2. Dividir Datos (Split Cronológico 60/20/20) ---
    n = len(data)
    train_end = int(n * 0.6)
    test_end = int(n * 0.8)
    
    train_data = data.iloc[0:train_end]
    test_data = data.iloc[train_end:test_end]
    validation_data = data.iloc[test_end:]
    
    print(f"Datos divididos: {len(train_data)} train, {len(test_data)} test, {len(validation_data)} validation.")
    
    # --- 3. Inicializar Componentes ---
    
    annual_borrow_rate = 0.0025 
    daily_borrow_rate = annual_borrow_rate / 365.0 
    
    exchange = Exchange(
        asset1_col=asset1_col,
        asset2_col=asset2_col,
        historical_data=data,
        balance=1_000_000,
        fee_rate=0.00125,
        borrow_rate_daily=daily_borrow_rate 
    )

    strategy = MarketNeutralStrategy(
        asset1_col=asset1_col,
        asset2_col=asset2_col,
        rolling_window=30,      # Ventana para Z-score
        entry_threshold=2.0,    # Entrar en 2.0 std dev
        exit_threshold=0.5      # Salir en 0.5 std dev
    )

    orchestrator = StrategyOrchestrator(
        strategy=strategy,
        exchange=exchange,
        capital_allocation_pct=0.4 # 40% por pata
    )

    # --- 4. Ejecutar el Backtest ---
    
    print(f"Ejecutando backtest en el 'Test Set' ({test_data.index.min().date()} a {test_data.index.max().date()})...")
    
    orchestrator.run(
        historical_data=test_data, 
        features_columns=features_cols
    )
    
    print("Backtest finalizado.")

    # Asumimos 252 días de trading al año
    performance_report = generate_performance_report(
        orchestrator.exchange, 
        orchestrator.trade_pnl_history,
        periods_per_year=252, 
        risk_free_rate_annual=0.0
    )

    # Imprimir el reporte
    pprint.pprint(performance_report)

    plot_results(orchestrator, data, test_data.index)


def plot_results(orchestrator: StrategyOrchestrator, all_data: pd.DataFrame, test_index: pd.DatetimeIndex):
    """
    Grafica los resultados clave del backtest.
    """
    print("Generando gráficos de resultados...")
    plt.style.use('seaborn-v0_8-darkgrid')

    # --- Gráfico 1: Curva de Capital (Equity Curve) ---
    results_df = pd.DataFrame(orchestrator.exchange.historical_balance)
    
    if results_df.empty:
        print("Advertencia: No se encontró historial de balance para graficar.")
    else:
        results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
        results_df = results_df.set_index('timestamp')

        plt.figure(figsize=(14, 7))
        plt.plot(results_df['portfolio_value'], label='Valor del Portafolio', color='blue')
        plt.title('Curva de Capital (Equity Curve) - Test Set')
        plt.ylabel('Valor del Portafolio (USD)')
        plt.xlabel('Fecha')
        plt.legend()
        plt.show()

    # --- Gráfico 2: Ratio de Cobertura Dinámico (Hedge Ratio) ---
    hedge_df = pd.DataFrame(orchestrator.strategy.hedge_ratio_history)
    
    if hedge_df.empty:
        print("Advertencia: No se encontró historial de hedge ratio para graficar.")
    else:
        hedge_df['timestamp'] = pd.to_datetime(hedge_df['timestamp'])
        hedge_df = hedge_df.set_index('timestamp')

        plt.figure(figsize=(14, 7))
        plt.plot(hedge_df['slope'], label='Hedge Ratio (Beta_1) Estimado (KF)', color='orange')
        
        plt.title('Evolución del Hedge Ratio Dinámico (Filtro de Kalman)')
        plt.ylabel('Hedge Ratio (Beta_1)')
        plt.xlabel('Fecha')
        plt.legend()
        plt.show()

    # --- Gráfico 3: Evolución del Spread y Z-Score ---
    z_score_df = pd.DataFrame(orchestrator.strategy.z_score_history)
    
    if z_score_df.empty:
        print("Advertencia: No se encontró historial de Z-Score para graficar.")
    else:
        z_score_df['timestamp'] = pd.to_datetime(z_score_df['timestamp'])
        z_score_df = z_score_df.set_index('timestamp')

        spread_history = orchestrator.strategy.spread_history
        spread_series = pd.Series(
            spread_history[len(spread_history)-len(z_score_df):], 
            index=z_score_df.index
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        ax1.plot(spread_series, label='Spread (P1 - (B0 + B1*P2))', color='purple')
        ax1.set_title('Evolución del Spread (Residuos)')
        ax1.set_ylabel('Valor del Spread')
        ax1.legend()
        
        ax2.plot(z_score_df['z_score'], label='Z-Score del Spread', color='green')
        ax2.axhline(orchestrator.strategy.entry_threshold, color='red', linestyle='--', label=f'Entry ({orchestrator.strategy.entry_threshold})')
        ax2.axhline(-orchestrator.strategy.entry_threshold, color='red', linestyle='--')
        ax2.axhline(orchestrator.strategy.exit_threshold, color='blue', linestyle=':', label=f'Exit ({orchestrator.strategy.exit_threshold})')
        ax2.axhline(-orchestrator.strategy.exit_threshold, color='blue', linestyle=':')
        
        ax2.set_title('Z-Score y Señales de Trading')
        ax2.set_ylabel('Z-Score (Std Dev)')
        ax2.set_xlabel('Fecha')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    # --- 4. Estadísticas de Trades ---
    trades_df = pd.DataFrame(orchestrator.exchange.executed_trades)
    print("\n--- Resumen de Trades Ejecutados ---")
    print(f"Número total de trades (patas): {len(trades_df)}")
    if not trades_df.empty:
        print(f"Comisiones totales pagadas: ${trades_df['commission_paid'].sum():.2f}")
    else:
        print("No se ejecutaron trades en este backtest.")


if __name__ == "__main__":
    run_strategy_backtest()