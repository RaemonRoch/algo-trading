import pandas as pd
import numpy as np
from src.core.market import Exchange 

# --- Funciones Auxiliares "Privadas" ---

def _get_returns_and_values(exchange: Exchange) -> tuple[pd.Series, pd.Series]:
    """
    Toma el objeto Exchange y extrae dos Series de pandas:
    1. Retornos diarios del portafolio.
    2. Valores del portafolio.
    Ambos indexados por timestamp.
    """
    # Convertir el historial de balance en un DataFrame
    balance_df = pd.DataFrame(exchange.historical_balance)
    
    if balance_df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
        
    balance_df['timestamp'] = pd.to_datetime(balance_df['timestamp'])
    balance_df = balance_df.set_index('timestamp')
    
    # 1. Obtener la serie de Valores del Portafolio
    portfolio_values = balance_df['portfolio_value']
    
    # 2. Calcular la serie de Retornos
    returns = portfolio_values.pct_change().dropna()
    
    return returns, portfolio_values

def _calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calcula el Max Drawdown (en porcentaje) de una serie de valores de portafolio.
    """
    if portfolio_values.empty:
        return 0.0
        
    # 1. Encontrar el valor máximo acumulado (el "pico" más alto hasta la fecha)
    cumulative_max = portfolio_values.cummax()
    
    # 2. Calcular el drawdown (caída desde el pico)
    drawdown = (portfolio_values - cumulative_max) / cumulative_max
    
    # 3. Encontrar el drawdown máximo (el valor más negativo)
    max_drawdown = drawdown.min()
    return max_drawdown # Nota: esto es un valor negativo (ej. -0.10 para un 10%)

def _calculate_sharpe_ratio(returns: pd.Series, periods_per_year: int, risk_free_rate_annual: float) -> float:
    """Calcula el Sharpe Ratio anualizado."""
    if returns.empty:
        return 0.0
        
    # Tasa libre de riesgo por período
    risk_free_rate_period = (1 + risk_free_rate_annual)**(1/periods_per_year) - 1
    
    excess_returns = returns - risk_free_rate_period
    
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    
    if std_excess_return == 0:
        return np.inf if mean_excess_return > 0 else 0.0 # Evitar división por cero
        
    sharpe_ratio = mean_excess_return / std_excess_return
    
    # Anualizar el ratio
    annual_sharpe = sharpe_ratio * np.sqrt(periods_per_year)
    return annual_sharpe

def _calculate_sortino_ratio(returns: pd.Series, periods_per_year: int, risk_free_rate_annual: float) -> float:
    """Calcula el Sortino Ratio anualizado."""
    if returns.empty:
        return 0.0

    risk_free_rate_period = (1 + risk_free_rate_annual)**(1/periods_per_year) - 1
    excess_returns = returns - risk_free_rate_period
    
    mean_excess_return = excess_returns.mean()
    
    # --- Diferencia clave con Sharpe: Downside Deviation ---
    # Solo nos importan los retornos por debajo de 0 (o del risk-free)
    negative_excess_returns = excess_returns[excess_returns < 0]
    
    if negative_excess_returns.empty:
        return np.inf # No hubo retornos negativos, ratio infinito
        
    downside_std = negative_excess_returns.std()
    
    if downside_std == 0:
        return np.inf
        
    sortino_ratio = mean_excess_return / downside_std
    annual_sortino = sortino_ratio * np.sqrt(periods_per_year)
    return annual_sortino

def _calculate_calmar_ratio(returns: pd.Series, portfolio_values: pd.Series, periods_per_year: int) -> float:
    """Calcula el Calmar Ratio."""
    if returns.empty or portfolio_values.empty:
        return 0.0
        
    # 1. Calcular Retorno Anualizado Total
    num_years = len(returns) / periods_per_year
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    annualized_return = ((1 + total_return) ** (1 / num_years)) - 1
    
    # 2. Calcular Max Drawdown
    max_dd = _calculate_max_drawdown(portfolio_values)
    
    if max_dd == 0:
        return np.inf # Sin drawdown
        
    # Calmar = Retorno Anualizado / Abs(Max Drawdown)
    calmar_ratio = annualized_return / abs(max_dd)
    return calmar_ratio


def generate_performance_report(exchange: Exchange, 
                                trade_pnl_history: list,
                                periods_per_year: int = 252, 
                                risk_free_rate_annual: float = 0.0):
    """
    Genera un reporte completo de métricas de rendimiento
    basado en el estado final de un objeto Exchange.
    """
    
    print("\n--- 📊 Generando Reporte de Métricas de Rendimiento ---")
    
    report = {}
    
    # 1. Obtener datos base
    returns, portfolio_values = _get_returns_and_values(exchange)
    
    if returns.empty or portfolio_values.empty:
        print("Advertencia: No hay datos de balance para generar el reporte.")
        return {}

    # 2. Calcular Métricas de Rendimiento
    report['sharpe_ratio'] = _calculate_sharpe_ratio(returns, periods_per_year, risk_free_rate_annual)
    report['sortino_ratio'] = _calculate_sortino_ratio(returns, periods_per_year, risk_free_rate_annual)
    report['max_drawdown'] = _calculate_max_drawdown(portfolio_values)
    report['calmar_ratio'] = _calculate_calmar_ratio(returns, portfolio_values, periods_per_year)
    
    # 3. Calcular Estadísticas de Trades y Costos
    trades_df = pd.DataFrame(exchange.executed_trades)
    
    if not trades_df.empty:
        report['total_commissions_paid'] = trades_df['commission_paid'].sum()
        report['number_of_trades (legs)'] = len(trades_df)
    else:
        report['total_commissions_paid'] = 0.0
        report['number_of_trades (legs)'] = 0

    report['total_borrow_costs'] = exchange.total_borrow_costs_paid

    if trade_pnl_history:
        pnl_series = pd.Series(trade_pnl_history)
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]
        
        report['number_of_roundtrips'] = len(pnl_series)
        
        if len(pnl_series) > 0:
            report['win_rate'] = len(wins) / len(pnl_series)
        else:
            report['win_rate'] = 0.0
        
        if not wins.empty and not losses.empty:
            report['avg_win_loss_ratio'] = abs(wins.mean() / losses.mean())
            report['profit_factor'] = wins.sum() / abs(losses.sum())
        else:
            report['avg_win_loss_ratio'] = np.inf if not wins.empty else 0.0
            report['profit_factor'] = np.inf if not wins.empty else 0.0
            
        report['avg_win_usd'] = wins.mean()
        report['avg_loss_usd'] = losses.mean()

    else:
        print("Nota: No se encontraron trades 'round-trip' para calcular Win Rate.")
        report['number_of_roundtrips'] = 0
        report['win_rate'] = 0.0
        report['avg_win_loss_ratio'] = 0.0
        report['profit_factor'] = 0.0
        report['avg_win_usd'] = 0.0
        report['avg_loss_usd'] = 0.0
    
    # Limpiar los 'N/A' que ya no usamos
    report.pop('avg_win_loss', None) 
    # ---------------------
    
    return report

