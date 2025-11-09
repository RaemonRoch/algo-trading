import pandas as pd
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Exchange:
    """
    Almacena el estado y el historial del portafolio para una estrategia
    de trading de pares (market-neutral).
    """
    asset1_col: str
    asset2_col: str
    historical_data: pd.DataFrame  # ¡NUEVO! Los datos se inyectan.
    
    balance: float = 1_000_000
    fee_rate: float = 0.00125      # Requerimiento: 0.125%
    borrow_rate_daily: float = 0.0025  # Requerimiento: 0.25%
    total_borrow_costs_paid: float = 0.0
    
    # --- Atributos de Estado ---
    positions: dict = field(default_factory=dict)
    historical_balance: list = field(default_factory=list)
    executed_trades: list = field(default_factory=list)
    last_timestamp: any = None

    def __post_init__(self):
        """Inicializa el diccionario de posiciones después de crear la instancia."""
        self.positions = {self.asset1_col: 0.0, self.asset2_col: 0.0}
        
        # Asumimos que el índice del DataFrame son los timestamps
        if not isinstance(self.historical_data.index, pd.DatetimeIndex):
            print("Advertencia: El índice de historical_data no es DatetimeIndex.")
            # Si tu índice no es 'Date', ajusta esto.
            # Por ahora, asumiremos que el 'timestamp' que nos pasan
            # se puede usar para localizar en el índice.

    def record_state(self, timestamp: any, portfolio_value: float):
        """Guarda una instantánea del valor del portafolio en un momento dado."""
        self.historical_balance.append({
            "timestamp": timestamp,
            "portfolio_value": portfolio_value
        })

    def get_last_price(self, timestamp: any, asset_col: str) -> float:
        """Obtiene el último precio de los datos para un activo específico."""
        try:
            return self.historical_data.loc[timestamp, asset_col]
        except KeyError:
            raise ValueError(f"Timestamp {timestamp} o asset {asset_col} no encontrados en data.")

    def execute_trade(self, timestamp: any, asset_col: str, trade_type: str, quantity_units: float):
        """
        Ejecuta una orden de mercado (compra o venta) para un activo específico.
        'quantity_units' es en unidades del activo (ej. 10 shares).
        """
        if quantity_units <= 0:
            return  # No hacer nada si la cantidad es cero
            
        price = self.get_last_price(timestamp, asset_col)
        trade_value_usd = quantity_units * price
        commission = trade_value_usd * self.fee_rate
        
        # Deducir comisión del balance EN TODAS LAS OPERACIONES
        self.balance -= commission

        if trade_type == 'buy':
            # Comprar: Deducir costo (USD) del balance, Aumentar posición (unidades)
            self.balance -= trade_value_usd
            self.positions[asset_col] += quantity_units
        
        elif trade_type == 'sell':
            # Vender: Aumentar balance (USD), Reducir posición (unidades)
            self.balance += trade_value_usd
            self.positions[asset_col] -= quantity_units
        
        else:
            raise ValueError(f"Tipo de trade '{trade_type}' no reconocido.")

        self.executed_trades.append({
            "timestamp": timestamp,
            "asset": asset_col,
            "type": trade_type,
            "quantity_units": quantity_units,
            "price": price,
            "trade_value_usd": trade_value_usd,
            "commission_paid": commission
        })

    def update_exchange_status(self, timestamp: any):
        """
        Actualiza el estado del exchange:
        1. Calcula y aplica costos de préstamo (borrow costs) diarios.
        2. Calcula el valor total del portafolio.
        3. Registra el estado (historial de balance).
        
        NOTA: Asume que cada llamada es un 'día' para el borrow_rate_daily.
        Si tus datos son horarios, deberías ajustar la tasa (rate / 24).
        """
        
        # --- 1. Aplicar Costos de Préstamo (Borrow Costs) ---
        short_value_usd = 0.0
        price1 = self.get_last_price(timestamp, self.asset1_col)
        price2 = self.get_last_price(timestamp, self.asset2_col)
        
        if self.positions[self.asset1_col] < 0:
            # Posición corta en asset1. Valor es negativo, lo hacemos positivo.
            short_value_usd += abs(self.positions[self.asset1_col] * price1)
            
        if self.positions[self.asset2_col] < 0:
            # Posición corta en asset2
            short_value_usd += abs(self.positions[self.asset2_col] * price2)
        
        if short_value_usd > 0:
            # Asumimos que los datos son DIARIOS según el requerimiento.
            borrow_cost = short_value_usd * self.borrow_rate_daily
            self.balance -= borrow_cost  # Deducir costo del balance
            self.total_borrow_costs_paid += borrow_cost
        
        # --- 2. Calcular Valor del Portafolio ---
        asset1_value = self.positions[self.asset1_col] * price1
        asset2_value = self.positions[self.asset2_col] * price2
        
        portfolio_value = self.balance + asset1_value + asset2_value
        
        # --- 3. Registrar Estado ---
        self.record_state(timestamp, portfolio_value)
        
        # --- 4. Actualizar timestamp para el próximo cálculo de costos ---
        self.last_timestamp = timestamp