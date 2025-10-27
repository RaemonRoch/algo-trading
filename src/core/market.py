import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Exchange:
    """Almacena el estado y el historial del portafolio."""
    initial_balance: float = 1_000_000
    fee_rate: float = 0.001  # 0.1% fee
    balance: float = field(init=False)

    data = pd.read_csv("../data/Binance_BTCUSDT_1h.csv")

    historical_balance: list = field(default_factory=list)
    active_orders: list = field(default_factory=list)
    execute_orders: list = field(default_factory=list)


    def record_state(self, timestamp: any, portfolio_value: float):
        """Guarda una instantánea del valor del portafolio en un momento dado."""
        self.historical_balance.append({
            "timestamp": timestamp,
            "portfolio_value": portfolio_value
        })

    def get_history_df(self) -> pd.DataFrame:
        """Devuelve el historial del portafolio como un DataFrame."""
        return pd.DataFrame(self.history).set_index("timestamp")
    
    def get_last_price(self, timestamp: any) -> float:
        """Obtiene el último precio conocido del activo."""
        row = self.data[self.data['timestamp'] == timestamp]
        if not row.empty:
            return row['close'].values[0]
        else:
            raise ValueError("Timestamp not found in data.")

    def place_order(self, timestamp:int , type:str, quantity:float, 
                    take_profit:float=None,
                    stop_loss:float=None):
        """Coloca una orden en el exchange."""
        self.active_orders.append({
            "timestamp": timestamp,
            "type": type,
            "quantity": quantity,
            "price": self.get_last_price(timestamp),
            "take_profit": take_profit,
            "stop_loss": stop_loss
        })

    def _check_order(self, timestamp: any, order: dict):
        """Verifica y cierra órdenes (TP/SL) basadas en el precio actual."""



    def check_orders_execution(self, timestamp: any, current_price: float):
            """
            Verifica y cierra órdenes (TP/SL) basadas en el precio actual.
            Asume que todos los cierres son órdenes de mercado.
            """
            
            # Lista temporal para órdenes que se cerrarán en esta vela
            orders_to_close = [] 
            
            for order in self.active_orders:
                difference = (current_price / order['price']) - 1
                should_close = False
                
                if order['type'] == 'long':
                    # Lógica para CERRAR un LONG
                    # 1. Take Profit: El precio actual supera o iguala el TP
                    if order['take_profit'] is not None and difference >= order['take_profit']:
                        should_close = True
                        
                    # 2. Stop Loss: El precio actual cae por debajo o iguala el SL
                    elif order['stop_loss'] is not None and difference <= order['stop_loss']:
                        should_close = True
                
                elif order['type'] == 'short':
                    # Lógica para CERRAR un SHORT
                    # 1. Take Profit: El precio actual cae por debajo o iguala el TP
                    if order['take_profit'] is not None and difference <= order['take_profit']:
                        should_close = True
                    # 2. Stop Loss: El precio actual supera o iguala el SL
                    elif order['stop_loss'] is not None and difference >= order['stop_loss']:
                        should_close = True

                # Si se cumple una condición, se marca para cerrar
                if should_close:
                    orders_to_close.append(order)

            # --- Procesamiento de órdenes cerradas ---
            # Iteramos sobre la lista temporal para modificar las listas principales
            for order in orders_to_close:
                
                # 1. Añadimos la información de cierre (¡tu requisito!)
                order['timestamp_close'] = timestamp
                order['price_close'] = current_price

                # 2. CORRECCIÓN: Usamos .append() para añadir a la lista de ejecutadas
                self.execute_orders.append(order)
                
                # 3. Quitamos la orden de la lista de activas
                self.active_orders.remove(order)