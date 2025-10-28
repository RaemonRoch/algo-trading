import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Exchange:
    """Almacena el estado y el historial del portafolio."""
    balance: float = 1_000_000
    fee_rate: float = 0.001  # 0.1% fee
    # Load data
    data = pd.read_csv("../data/Binance_BTCUSDT_1h.csv", header=1)[['Date', 'Close']]
    # Control attributes
    historical_balance: list = field(default_factory=list)
    active_orders: list = field(default_factory=list)
    executed_orders: list = field(default_factory=list)


    def record_state(self, timestamp: any, portfolio_value: float):
        """Guarda una instantánea del valor del portafolio en un momento dado."""
        self.historical_balance.append({
            "timestamp": timestamp,
            "portfolio_value": portfolio_value
        })

    def _check_balace(self, quantity:float):
        """Verify if there is enough balance to place an order."""
        if self.balance >= quantity:
            return True
        else:
            return False
    
    def get_last_price(self, timestamp: any) -> float:
        """Get the last price from data based on timestamp."""
        row = self.data[self.data['Date'] == timestamp]
        if not row.empty:
            return row['Close'].values[0]
        else:
            raise ValueError("Timestamp not found in data.")

    def place_order(self, timestamp:object|str , type:str, quantity:float, 
                    take_profit:float=None, stop_loss:float=None):
        """
        Places an order in the market.

        Parameters:
            timestamp : int
                The time at which the order is placed, represented as a Unix timestamp.
            type : str
                The type of the order (e.g. 'long' or 'short').
            quantity : float
                The amount of the USD to be ordered.
            take_profit : float
                The returns level at which to take profit from the order (e.g. 0.01). Default is None.
            stop_loss : float
                The returns level at which to stop loss on the order. Default is None.
        Returns:
            None
        """
        if self._check_balace(quantity):
            self.balance -= quantity  # Deduct the quantity from balance
            self.active_orders.append({
                "timestamp": timestamp,
                "type": type,
                "quantity": quantity,
                "price": self.get_last_price(timestamp),
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "order_value": quantity
            })

    def _check_order(self, timestamp: any, order: dict):
        current_price = self.get_last_price(timestamp)
        difference = (current_price / order['price']) - 1
        should_close = False
        if order['type'] == 'long':
            # Lógica para CERRAR un LONG
            # 1. Take Profit: El precio actual supera o iguala el TP
            if order['take_profit'] is not None and difference >= order['take_profit']:
                should_close = True
                result = order['quantity'] * (1 + order['take_profit'])
                self.balance += result * (1 - self.fee_rate)
            # 2. Stop Loss: El precio actual cae por debajo o iguala el SL
            elif order['stop_loss'] is not None and difference <= order['stop_loss']:
                should_close = True
                result = order['quantity'] * (1 + order['stop_loss'])
                self.balance += result * (1 - self.fee_rate)
            # Actualizar el valor de la orden
            else:
                order['order_value'] = order['quantity'] * (1 + difference)
            
        
        elif order['type'] == 'short':
            # Lógica para CERRAR un SHORT
            # 1. Take Profit: El precio actual cae por debajo o iguala el TP
            if order['take_profit'] is not None and difference <= order['take_profit']:
                should_close = True
                result = order['quantity'] * (1 - order['take_profit'])
                self.balance += result * (1 - self.fee_rate)
            # 2. Stop Loss: El precio actual supera o iguala el SL
            elif order['stop_loss'] is not None and difference >= order['stop_loss']:
                should_close = True
                result = order['quantity'] * (1 - order['stop_loss'])
                self.balance += result * (1 - self.fee_rate)
            # Actualizar el valor de la orden
            else:
                order['order_value'] = order['quantity'] * (1 - difference)

        if should_close:
            # 1. Añadimos la información de cierre (¡tu requisito!)
            order['timestamp_close'] = timestamp
            order['price_close'] = current_price

            # 2. CORRECCIÓN: Usamos .append() para añadir a la lista de ejecutadas
            self.executed_orders.append(order)
            # 3. Quitamos la orden de la lista de activas
            self.active_orders.remove(order) 
        
    def update_exchange_status(self, timestamp: any):
        """
        Update the engine's active orders for the given timestamp and record the resulting portfolio value.
        This method iterates over a shallow copy of self.active_orders and checks each order
        against the provided timestamp by calling self._check_order(timestamp, order). Using
        a copy allows individual orders to be removed or modified inside _check_order
        without disrupting iteration.
        After processing all orders, the method computes the current portfolio value as:
        and then records the state by calling self.record_state(timestamp, portfolio_value).
        Parameters:
            timestamp : any
                The current time indicator used to evaluate orders (e.g., a datetime, int, or float).
                This value is passed to self._check_order and self.record_state.
        Side effects
            - May modify self.active_orders (orders can be filled, cancelled, or otherwise removed).
            - May modify other mutable state via self._check_order (for example, self.balance or positions).
            - Calls self.record_state(timestamp, portfolio_value) to persist the computed portfolio value.
        Returns:
            None

        Notes:
            - The method assumes orders in self.active_orders are mappings containing an 'order_value' numeric field.
            - Uses a shallow copy (self.active_orders[:]) to permit safe in-loop modifications performed by _check_order.
        """
        for order in self.active_orders[:]:  # Usamos una copia para evitar problemas al eliminar
            self._check_order(timestamp, order)
        
        portfolio_value = self.balance + sum(order['order_value'] for order in self.active_orders)
        self.record_state(timestamp, portfolio_value) 