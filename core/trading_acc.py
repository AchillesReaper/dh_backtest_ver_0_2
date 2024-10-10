import copy
import sys
from termcolor import cprint


class FutureTradingAccount():
    def __init__(self, initail_cash_bal: float, margin_rate:float = 0.1, commission_rate:float = 11, contract_multiplier:int = 50):
        self.bal_initial            = initail_cash_bal
        self.bal_cash               = initail_cash_bal          # cash balance
        self.bal_avialable          = initail_cash_bal          # cash available for trading = cash balance - initial margin + unrealized profit and loss
        self.bal_equity             = initail_cash_bal          # total equity(NAV) = cash balance + unrealized profit and loss
        self.pnl_unrealized         = 0                         # unrealized profit and loss
        self.margin_rate            = margin_rate               # margin rate for opening a position
        self.margin_initial         = 0                         # initial margin in $ term
        self.cap_usage              = 0                         # usage of the capital = initial margin / cash balance
        self.margin_maintanence_rate = 0.8                      # margin call level
        self.margin_force_close_rate = 0.6                      # margin force close level
        self.contract_multiplier    = contract_multiplier       # contract multiplier for the future
        self.commission_rate        = commission_rate
        self.position_size          = 0                         # position size -> number of contracts. note: -ve denotes short position
        self.position_price         = 0                         # position price -> the averave price of the current position
        self.stop_level             = 0                         # stop level for the current position

    def mark_to_market(self, mk_price):
        if self.position_size > 0:
            self.pnl_unrealized = (mk_price - self.position_price) * self.position_size * self.contract_multiplier
        elif self.position_size < 0:
            self.pnl_unrealized = (self.position_price - mk_price) * self.position_size * self.contract_multiplier
        else:
            self.pnl_unrealized = 0
            self.position_price = 0

        self.margin_initial = abs(self.position_size) * mk_price * self.contract_multiplier * self.margin_rate
        self.bal_avialable  = self.bal_cash - self.margin_initial + self.pnl_unrealized
        self.bal_equity     = self.bal_cash + self.pnl_unrealized
        self.cap_usage      = round(self.margin_initial / (self.bal_cash + 0.0001), 4)
        
        if self.bal_equity < self.margin_initial * self.margin_maintanence_rate:
            cprint(f"Warning! Margin call: ${self.margin_initial - self.bal_equity}, Margin-level: {(self.bal_equity / self.margin_initial * 100):.2f}%, ", "red")
            return {'signal': 'margin call', 'action': None}
        if self.bal_equity < self.margin_initial * self.margin_force_close_rate:
            cprint(f"Warning! Force Closure!!! \nMargin-level: {(self.bal_equity / self.margin_initial * 100):.2f}%, ", "red")
            t_size = copy.deepcopy(-self.position_size)
            t_price = mk_price
            commission, pnl_realized = self.close_position(t_size, t_price)
            return {'signal': 'margin call', 'action': 'force close', 'logic':'margin call' , 't_size': t_size, 't_price': t_price,'commission': commission, 'pnl_realized': pnl_realized}

        return {'signal': '', 'action': None}

    def open_position(self, t_size:int, t_price:float):
        # new position size shall have the same sign as the current position size
        if t_size == 0 or self.position_size/t_size < 0:
            cprint("Error: New position size is 0 or direction is wrong", "red")
            sys.exit()

        self.position_price  = (self.position_size * self.position_price + t_size * t_price) / (self.position_size + t_size)
        self.position_size  += t_size
        commission           = abs(t_size) * self.commission_rate
        self.bal_cash       -= commission
        self.mark_to_market(t_price)
        return commission


    def close_position(self, t_size:int, t_price:float):
        # assume the t_size comes in with direction => t_size must have the opposite sign of the position size
        if t_size == 0 or self.position_size/t_size > 0:
            cprint("Error: Close position size is 0 or direction is wrong", "red")
            sys.exit()

        self.position_size  += t_size
        commission           = abs(t_size) * self.commission_rate

        pnl_realized = (self.position_price - t_price) * t_size * self.contract_multiplier - commission

        self.bal_cash += pnl_realized
        self.mark_to_market(t_price)
        return commission, pnl_realized
        
