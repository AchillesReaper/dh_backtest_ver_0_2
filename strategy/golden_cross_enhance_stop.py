'''
run below command in the terminal before running this script
export PYTHONPATH="/Users/achillesreaper/Project-Repositoris/dh_backtest/dh-backtest_ver_0_2/core:$PYTHONPATH"
format => export PYTHONPATH="/path/to/your/module:$PYTHONPATH"
'''
import copy
from futu import KLType
import pandas as pd

# local imports
from core.backtest import BacktestEngine
from core.utilities.remote_data import get_stock_futu_api
from core.trading_acc import FutureTradingAccount
from core.visualization import PlotApp

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

class GoldenCrossEnhanceStop(BacktestEngine):
    def get_hist_data(self) -> pd.DataFrame:
        df_raw = get_stock_futu_api(
            underlying  = self.underlying,
            start_date  = self.start_date,
            end_date    = self.end_date,
            ktype       = self.bar_size
        )
        df_raw.drop(columns=['pe_ratio', 'turnover_rate', 'last_close','turnover'], inplace=True)
        return  df_raw
    

    def generate_signal(self, df_testing:pd.DataFrame, para_comb:dict) -> pd.DataFrame:
        short_window = para_comb['short_window']
        long_window  = para_comb['long_window']
        ma_s         = f'ma_{short_window}'
        ma_l         = f'ma_{long_window}'

        df_testing[ma_s]     = df_testing['close'].rolling(window=short_window).mean()
        df_testing[ma_l]     = df_testing['close'].rolling(window=long_window).mean()
        df_testing['signal'] = 0

        prev_index = None
        for index, row in df_testing.iterrows():
            if prev_index is None:
                prev_index = index
                continue
            if row[ma_s] > row[ma_l] and df_testing.loc[prev_index, ma_s] < df_testing.loc[prev_index, ma_l]:
                # golden cross
                df_testing.loc[index, 'signal'] = 1
            elif row[ma_s] < row[ma_l] and df_testing.loc[prev_index, ma_s] > df_testing.loc[prev_index, ma_l]:
                # death cross
                df_testing.loc[index, 'signal'] = -1
            prev_index = index

        return df_testing


    def action_on_signal(self, df_testing_signal: pd.DataFrame, para_comb:dict) -> pd.DataFrame:
        # 1. initialize the trading, a) create a trade account, b) initialize the trading dataframe
        trade_account = FutureTradingAccount(self.init_capital)
        df_bt_resilt = self.init_trading(df_testing_signal, trade_account)

        # 2. loop through the signals
        for index, row in df_testing_signal.iterrows():
            is_signal_buy   = row['signal'] == 1
            is_signal_sell  = row['signal'] == -1
            is_mtm          = True  # current row is marked to market if open or close position

            # a) determine if it is time to open position
            ''' Strategy: 
            1. if signal is buy and current position long or zero, add a long position
            2. if signal is sell and current position short or zero, add a short position
            3. if the signal direction is opposit to current position -> next step: is close position?
            '''
            initial_margin_per_contract = row['close']* trade_account.contract_multiplier * trade_account.margin_rate
            if trade_account.bal_avialable > initial_margin_per_contract:   # check if sufficient cash to open a position
                if is_signal_buy and trade_account.position_size >= 0:
                    t_size     = 1
                    commission = trade_account.open_position(t_size, row['close'])
                    df_bt_resilt.loc[index, 'action']       = 'open'
                    df_bt_resilt.loc[index, 'logic']        = 'signal buy'
                    df_bt_resilt.loc[index, 't_size']       = t_size
                    df_bt_resilt.loc[index, 't_price']      = row['close']
                    df_bt_resilt.loc[index, 'commission']   = commission
                    df_bt_resilt.loc[index, 'pnl_action']   = -commission
                elif is_signal_sell and trade_account.position_size <= 0:
                    t_size     = -1
                    commission = trade_account.open_position(t_size, row['close'])
                    df_bt_resilt.loc[index, 'action']       = 'open'
                    df_bt_resilt.loc[index, 'logic']        = 'signal sell'
                    df_bt_resilt.loc[index, 't_size']       = t_size
                    df_bt_resilt.loc[index, 't_price']      = row['close']
                    df_bt_resilt.loc[index, 'commission']   = commission
                    df_bt_resilt.loc[index, 'pnl_action']   = -commission
            else:
                is_mtm = False

            # b) determine if it is time to close position
            ''' Strategy:
            1. when the position profit reach the target, close the position
            2. when the position loss reach the stop loss, close the position
            3. when the margin call, close the position -> this is handled in the mark_to_market function
            '''
            target_pnl      = para_comb['target_profit']
            stop_loss       = -para_comb['stop_loss']

            if trade_account.position_size != 0:
                contract_pnl    = (row['close'] - trade_account.position_price) * trade_account.position_size
                if contract_pnl >= target_pnl or contract_pnl < stop_loss:
                    t_size     = copy.deepcopy(-trade_account.position_size)
                    t_price    = row['close']
                    commission, pnl_realized = trade_account.close_position(t_size, t_price)
                    df_bt_resilt.loc[index, 'action']       = 'close'
                    df_bt_resilt.loc[index, 'logic']        = 'target profit' if contract_pnl >= target_pnl else 'stop loss'
                    df_bt_resilt.loc[index, 't_size']       = t_size
                    df_bt_resilt.loc[index, 't_price']      = t_price 
                    df_bt_resilt.loc[index, 'commission']   = commission
                    df_bt_resilt.loc[index, 'pnl_action']   = pnl_realized
            else:
                is_mtm = False

            # c) mark profile value to market
            if not is_mtm:
                mtm_res = trade_account.mark_to_market(row['close'])
                if mtm_res['action'] == 'force close':
                    df_bt_resilt.loc[index, 'action']       = mtm_res['action']
                    df_bt_resilt.loc[index, 'logic']        = mtm_res['logic']
                    df_bt_resilt.loc[index, 't_size']       = mtm_res['t_size']
                    df_bt_resilt.loc[index, 't_price']      = mtm_res['t_price']
                    df_bt_resilt.loc[index, 'commission']   = mtm_res['commission']
                    df_bt_resilt.loc[index, 'pnl_action']   = mtm_res['pnl_realized']
            
            # d) record account status in df_bt_resilt
            df_bt_resilt.loc[index,'pos_size']       = trade_account.position_size
            df_bt_resilt.loc[index,'pos_price']      = trade_account.position_price
            df_bt_resilt.loc[index,'pnl_unrealized'] = trade_account.pnl_unrealized
            df_bt_resilt.loc[index,'nav']            = trade_account.bal_equity
            df_bt_resilt.loc[index,'bal_cash']       = trade_account.bal_cash
            df_bt_resilt.loc[index,'bal_avialable']  = trade_account.bal_avialable
            df_bt_resilt.loc[index,'margin_initial'] = trade_account.margin_initial
            df_bt_resilt.loc[index,'cap_usage']      = f'{trade_account.cap_usage:.2f}%'

        return df_bt_resilt



if __name__ == "__main__":
    engine = GoldenCross(
        folder_path         = "strategy/data/golden_cross",
        is_rerun_backtest   = False,
        is_update_data      = False,
        initial_capital     = 150_000,
        underlying  = "HK.HSImain",
        start_date  = "2024-03-01",
        end_date    = "2024-08-30",
        bar_size    = KLType.K_30M,
        para_dict   = {
            'short_window'  : [5],
            'long_window'   : [20],
            'stop_loss'     : [50],
            'target_profit' : [100,150,200],
        }
    )

    bt_result_list = engine.run()
    PlotApp(bt_result_list).plot()