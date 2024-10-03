'''
run below command in the terminal before running this script
export PYTHONPATH="/Users/achillesreaper/Project-Repositoris/dh_backtest/dh-backtest_ver_0_2/core:$PYTHONPATH"
format => export PYTHONPATH="/path/to/your/module:$PYTHONPATH"
'''
import copy
import sys
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

        # add market profile shape column
        df_shape = pd.read_csv('strategy/data/mp_ling/shape/HK_HSImain_shape_2021-01_2024-07.csv')[['trade_date', 'shape']]
        shape_dict = df_shape.set_index('trade_date').to_dict()['shape']
        df_raw['shape'] = df_raw['trade_date'].apply(lambda x: shape_dict[x] if x in shape_dict else None)

        return  df_raw
    

    def generate_signal(self, df_testing:pd.DataFrame, para_comb:dict) -> pd.DataFrame:


        return df_testing


    def action_on_signal(self, df_testing_signal: pd.DataFrame, para_comb:dict) -> pd.DataFrame:
        # 1. initialize the trading, a) create a trade account, b) initialize the trading dataframe
        trade_account = FutureTradingAccount(self.init_capital)
        df_bt_resilt = self.init_trading(df_testing_signal, trade_account)

        # 2. loop through the signals
        for index, row in df_testing_signal.iterrows():
            is_signal_buy   = row['signal'] == 1
            is_signal_sell  = row['signal'] == -1
            is_mtm          = False  # current row is marked to market if open or close position

            # a) determine if it is time to open position
            ''' Strategy: 
            1. if signal is buy and current position long or zero, add a long position
            2. if signal is sell and current position short or zero, add a short position
            3. if the signal direction is opposit to current position -> next step: is close position?
            '''


            # b) determine if it is time to close position
            ''' Strategy:
            1. when the position loss reach the stop loss, close the position
            2. when the margin call, close the position -> this is handled in the mark_to_market function
            '''




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
                if trade_account.position_size > 0 and row['close'] > trade_account.stop_level + para_comb['stop_loss'] + para_comb['ladder']:
                    trade_account.stop_level += para_comb['stop_loss']
                elif trade_account.position_size < 0 and row['close'] < trade_account.stop_level - para_comb['stop_loss'] - para_comb['ladder']:
                    trade_account.stop_level -= para_comb['stop_loss']
            
            # d) record account status in df_bt_resilt
            df_bt_resilt.loc[index,'pos_size']       = trade_account.position_size
            df_bt_resilt.loc[index,'pos_price']      = trade_account.position_price
            df_bt_resilt.loc[index,'pnl_unrealized'] = trade_account.pnl_unrealized
            df_bt_resilt.loc[index,'nav']            = trade_account.bal_equity
            df_bt_resilt.loc[index,'bal_cash']       = trade_account.bal_cash
            df_bt_resilt.loc[index,'bal_avialable']  = trade_account.bal_avialable
            df_bt_resilt.loc[index,'margin_initial'] = trade_account.margin_initial
            df_bt_resilt.loc[index,'cap_usage']      = f'{trade_account.cap_usage:.4f}'

        return df_bt_resilt



if __name__ == "__main__":
    engine = GoldenCrossEnhanceStop(
        folder_path         = "strategy/data/mp_ling",
        summary_mode        = True,
        is_rerun_backtest   = True,
        is_update_data      = True,
        initial_capital     = 500_000,
        underlying  = "HK.HSImain",
        start_date  = "2024-05-01",
        end_date    = "2024-07-30",
        bar_size    = KLType.K_5M,
        para_dict   = {
            'short_window'  : [5],
            'long_window'   : [20],
            'stop_loss'     : [50],
            'ladder'        : [20,30,40],
        }
    )

    bt_result_list = engine.run()
    PlotApp(bt_result_list).plot()