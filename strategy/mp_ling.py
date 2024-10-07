'''
run below command in the terminal before running this script
export PYTHONPATH="/Users/achillesreaper/Project-Repositoris/dh_backtest/dh-backtest_ver_0_2/core:$PYTHONPATH"
format => export PYTHONPATH="/path/to/your/module:$PYTHONPATH"
'''
import copy
import os
import sys
from futu import KLType
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from termcolor import cprint


# local imports
from core.backtest import BacktestEngine
from core.utilities.remote_data import get_stock_futu_api
from core.trading_acc import FutureTradingAccount
from core.visualization import PlotApp

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

class GoldenCrossEnhanceStop(BacktestEngine):
    def combine_rows(self, df_td:pd.DataFrame, start_time:str, end_time:str) -> pd.DataFrame:
        mask_start = pd.to_datetime(start_time).time()
        mask_end = pd.to_datetime(end_time).time()
        mask = df_td.between_time(mask_start, mask_end).copy()
        if not mask.empty:
            df_td = df_td.drop(mask.index)
            df_td.loc[mask.index[len(mask)-1]] = {
                'open'      : mask['open'].iloc[0],
                'high'      : mask['high'].max(),
                'low'       : mask['low'].min(),
                'close'     : mask['close'].iloc[-1],
                'volume'    : mask['volume'].sum(),
                'trade_date': mask['trade_date'].iloc[0],
            }

        return df_td

    def generate_mp(self) -> None:
        '''
        This function generates the market profile shape for HSI with 
        30 minutes bar, 
        entension trading hours bin into 2 bins,  17:00 to 21:29 and 21:30 to 02:30 
        '''
        df_mp_raw = get_stock_futu_api(
            underlying  = self.underlying,
            start_date  = self.start_date,
            end_date    = self.end_date,
            ktype       = KLType.K_30M
        )
        df_mp_raw = df_mp_raw[['datetime', 'open', 'high', 'low', 'close', 'volume', 'trade_date']]
        df_mp_raw['datetime'] = df_mp_raw['datetime'].apply(lambda x: x+'+08:00')
        df_mp_raw.set_index('datetime', inplace=True)
        df_mp_raw.index = pd.to_datetime(df_mp_raw.index)

        df_mp_clean = df_mp_raw.iloc[0:0]
        for td in df_mp_raw['trade_date'].unique():
            df_td = df_mp_raw[df_mp_raw['trade_date'] == td]
            # combin the rows btw 17:00 to 21:29 into one
            df_td = self.combine_rows(df_td, '17:00', '21:29')
            # combin the rows btw 21:30 to 03:30 into one
            df_td = self.combine_rows(df_td, '21:30', '04:00')
            df_mp_clean = df_mp_clean._append(df_td)
        # print(df_mp_clean)
        # sys.exit()
        # work today
        trade_date_list     = df_mp_clean["trade_date"].unique()
        open_price_list     = []
        high_price_list     = []
        low_price_list      = []
        close_price_list    = []
        volume_list         = []
        skewness_list       = []
        kurtosis_list       = []
        val_list            = []
        vah_list            = []
        spike_lower_list    = []
        spike_upper_list    = []
        pocs_list           = []
        tpo_count_list      = []
        for td in trade_date_list:
            df_td = df_mp_clean[df_mp_clean["trade_date"] == td]
            # summarize the trade date data
            open_price_list.append(df_td["open"].iloc[0])
            high_price_list.append(df_td["high"].max())
            low_price_list.append(df_td["low"].min())
            close_price_list.append(df_td["close"].iloc[-1])
            volume_list.append(df_td["volume"].sum())
            
            # calculate the market profile varibales
            td_tpo_dict = {}
            price_record = []
            
            # td_tpo_dict_start   = (df_td['low'].min()/5).__ceil__() * 5
            td_tpo_dict_start   = int(df_td['low'].min() / 5) * 5
            td_tpo_dict_end     = (df_td['high'].max()/5).__ceil__() * 5

            for tag in range(td_tpo_dict_start, td_tpo_dict_end+1, 5):
                td_tpo_dict[tag] = 0

            for index, row in df_td.iterrows():
                price_record += list(range(int(row['low']), int(row['high'])+1))
                for tag in td_tpo_dict:
                    if tag >= row['low'] and tag <= row['high']+4:
                        td_tpo_dict[tag] += 1

            del td_tpo_dict[td_tpo_dict_start]
            tpo_count_list.append(td_tpo_dict)

            # for pocs[], VAL[], VAH[], and skewness[]
            if len(td_tpo_dict) > 0:
                td_poc_count = max(td_tpo_dict.values())
                td_poc = [
                    price for price, count in td_tpo_dict.items() if count == td_poc_count
                ]

                tpo_stdev   = int(np.std(price_record, ddof=1))
                mode_price  = int(np.mean(td_poc))
                val         = mode_price - tpo_stdev
                vah         = mode_price + tpo_stdev

                spkl = td_tpo_dict_start
                for price in td_tpo_dict.keys():
                    if (price < val) & (td_tpo_dict[price] <= 2):
                        spkl = price
                    else:
                        break

                spkh = td_tpo_dict_end
                for price in sorted(td_tpo_dict.keys(), reverse=True):
                    if (price > vah) & (td_tpo_dict[price] <=2):
                        spkh = price
                    else:
                        break

                td_skew = round(skew(price_record), 4)
                td_kurt = round(kurtosis(price_record), 4)

                pocs_list.append(td_poc)
                val_list.append(val)
                vah_list.append(vah)
                spike_lower_list.append(spkl)
                spike_upper_list.append(spkh)
                skewness_list.append(td_skew)
                kurtosis_list.append(td_kurt)
            else:
                pocs_list.append([])
                val_list.append(None)
                vah_list.append(None)
                skewness_list.append(None)
                kurtosis_list.append(None)

        df_mp = pd.DataFrame(
            {
                "trade_date": trade_date_list,
                "open"      : open_price_list,
                "high"      : high_price_list,
                "low"       : low_price_list,
                "close"     : close_price_list,
                "volume"    : volume_list,
                "skewness"  : skewness_list,
                "kurtosis"  : kurtosis_list,
                "val"       : val_list,
                "vah"       : vah_list,
                "spkl"      : spike_lower_list,
                "spkh"      : spike_upper_list,
                "pocs"      : pocs_list,
                "tpo_count" : tpo_count_list,
            },
        )

        # add market profile shape column
        df_shape = pd.read_csv('strategy/data/mp_ling/shape/HK_HSImain_shape_2021-01_2024-07.csv')[['trade_date', 'shape']]
        shape_dict = df_shape.set_index('trade_date').to_dict()['shape']
        df_mp['shape'] = df_mp['trade_date'].apply(lambda x: shape_dict[x] if x in shape_dict else None)

        mp_file_name = f'{self.file_name}_mp_{self.start_date}_{self.end_date}.csv'
        if not os.path.exists(f'{self.folder_path}/shape'): os.makedirs(f'{self.folder_path}/shape')
        df_mp.to_csv(f'{self.folder_path}/shape/{mp_file_name}', index=False)
        cprint(f'Market Profile data saved to {self.folder_path}/shape/{mp_file_name}', 'green')
        return None
        

    def get_hist_data(self) -> pd.DataFrame:
        df_raw = get_stock_futu_api(
            underlying  = self.underlying,
            start_date  = self.start_date,
            end_date    = self.end_date,
            ktype       = self.bar_size
        )
        df_raw.drop(columns=['pe_ratio', 'turnover_rate', 'last_close','turnover'], inplace=True)

        # trim the raw data frame, only sessions between 9:00 to 11:00 are left
        df_raw['datetime'] = df_raw['datetime'].apply(lambda x: x+'+08:00')
        df_raw.set_index('datetime', inplace=True)
        df_raw.index = pd.to_datetime(df_raw.index)
        df_raw = df_raw.between_time('09:00', '11:00')

        # match the trade date with market profile data
        df_mp = pd.read_csv(f'{self.folder_path}/shape/{self.file_name}_mp_{self.start_date}_{self.end_date}.csv', index_col=0)
        df_raw = df_raw.merge(df_mp[['skewness', 'kurtosis', 'val', 'vah', 'spkl', 'spkh', 'shape', 'pocs', 'tpo_count']], how='left', on='trade_date')
        
        return  df_raw
    

    def generate_signal(self, df_testing:pd.DataFrame, para_comb:dict) -> pd.DataFrame:
        # for index, row in df_testing.iterrows():
        print(df_testing.head(10))
        sys.exit()
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
        initial_capital     = 500_000,
        underlying  = "HK.HSImain",
        start_date  = "2024-01-01",
        end_date    = "2024-01-30",
        bar_size    = KLType.K_5M,
        para_dict   = {
            'short_window'  : [5],
            'long_window'   : [20],
            'stop_loss'     : [50],
            'ladder'        : [20],
        },
        folder_path         = "strategy/data/mp_ling",
        is_rerun_backtest   = True,
        is_update_data      = True,
        summary_mode        = True,
        multi_process_mode  = False,
    )
    if engine.is_update_data: engine.generate_mp()
    bt_result_list = engine.run()
    # PlotApp(bt_result_list).plot()

