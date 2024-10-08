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

class MPLing(BacktestEngine):
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
        start_date = pd.Timestamp(self.start_date).__add__(pd.Timedelta(days=-7)).strftime('%Y-%m-%d')
        end_date   = pd.Timestamp(self.end_date).__add__(pd.Timedelta(days=7)).strftime('%Y-%m-%d')   
        df_mp_raw = get_stock_futu_api(
            underlying  = self.underlying,
            start_date  = start_date,
            end_date    = end_date,
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
        df_mp['for_td'] = df_mp['trade_date'].shift(-1)

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
        df_raw.reset_index(inplace=True)

        # match the trade date with market profile data
        df_mp = pd.read_csv(f'{self.folder_path}/shape/{self.file_name}_mp_{self.start_date}_{self.end_date}.csv', index_col=0)
        df_raw = pd.merge(df_raw, df_mp[['for_td', 'skewness', 'kurtosis', 'high', 'low', 'val', 'vah', 'spkl', 'spkh', 'shape', 'pocs', 'tpo_count']], left_on='trade_date', right_on='for_td', how='left')
        df_raw.set_index('datetime', inplace=True)
        df_raw.rename(columns={'high_x':'high', 'low_x':'low', 'high_y':'high_d', 'low_y': 'low_d'}, inplace=True)

        return  df_raw
    

    def is_stand_above(self, td_shape:str, spike_mid_l:float, spike_mid_h:float, ao_loc:str, session_close:float) -> bool:
        if (td_shape not in ['b', 'p', 'n', 'double']) or (ao_loc not in ['u_spk', 'l_spk', 'va', 'poc']): 
            return False
        match td_shape:
            case 'b' | 'p':
                match ao_loc:
                    case 'u_spk':
                            return True if session_close > spike_mid_h else False
                    case 'l_spk':
                            return True if session_close > spike_mid_l else False
                    case _:
                        return False
            case _:
                return False


    def is_signal(self, td_shape:str, spike_mid_l:float, spike_mid_h:float, val:float, vah:float, ao_loc:str, session_close:float, stand_above:bool) -> int:
        match td_shape:
            case 'b':
                match ao_loc:
                    case 'u_spk':
                        if stand_above and (session_close < spike_mid_h + 50):
                            return 1
                        elif not stand_above and (session_close > vah):
                            return -1
                        else:
                            return 0
                    case 'l_spk':
                        if stand_above and (session_close < val):
                            return 1
                        elif not stand_above and (session_close > spike_mid_l - 50):
                            return -1
                        else:
                            return 0
            case 'p':
                match ao_loc:
                    case 'u_spk':
                        if stand_above and (session_close > spike_mid_h):
                            return 1
                        elif not stand_above and (session_close > spike_mid_h - 50):
                            return -1
                        else:
                            return 0
                    case 'l_spk':
                        if stand_above and (session_close < val):
                            return 1
                        elif not stand_above and (session_close > spike_mid_l - 50):
                            return -1
                        else:
                            return 0

            case _:
                return 0

    def generate_signal(self, df_testing:pd.DataFrame, para_comb:dict) -> pd.DataFrame:
        df_testing['signal'] = 0
        for td in df_testing['trade_date'].unique():
            df_td = df_testing[df_testing['trade_date'] == td]
            ao_loc          = 'na'      # 'na', 'u_spk', 'l_spk', 'va', 'poc
            td_shape        = df_td['shape'].iloc[0]
            spike_mid_l     = (df_td['spkl'].iloc[0] + df_td['low_d'].iloc[0]) / 2
            spike_mid_h     = (df_td['spkh'].iloc[0] + df_td['high_d'].iloc[0]) / 2

            for index, row in df_td.iterrows():
                index_time = pd.Timestamp(index).time()
                if index_time == pd.Timestamp('09:20').time():
                    if row['open'] >= row['vah'] and row['open'] <= row['high_d']:
                        ao_loc = 'u_spk'
                    elif row['open'] <= row['val'] and row['open'] >= row['low_d']:
                        ao_loc = 'l_spk'
                    else:
                        ao_loc = 'na'
                    df_testing.loc[df_td.index[0]:df_td.index[len(df_td)-1], 'ao_loc'] = ao_loc
                    if ao_loc == 'na': break
                elif index_time == pd.Timestamp('09:25').time():
                    pass
                elif index_time == pd.Timestamp('09:30').time():
                    stand_above = self.is_stand_above(td_shape, spike_mid_l, spike_mid_h, ao_loc, row['close'])
                    df_testing.loc[index, 'stand_above'] = stand_above
                    df_testing.loc[index, 'signal'] = self.is_signal(td_shape, spike_mid_l, spike_mid_h, row['val'], row['vah'], ao_loc, row['close'], stand_above)
                    # cprint(f'index_time: {index}, break_through:{stand_above}, signal:{df_testing.loc[index, 'signal']}', 'blue')
                else:
                    break
        return df_testing


    def action_on_signal(self, df_testing_signal: pd.DataFrame, para_comb:dict) -> pd.DataFrame:
        # 1. initialize the trading, a) create a trade account, b) initialize the trading dataframe
        trade_account = FutureTradingAccount(self.init_capital)
        df_bt_result = self.init_trading(df_testing_signal, trade_account)

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
            initial_margin_per_contract = row['close']* trade_account.contract_multiplier * trade_account.margin_rate
            if trade_account.bal_avialable > initial_margin_per_contract:   # check if sufficient cash to open a position
                if is_signal_buy and trade_account.position_size >= 0:
                    t_size     = 1
                    t_price    = row['close']
                    commission = trade_account.open_position(t_size, t_price)
                    df_bt_result.loc[index, 'action']       = 'open'
                    df_bt_result.loc[index, 'logic']        = 'signal buy'
                    df_bt_result.loc[index, 't_size']       = t_size
                    df_bt_result.loc[index, 't_price']      = t_price
                    df_bt_result.loc[index, 'commission']   = commission
                    df_bt_result.loc[index, 'pnl_action']   = -commission
                    trade_account.stop_level                = t_price - para_comb['stop_loss']
                    is_mtm                                  = True
                elif is_signal_sell and trade_account.position_size <= 0:
                    t_size     = -1
                    t_price    = row['close']
                    commission = trade_account.open_position(t_size, t_price)
                    df_bt_result.loc[index, 'action']       = 'open'
                    df_bt_result.loc[index, 'logic']        = 'signal sell'
                    df_bt_result.loc[index, 't_size']       = t_size
                    df_bt_result.loc[index, 't_price']      = t_price
                    df_bt_result.loc[index, 'commission']   = commission
                    df_bt_result.loc[index, 'pnl_action']   = -commission
                    trade_account.stop_level                = t_price + para_comb['stop_loss']
                    df_bt_result.loc[index, 'stop_level']   = trade_account.stop_level
                    is_mtm                                  = True


            # b) determine if it is time to close position
            ''' Strategy:
            1. when the position loss reach the stop loss, close the position
            2. when the margin call, close the position -> this is handled in the mark_to_market function
            '''
            if not is_mtm:
                is_stop_loss      = (trade_account.position_size > 0 and row['close'] < trade_account.stop_level) or (trade_account.position_size < 0 and row['close'] > trade_account.stop_level)
                is_close_position =  is_stop_loss or (trade_account.position_size != 0 and pd.Timestamp(index).time() == pd.Timestamp('10:15').time())
                if is_close_position:
                    t_size     = copy.deepcopy(-trade_account.position_size)
                    t_price    = row['close']
                    commission, pnl_realized = trade_account.close_position(t_size, t_price)
                    df_bt_result.loc[index, 'action']       = 'close'
                    df_bt_result.loc[index, 'logic']        = 'enhanced stop' if is_stop_loss else 'max_hold'
                    df_bt_result.loc[index, 't_size']       = t_size
                    df_bt_result.loc[index, 't_price']      = t_price 
                    df_bt_result.loc[index, 'commission']   = commission
                    df_bt_result.loc[index, 'pnl_action']   = pnl_realized
                    is_mtm                                  = True

            # c) mark profile value to market
            if not is_mtm:
                mtm_res = trade_account.mark_to_market(row['close'])
                if mtm_res['action'] == 'force close':
                    df_bt_result.loc[index, 'action']       = mtm_res['action']
                    df_bt_result.loc[index, 'logic']        = mtm_res['logic']
                    df_bt_result.loc[index, 't_size']       = mtm_res['t_size']
                    df_bt_result.loc[index, 't_price']      = mtm_res['t_price']
                    df_bt_result.loc[index, 'commission']   = mtm_res['commission']
                    df_bt_result.loc[index, 'pnl_action']   = mtm_res['pnl_realized']
                if trade_account.position_size > 0 and row['close'] > trade_account.stop_level + para_comb['stop_loss'] + para_comb['ladder']:
                    trade_account.stop_level += para_comb['stop_loss']
                elif trade_account.position_size < 0 and row['close'] < trade_account.stop_level - para_comb['stop_loss'] - para_comb['ladder']:
                    trade_account.stop_level -= para_comb['stop_loss']
                    df_bt_result.loc[index, 'stop_level']   = trade_account.stop_level
            
            # d) record account status in df_bt_result
            df_bt_result.loc[index,'pos_size']       = trade_account.position_size
            df_bt_result.loc[index,'pos_price']      = trade_account.position_price
            df_bt_result.loc[index,'pnl_unrealized'] = trade_account.pnl_unrealized
            df_bt_result.loc[index,'nav']            = trade_account.bal_equity
            df_bt_result.loc[index,'bal_cash']       = trade_account.bal_cash
            df_bt_result.loc[index,'bal_avialable']  = trade_account.bal_avialable
            df_bt_result.loc[index,'margin_initial'] = trade_account.margin_initial
            df_bt_result.loc[index,'cap_usage']      = f'{trade_account.cap_usage:.4f}'
        df_bt_result.reset_index(inplace=True)
        return df_bt_result



if __name__ == "__main__":
    engine = MPLing(
        initial_capital     = 1_000_000,
        underlying  = "HK.HSImain",
        start_date  = "2020-01-01",
        end_date    = "2024-01-31",
        bar_size    = KLType.K_5M,
        para_dict   = {
            'stop_loss'     : [50],
            'ladder'        : [10, 20, 30, 40],
        },
        folder_path         = "strategy/data/mp_ling",
        is_rerun_backtest   = True,
        is_update_data      = False,
        summary_mode        = False,
        multi_process_mode  = True,
    )
    if engine.is_update_data: engine.generate_mp()
    bt_result_list = engine.run()
    PlotApp(bt_result_list).plot()

