from datetime import datetime
import itertools
import multiprocessing
import os
import sys
from typing import List
import numpy as np
import pandas as pd
import requests
from termcolor import cprint

# local import
from utilities.local_data import to_csv_with_metadata, read_csv_with_metadata

class BacktestEngine:
    risk_free_rate_url = 'https://app-kw6fovhcnq-uc.a.run.app/getYieldCurveTB52'
    local_rf_folder_path = 'data/risk_free_rate'
    local_rf_daily_path = 'data/risk_free_rate/tb_52w_rate_daily.csv'
    local_rf_monthly_path = 'data/risk_free_rate/tb_52w_rate_monthly.csv'
    
    def __init__(
            self, 
            initial_capital:float,
            underlying:str, start_date:str, end_date:str, bar_size:str, para_dict:dict,
            folder_path:str, 
            is_rerun_backtest:bool=True, is_update_data:bool=True, summary_mode:bool=False, multi_process_mode:bool=False,
        ) -> None:
        self.folder_path        = folder_path
        self.is_rerun_backtest  = is_rerun_backtest
        self.summary_mode       = summary_mode
        self.multi_process_mode = multi_process_mode
        self.is_update_data     = is_update_data
        self.init_capital       = initial_capital
        self.underlying         = underlying
        self.start_date         = start_date
        self.end_date           = end_date
        self.bar_size           = bar_size
        self.para_dict          = para_dict
        self.file_name          = underlying.replace('-', '').replace('.', '').replace(' ', '')



    # preparing for the backtest
    def get_all_para_combinations(self) -> None:
        '''
        This is a function to generate all possible combinations of the parameters for the strategy.
        Return a dictionary with reference tags as keys for each possible combination of the parameters.
        eg:
        arg = {'stop_loss': [10, 20, 30], 'target_profit': [10, 20, 30]}
        return {
                'ref_001': {'stop_loss': 10, 'target_profit': 10},
                'ref_002': {'stop_loss': 10, 'target_profit': 20},
                'ref_003': {'stop_loss': 10, 'target_profit': 30},
                ...
                }
        '''
        para_values = list(self.para_dict.values())
        para_keys = list(self.para_dict.keys())
        para_list = list(itertools.product(*para_values))

        df = pd.DataFrame(para_list, columns=para_keys)
        ref_tag = [f'{self.file_name}_bt_{i+1:03d}' for i in df.index]
        df['ref_tag'] = ref_tag
        df.set_index('ref_tag', inplace=True)
        self.para_comb_dict = df.to_dict(orient='index')
        


    # Strategic specific functions
    def get_hist_data(self) -> pd.DataFrame:
        '''
        This function will return the historical data for the underlying.
        Minimum columns required: ['timestamp', 'datetim', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
        '''
        pass

   
    def generate_signal(self, df_hist_data:pd.DataFrame, para_comb:dict) -> pd.DataFrame:
        '''
        This is a function to generate the signals based on the historical data and the parameters.
        '''
        pass 


    def init_trading(self, df_testing:pd.DataFrame, trade_account) -> pd.DataFrame:
        '''
        This is a function to initialize the trading simulation process.
        '''
        df_testing['action']     = ''   # action: buy, sell, close
        df_testing['logic']      = ''   # logic: open, reach profit target, reach stop loss, stop loss, force close
        df_testing['t_size']     = 0    # size in the transaction
        df_testing['t_price']    = 0    # price in the transaction
        df_testing['commission'] = 0    # commission in the transaction

        df_testing['pnl_action'] = 0.0  # realised P/L from the action, including commission
        df_testing['pos_size']   = 0    # position size
        df_testing['pos_price']  = 0.0  # position average price

        df_testing['pnl_unrealized'] = float(trade_account.pnl_unrealized)        # unrealized profit and loss
        df_testing['nav']            = float(trade_account.bal_equity)            # net asset value = cash balance + unrealized profit and loss
        df_testing['bal_cash']       = float(trade_account.bal_cash)              # cash balance: booked equity
        df_testing['bal_avialable']  = float(trade_account.bal_avialable)         # cash available for trading = cash balance - initial margin + unrealized profit and loss
        df_testing['margin_initial'] = float(trade_account.margin_initial)        # initial margin in $ term
        df_testing['cap_usage']      = f'{trade_account.cap_usage:.2f}%'          # usage of the capital = initial margin / cash balance
        
        return df_testing


    def action_on_signal(self, df_testing_signal:pd.DataFrame, para_comb:dict) -> pd.DataFrame:
        '''
        This function responsible for generating trading signals based on the raw data and para combination.
        '''
        # 1. initialize the trading, a) create a trade account, b) initialize the trading dataframe
        df_bt_result = self.init_trading(df_testing_signal)

        # 2. loop through the signals
        return df_bt_result

    # summary functions
    def update_risk_free_rate(self) -> pd.DataFrame:
        try:
            rate_data = requests.get(self.risk_free_rate_url).json()
            df = pd.DataFrame(data=rate_data.values(), index=rate_data.keys())
            df.index.name = 'date'
            df.to_csv(self.local_rf_daily_path, index=True)

            grouped = df.groupby('month')
            df_tb_52w = grouped['bank_discount'].mean().reset_index()
            df_tb_52w.columns = ['month', 'tb_52w_rate']
            df_tb_52w.to_csv(self.local_rf_monthly_path, index=False)
            return df_tb_52w
        except Exception as e:
            cprint("Error: failed to get the risk free rate!", 'red')
            print(e)
            sys.exit()


    def get_risk_free_rate(self) -> float:
        '''
        This function will return a dictionary of the risk free rates for the period of the backtest period.
        '''
        cprint('Getting the risk free rate dictionary......', 'yellow')
        start_month = int(self.start_date[:7].replace('-', ''))
        end_month = int(self.end_date[:7].replace('-', ''))

        if (not os.path.exists(self.local_rf_folder_path)): os.makedirs(self.local_rf_folder_path)
        # check if the monthly_mean_rate is already saved
        if (os.path.exists(self.local_rf_monthly_path)):
            df_tb_52w = pd.read_csv(self.local_rf_monthly_path)
            # check if the local data is up to date, otherwise update the data
            if (df_tb_52w['month'].max() < end_month):
                df_tb_52w = self.update_risk_free_rate()
        else:
            df_tb_52w = self.update_risk_free_rate()

        df_tb_52w_range = df_tb_52w[(df_tb_52w['month'] >= start_month) & (df_tb_52w['month'] <= end_month)]
        df_tb_52w_range.set_index('month', inplace=True)
        rf_rate_dict = df_tb_52w_range.to_dict(orient='dict')['tb_52w_rate']
        # calculate the geometric mean of the risk free rate
        risk_free_rate = 1
        for rate in rf_rate_dict.values():
            risk_free_rate *= (1+rate/12/100)
        risk_free_rate = risk_free_rate**(12/(len(rf_rate_dict))) - 1
        cprint(f"Geometric mean of risk free rates in the period: {start_month} to {end_month} is: \n{risk_free_rate:.2%}", 'green')
        return risk_free_rate


    def generate_bt_report(self, df_bt_result:pd.DataFrame, risk_free_rate:float=0.02) -> dict:
        # performance metrics
        number_of_trades = df_bt_result[df_bt_result['action']=='close'].shape[0]
        if number_of_trades == 0:
            win_rate            = 0.0
            total_cost          = 0.0
            pnl_trading         = 0.0
            roi_trading         = 0.0
            mdd_pct_trading     = 0.0
            mdd_dollar_trading  = 0.0
            sharpe_ratio        = 0.0
            pnl_bah             = 0.0
            roi_bah             = 0.0
            mdd_pct_bah         = 0.0
            mdd_dollar_bah      = 0.0
        else:
            win_rate = df_bt_result[df_bt_result['pnl_action'] > 0].shape[0] / number_of_trades
            total_cost = df_bt_result['commission'].sum()
            # MDD
            df_bt_result['cum_max_nav']     = df_bt_result['nav'].cummax()
            df_bt_result['dd_pct_nav']      = df_bt_result['nav'] / df_bt_result['cum_max_nav'] -1
            df_bt_result['dd_dollar_nav']   = df_bt_result['nav']- df_bt_result['cum_max_nav']
            mdd_pct_trading                 = df_bt_result['dd_pct_nav'].min()
            mdd_dollar_trading              = df_bt_result['dd_dollar_nav'].min()

            df_bt_result['cum_max_bah']     = df_bt_result['close'].cummax()
            df_bt_result['dd_pct_bah']      = df_bt_result['close'] / df_bt_result['cum_max_bah'] -1
            df_bt_result['dd_dollar_bah']   = df_bt_result['close']- df_bt_result['cum_max_bah']
            mdd_pct_bah                     = df_bt_result['dd_pct_bah'].min()
            mdd_dollar_bah                  = df_bt_result['dd_dollar_bah'].min()


            # net profit
            pnl_trading = df_bt_result['nav'].iloc[-1] - df_bt_result['nav'].iloc[0]
            roi_trading = pnl_trading / df_bt_result['nav'].iloc[0]

            pnl_bah     = df_bt_result['close'].iloc[-1] - df_bt_result['close'].iloc[0]
            roi_bah     = pnl_bah / df_bt_result['close'].iloc[0]

            # sharpe ratio
            df_bt_result['trade_month'] = df_bt_result['trade_date'].apply(lambda x: int(x[:7].replace('-', '')))
            grouped = df_bt_result.groupby('trade_month')
            df_monthly_pnl = grouped['nav'].agg(['first', 'last']).reset_index()
            df_monthly_pnl.columns = ['trade_month', 'nav_first', 'nav_last']

            rf = (1 + risk_free_rate)**(1/12) - 1
            df_monthly_pnl['roi'] = df_monthly_pnl['nav_last'].pct_change()
            df_monthly_pnl.loc[0, 'roi'] = df_monthly_pnl['nav_last'].iloc[0] / df_monthly_pnl['nav_first'].iloc[0] - 1
            df_monthly_pnl['excess_return'] = df_monthly_pnl['roi'] - rf
            sharpe_ratio = df_monthly_pnl['excess_return'].mean() / df_monthly_pnl['excess_return'].std()

        performance_report = {
            'number_of_trades'      : number_of_trades,
            'win_rate'              : win_rate,
            'total_cost'            : total_cost,
            'pnl_trading'           : pnl_trading,
            'roi_trading'           : roi_trading,
            'mdd_pct_trading'       : mdd_pct_trading,
            'mdd_dollar_trading'    : mdd_dollar_trading,
            'pnl_bah'               : pnl_bah,
            'roi_bah'               : roi_bah,
            'mdd_pct_bah'           : mdd_pct_bah,
            'mdd_dollar_bah'        : mdd_dollar_bah,
            'sharpe_ratio'          : sharpe_ratio
        }
        performance_report = {k: round(float(v), 4) if isinstance(v, (np.int64, np.float64, float)) else v for k, v in performance_report.items()}
        return performance_report


    def simulate_trade(self, ref_tag:str,) -> pd.DataFrame:
        '''
        This is a function to run a single backtest on one of the parameter combination.
        Return a pandas dataframe with 
            index: timestamp
            columns: ['timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'rolling_gain', 'calculation_col_1', 'calculation_col_2', 'signal', 'action', 'logic', 't_price', 't_size', 'commission', 'pnl_action', 'acc_columns'].
            metadata: {
                'ref_tag':           ref_tag,
                'para_comb':         para_comb,
                'performace_report': { },
                benchmark:{
                    'roi_sp500': 0,
                    'tbill_52w': 0,
                }
            }
        '''
        cprint(f"Running backtest for {ref_tag}", 'green')
        # 1. run backtest for the specific parameter combination
        para_comb           = self.para_comb_dict[ref_tag]
        df_testing_signal   = self.generate_signal(self.df_hist_data.copy(), para_comb)
        # print(df_testing_signal)
        # sys.exit()
        df_bt_result        = self.action_on_signal(df_testing_signal, para_comb)
        cprint(f"Signal generated for {ref_tag}", 'green')

        df_bt_result.attrs  = {
            'ref_tag'           : ref_tag,
            'strategy'          : self.__class__.__name__,
            'init_capital'      : self.init_capital,
            'para_comb'         : para_comb,
            'performance_report': self.generate_bt_report(df_bt_result),
            'benchmark': {
                'roi_sp500': 0,
                'tbill_52w': round(self.risk_free_rate, 4),
            }
        }

        # 2. save the backtest result to local file
        bt_result_folder = os.path.join(self.folder_path, 'bt_results')
        if self.summary_mode:
            df_bt_result = df_bt_result[df_bt_result['t_size'] != 0]
        to_csv_with_metadata(df_bt_result, ref_tag, bt_result_folder)
        
        # 3. return the backtest result
        return df_bt_result


    def read_backtest_result(self) -> List[pd.DataFrame]:
        '''Read the backtest results from the the designated folder.'''
        backtest_results = []
        try:
            bt_result_path = os.path.join(self.folder_path, 'bt_results')
            file_list = list(set(file_n.split('.')[0] for file_n in os.listdir(bt_result_path)))
            for file in file_list:
                if self.file_name in file:
                    cprint(f'Reading backtest result from: {file} ......', 'yellow')
                    backtest_results.append(read_csv_with_metadata(file, folder=bt_result_path))
            return backtest_results
        except Exception as e:
            cprint("Error: failed to read the backtest results!", 'red')
            print(e)
            sys.exit()


    def run(self) -> List[pd.DataFrame]:
        '''
        This is the main controller to run the backtests.
        It will return a list of pandas dataframes containing the backtest results.
        '''
        if datetime.strptime(self.end_date, "%Y-%m-%d") > datetime.today():
            cprint("Error: End date is in the future!", 'red')
            sys.exit()

        # get the backtest results
        backtest_results = []
        if self.is_rerun_backtest:
            # get the historical data
            raw_data_folder = f'{self.folder_path}/raw'
            if not os.path.exists(raw_data_folder): os.makedirs(raw_data_folder)
            raw_data_file_name = f'{self.underlying}_{self.bar_size}_{self.start_date}_{self.end_date}'
            raw_data_file_name = raw_data_file_name.replace('-', '').replace('.', '').replace(' ', '') + '.csv'
            raw_data_file_path = os.path.join(raw_data_folder, raw_data_file_name)
            if self.is_update_data:
                self.df_hist_data = self.get_hist_data()
                self.df_hist_data.to_csv(raw_data_file_path, index=True)
            else:
                try:
                    self.df_hist_data = pd.read_csv(raw_data_file_path, index_col=0)
                except FileNotFoundError as e:
                    cprint(f"Error: {e}", 'red')
                    sys.exit()

            cprint('historical data is ready!', 'green')

            # generate all possible combinations of the parameters
            self.get_all_para_combinations()

            # get the risk free rate
            self.risk_free_rate = self.get_risk_free_rate()

            # run the backtest for each parameter combination
            if self.multi_process_mode:
                num_processors = multiprocessing.cpu_count()
                print(f"Running backtest with processors of: {num_processors}")
                with multiprocessing.Pool(num_processors) as pool:
                    backtest_results = pool.starmap(self.simulate_trade, [(ref_tag,) for ref_tag in self.para_comb_dict.keys()])
            else:
                for ref_tag in self.para_comb_dict.keys():
                    backtest_results.append(self.simulate_trade(ref_tag))
        else:
            backtest_results = self.read_backtest_result()

        return backtest_results



