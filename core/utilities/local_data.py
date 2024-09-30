import json
import os
import pandas as pd
from termcolor import cprint

##### ***** Save & Read local Data ***** #####
def to_csv_with_metadata(df:pd.DataFrame, file_name:str, folder:str = 'data/backtest'):
    if not os.path.exists(folder): os.makedirs(folder)
    path_name = f'{folder}/{file_name}'
    df.to_csv(f'{path_name}.csv', index=True)
    with open(f'{path_name}.json', 'w') as f:
        json.dump(df.attrs, f)
        f.close()
    cprint(f'DataFrame saved to {path_name}.csv', 'yellow')
    cprint(f'Metadata saved to {path_name}.json', 'green')

def read_csv_with_metadata(file_name:str, folder:str = 'data/backtest') -> pd.DataFrame:
    path_name = f'{folder}/{file_name}'
    df = pd.read_csv(f'{path_name}.csv', index_col=0)
    with open(f'{path_name}.json', 'r') as f:
        df.attrs = json.load(f)
        f.close()
    return df