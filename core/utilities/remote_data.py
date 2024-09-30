import sys
from termcolor import cprint
from datetime import datetime
import pandas as pd

from futu import OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK


##### ***** futu ***** #####
def get_stock_futu_api(underlying:str, start_date:str, end_date:str, ktype:KLType) -> pd.DataFrame:
    '''
    This function gets the spot contract trading data from futu-api, with (host='127.0.0.1', port=11111)
    return dataframe with columns: ["code", "name", "time_key", "open", "close", "high", "low", "pe_ratio", "turnover_rate", "volume", "turnover", "change_rate"  "last_close"  "trade_date"]
    '''
    futu_client = OpenQuoteContext(host='127.0.0.1', port=11111)
    ret, data, page_req_key = futu_client.request_history_kline(
        code            = underlying,
        start           = start_date,
        end             = end_date,
        ktype           = ktype,
        autype          = AuType.QFQ,
        fields          = [KL_FIELD.ALL],
        max_count       = 1000000,
        page_req_key    = None,
        extended_time   = True
    )
    futu_client.close()
    if ret != RET_OK:
        cprint(f"Error: {data}", "red")
        sys.exit()

    # identify the trade_date for each row
    data['dummy_col'] = data['time_key'].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    data['real_td'] = data['time_key'].apply(lambda x: x[:10])
    norm_td = []
    for index, row in data.iterrows():
        if row['dummy_col'].hour > 8:
            norm_td.append(row['real_td'])
        elif index == 0:
            norm_td.append('NA')
        else:
            norm_td.append(norm_td[-1])
    data['trade_date'] = norm_td
    data = data.drop(columns=['dummy_col', 'real_td'])
    data = data[data['trade_date'] != 'NA']
    data.rename(columns={'time_key':'datetime'}, inplace=True)

    return data





# test the functions
# if __name__ == "__main__":
#     underlying = Underlying(
#         symbol          = "HK.00388",
#         exchange        = "HKFE",
#         contract_type   = "FUT",
#         barSizeSetting  = KLType.K_5M,
#         start_date      = "2024-08-01",
#         end_date        = "2024-08-30",
#         durationStr     = "2 M",
#         rolling_days    = 4,
#         timeZone        = "Asia/Hong_Kong",
#     )

#     df_stock = get_stock_futu_api(underlying)
#     datetime = df_stock.at[0, 'time_key']
#     print(df_stock.head(20))
#     print(datetime)
#     print(type(datetime))

