from typing import List
import pandas as pd
import dash
from dash import Dash, html, dcc, Output, Input, State, dash_table
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from termcolor import cprint


class PlotApp:
    style_root_div = {
        'backgroundColor': '#D0B8A8', 
        'width': '100%', 
        'height': '100vh',
        'display': 'flex', 
        'flexDirection': 'column', 
    }

    style_header = {
        'backgroundColor': '#D0B8A8', 
        'width': '100%',
        'maxHeight': '10vh',
        'display': 'flex', 
        'justifyContent': 'center', 
        'fontSize': 30, 
        'fontfamily': 'Arial', 
        'margin': 'auto', 
    }

    style_body = {
        'backgroundColor': '#DFD3C3', 
        'height': '90%',
        'minHeight': '800px',
        'width': '100%', 
        'display': 'flex', 
        'justifyContent': 'space-around'
    }

    style_body_sub_div = {
        'backgroundColor': '#F8EDE3', 
        'display': 'inline-block', 
        'minHeight': '780px',
        'height': '95%',
        'borderRadius': '10px',
        'marginTop': '10px',
        'marginLeft': 'auto', 
        'marginRight': 'auto',
    }

    style_element = {
        'borderRadius': '5px',
        # 'padding': '10px',
        'width': '95%',
        'marginTop': '10px',
        'marginLeft': 'auto', 
        'marginRight': 'auto',
    }

    def __init__(self, df_bt_result_list: List[pd.DataFrame]) -> None:
        self.df_bt_result_list = df_bt_result_list
        df_performance_columns = ['ref_tag'] + list(df_bt_result_list[0].attrs['performance_report'].keys())
        df_performance = pd.DataFrame(columns=df_performance_columns)
        df_para_columns = ['ref_tag'] + (list(df_bt_result_list[0].attrs['para_comb'].keys()))
        df_para = pd.DataFrame(columns=df_para_columns)
        df_bt_result_dict = {}

        for df in df_bt_result_list:
            df_performance.loc[df.attrs['ref_tag']] = [
                df.attrs['ref_tag'],
                df.attrs['performance_report']['number_of_trades'],
                df.attrs['performance_report']['win_rate'],
                df.attrs['performance_report']['total_cost'],
                df.attrs['performance_report']['pnl_trading'],
                df.attrs['performance_report']['roi_trading'],
                df.attrs['performance_report']['mdd_pct_trading'],
                df.attrs['performance_report']['mdd_dollar_trading'],
                df.attrs['performance_report']['pnl_bah'],
                df.attrs['performance_report']['roi_bah'],
                df.attrs['performance_report']['mdd_pct_bah'],
                df.attrs['performance_report']['mdd_dollar_bah'],
                df.attrs['performance_report']['sharpe_ratio'],
            ]

            df_para.loc[df.attrs['ref_tag']] = [df.attrs['ref_tag']] + list(df.attrs['para_comb'].values())
            df_bt_result_dict[df.attrs['ref_tag']] = df

        # dash data formater
        money       = dash_table.FormatTemplate.money(2)
        percentage  = dash_table.FormatTemplate.percentage(2)
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.title  = f'{df_bt_result_list[0].attrs["strategy"]} - Backtest Results'
        self.app.layout = html.Div(
                style    = self.style_root_div,
                children = [
                    html.Div( id    = "header", style = self.style_header,
                        className   = 'row', 
                        children    = 'Backtest Results'
                    ),
                    html.Div( id    = "body", style = self.style_body,
                        className   = 'row', 
                        children    = [
                            dcc.Store(id='current_ref', data=''),
                            html.Div( id  = "plot-area", style = {**self.style_body_sub_div, 'width': '60%'},
                                children    = [
                                    dbc.Tabs( id='graph_tab',
                                        children=[
                                            dbc.Tab(label='Strategy Detail',tab_id='strategy_detail'),
                                            dbc.Tab(label='All Curves', tab_id='all_curves'),
                                        ],
                                        active_tab='strategy_detail',
                                    ),
                                    dcc.Graph(id='graph_curve_detail', figure={}, style=self.style_element),
                                    dcc.Graph(id='grapph_all_curves', figure=self.plot_summary(), style=self.style_element),
                                ]
                            ),
                            html.Div( id  = "data-table", style = {**self.style_body_sub_div, 'width': '35%'},
                                children=[
                                    dash_table.DataTable(id='bt_result_table', style_table = self.style_element,
                                        data=df_performance[['ref_tag', 'pnl_trading', 'roi_trading', 'mdd_pct_trading', 'sharpe_ratio']].to_dict('records'),
                                        columns=[
                                            {'name': 'Backtest Reference', 'id': 'ref_tag'},
                                            {'name': 'Profit/Loss', 'id': 'pnl_trading', 'type': 'numeric', 'format': money},
                                            {'name': 'ROI', 'id': 'roi_trading', 'type': 'numeric', 'format': percentage},
                                            {'name': 'MDD', 'id': 'mdd_pct_trading', 'type': 'numeric', 'format': percentage},
                                            {'name': 'Sharpe', 'id': 'sharpe_ratio', 'type': 'numeric'},
                                        ],
                                        sort_by=[{'column_id': 'sharpe_ratio', 'direction': 'desc'}],
                                        sort_action='native',
                                        active_cell={'row': 0, 'column': 0, 'column_id': 'ref_tag'},
                                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                        style_cell={'textAlign': 'left'},
                                        style_cell_conditional=[
                                            {'if': {'column_id': 'pnl_trading'}, 'textAlign': 'right'},
                                            {'if': {'column_id': 'roi_trading'}, 'textAlign': 'right'},
                                            {'if': {'column_id': 'mdd_pct_trading'}, 'textAlign': 'right'},
                                            {'if': {'column_id': 'sharpe_ratio'}, 'textAlign': 'right'},

                                        ],
                                        page_size=8,
                                    ),
                                    dash_table.DataTable(id='performance_table_1', style_table = self.style_element,
                                        columns=[
                                            {'name': 'Performance', 'id': 'bt_performance'},
                                            {'name': 'Value', 'id': 'performance_value'},
                                        ],
                                        style_cell={'textAlign': 'left'},
                                        style_cell_conditional=[
                                            {'if': {'column_id': 'performance_value'}, 'textAlign': 'right'},
                                        ],
                                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                    ),
                                    dash_table.DataTable(id='performance_table_2', style_table = self.style_element,
                                        style_cell={'textAlign': 'left'},
                                        style_cell_conditional=[
                                            {'if': {'column_id': 'Metrics'}, 'fontWeight': 'bold', 'textAlign': 'left'},
                                            {'if': {'column_id': 'Trading'}, 'backgroundColor': 'lightblue', 'textAlign': 'right'},
                                            {'if': {'column_id': 'Buy & Hold'}, 'backgroundColor': 'lightgreen', 'textAlign': 'right'}
                                        ],
                                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                    ),
                                    dash_table.DataTable(id='para_table', style_table = self.style_element,
                                        style_cell={'textAlign': 'left'},
                                        style_cell_conditional=[
                                            {'if': {'column_id': 'para_name'}, 'textAlign': 'left', 'fontWeight': 'bold'},
                                            {'if': {'column_id': 'para_value'}, 'textAlign': 'right'},
                                        ],
                                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                    )
                                ]
                            ),
                        ]
                    ),
                ]
            )
        
        @self.app.callback(
            [Output('bt_result_table', 'data'),Output('bt_result_table', 'active_cell')],
            Input('bt_result_table', 'sort_by'),
            State('bt_result_table', 'data')
        )
        def sort_table_data(sort_by, tableData):
            if not sort_by:
                raise PreventUpdate

            df = pd.DataFrame(tableData)
            for sort in sort_by:
                df = df.sort_values(by=sort['column_id'], ascending=(sort['direction'] == 'asc'))

            return df.to_dict('records'), {'row': 0, 'column': 0, 'column_id': 'ref_tag'} 

        @self.app.callback(
            [
                Output('graph_curve_detail', 'style'), 
                Output('grapph_all_curves', 'style')
            ],
            Input('graph_tab', 'active_tab')
        )
        def switch_graph_tab(active_tab):
            match active_tab:
                case 'strategy_detail':
                    return self.style_element, {**self.style_element, 'display': 'none'}
                case 'all_curves':
                    return {**self.style_element, 'display': 'none'}, self.style_element

        # update state of current reference
        @self.app.callback(
            Output('current_ref', 'data'),
            [Input('grapph_all_curves', 'clickData'), Input('bt_result_table', 'active_cell'),],
            State('bt_result_table', 'data')
        )
        def update_current_ref(clickData, active_cell, tableData):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise PreventUpdate
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            cprint(f'trigger_id: {trigger_id} -> upadte current ref', 'yellow')

            if trigger_id == 'grapph_all_curves' and clickData:
                ref_tag = clickData['points'][0]['customdata']
                cprint(f'current_ref: {ref_tag}', 'yellow')
                return ref_tag
            
            if trigger_id == 'bt_result_table' and active_cell:
                ref_tag = tableData[active_cell['row']]['ref_tag']
                cprint(f'current_ref: {ref_tag}', 'yellow')
                return ref_tag

        # consquence of updating the current reference state
        @self.app.callback(
            [
                Output('graph_curve_detail', 'figure'),
                Output('grapph_all_curves', 'figure'),
                Output('bt_result_table', 'style_data_conditional'),
                Output('performance_table_1', 'data'),
                Output('performance_table_2', 'data'),
                Output('para_table', 'data'),
            ],
            [
                Input('current_ref', 'data'),
            ],
            [
                State('grapph_all_curves', 'figure'),
            ],
            allow_duplicate=True
        )
        def update_for_ceuurent_ref(current_ref, figure_all_curves):
            ctx = dash.callback_context
            if (not ctx.triggered) or (not current_ref): raise PreventUpdate
            
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            cprint(f'trigger_id: {trigger_id}', 'green')

            if current_ref:
                for trace in figure_all_curves['data']:
                    if trace['customdata'][0] == current_ref:
                        trace['line']['width'] = 3
                        trace['opacity'] = 1
                    else:
                        trace['line']['width'] = 1
                        trace['opacity'] = 0.5
                figure_curve_detail = self.plot_detail(df_bt_result_dict[current_ref])

            style_data_conditional = [{
                'if': {'filter_query': f'{{ref_tag}} eq "{current_ref}"'},
                'backgroundColor': 'lightblue'
            }]

            current_ref_performance = df_performance.loc[current_ref]
            df_table_1 = pd.DataFrame(
                {
                    'bt_performance':['Number of Trades', 'Win Rate', 'Total Cost', 'PnL Trading', 'ROI Trading'],
                    'performance_value':[
                        f'{current_ref_performance["number_of_trades"]:,}',
                        f'{current_ref_performance["win_rate"]:.2%}',
                        f'{current_ref_performance["total_cost"]:,.2f}',
                        f'{current_ref_performance["pnl_trading"]:,.2f}',
                        f'{current_ref_performance["roi_trading"]:.2%}',
                    ]
                },
            )
            df_table_2 = pd.DataFrame(
                {
                    'Metrics':['Profit/Loss', 'Return on Investment', 'MDD Dollar', 'MDD Percentage' ],
                    'Trading':[
                        f'{current_ref_performance["pnl_trading"]:,.2f}',
                        f'{current_ref_performance["roi_trading"]:.2%}',
                        f'{current_ref_performance["mdd_dollar_trading"]:,.2f}',
                        f'{current_ref_performance["mdd_pct_trading"]:.2%}',
                    ],
                    'Buy & Hold':[
                        f'{current_ref_performance["pnl_bah"]:,.2f}',
                        f'{current_ref_performance["roi_bah"]:.2%}',
                        f'{current_ref_performance["mdd_dollar_bah"]:,.2f}',
                        f'{current_ref_performance["mdd_pct_bah"]:.2%}',
                    ]
                },
            )

            df_table_para = pd.DataFrame(
                {
                    'para_name': df_para.columns[1:],
                    'para_value': df_para.loc[current_ref][1:]
                }
            )
            return figure_curve_detail, figure_all_curves, style_data_conditional, df_table_1.to_dict('records'), df_table_2.to_dict('records'), df_table_para.to_dict('records')



    def plot_detail(self, df_bt_result: pd.DataFrame) -> go.Figure:
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        # price movement of the underlying
        fig.add_trace(
            go.Candlestick(
                x       = df_bt_result['datetime'],
                open    = df_bt_result['open'],
                high    = df_bt_result['high'],
                low     = df_bt_result['low'],
                close   = df_bt_result['close'],
                name    = 'Price',
            ),
            secondary_y=False
            )
        # equity curve
        fig.add_trace(
            go.Scatter(
                x       =df_bt_result['datetime'],
                y       =df_bt_result['nav'],
                mode    ='lines',
                name    ='Equity',
                line    =dict(color='green', width=2),
                customdata = [df_bt_result.attrs['ref_tag']] * len(df_bt_result),
            ),
            secondary_y=True,
        )
        # actions - buy
        action_buy_df = df_bt_result[df_bt_result['action'] == 'buy']
        fig.add_trace(
            go.Scatter(
                x       =action_buy_df['datetime'],
                y       =action_buy_df['t_price'],
                mode    ='markers',
                marker  =dict(
                    symbol  ='triangle-up-open',
                    size    =10,
                    color   ='brown',
                ),
                name='Buy',
                text='Open: ' 
                    + action_buy_df['t_size'].astype(str) + '@' + action_buy_df['t_price'].astype(str) 
                    + ' (signal: ' + action_buy_df['signal'] + ')',
                hoverinfo='text',
                customdata = [df_bt_result.attrs['ref_tag']] * len(df_bt_result),
            ),
            secondary_y=False,
        )
        # actions - sell
        action_sell_df = df_bt_result[df_bt_result['action'] == 'sell']
        fig.add_trace(
            go.Scatter(
                x       =action_sell_df['datetime'],
                y       =action_sell_df['t_price'],
                mode    ='markers',
                marker  =dict(
                    symbol='triangle-down-open',
                    size=10,
                    color='brown',
                ),
                name='Sell',
                text='Open: ' 
                    + action_sell_df['t_size'].astype(str) + '@' + action_sell_df['t_price'].astype(str) 
                    + ' (signal: ' + action_sell_df['signal'] + ')',
                hoverinfo='text',
                customdata = [df_bt_result.attrs['ref_tag']] * len(df_bt_result),
            ),
            secondary_y=False,
        )
        #actions - close
        action_close_df = df_bt_result[df_bt_result['action'] == 'close']
        fig.add_trace(
            go.Scatter(
                x=action_close_df['datetime'],
                y=action_close_df['t_price'],
                mode='markers',
                marker=dict(
                    symbol='circle-open',
                    size=10,
                    color='blue',
                ),
                name='Close',
                text='Close: ' 
                    + action_close_df['t_size'].astype(str) + '@' + action_close_df['t_price'].astype(str) 
                    + ' (' + action_close_df['logic'] + ', P/L: ' + action_close_df['pnl_action'].astype(str) + ')',
                hoverinfo='text',
                customdata = [df_bt_result.attrs['ref_tag']] * len(df_bt_result),
            ),
            secondary_y=False,
        )

        fig.update_layout(
            height=800,
            yaxis=dict(tickformat=',', autorange=True),
            yaxis2=dict(tickformat=',', autorange=True),
            xaxis=dict(showticklabels=True, autorange=True, type='date'),
            autosize=True,
            hovermode='closest',
            hoverlabel=dict(bgcolor='#af9b46', font_size=16, font_family='Rockwell',),
            paper_bgcolor='#F8EDE3',
            xaxis_rangeslider_visible=False,
        )

        return fig



    def plot_summary(self) -> go.Figure:
        fig = go.Figure()
        for df in self.df_bt_result_list:
            fig.add_trace(go.Scatter(
                x       = df['datetime'], 
                y       = df['nav'], 
                mode    = 'lines', 
                name    = 'nav',
                line    = {'width': 1},
                opacity = 0.5,
                customdata = [df.attrs['ref_tag']] * len(df),
                text    =   f'Ref: {df.attrs["ref_tag"]} <br>' +
                            f'total_trades: {df.attrs["performance_report"]["number_of_trades"]} <br>' +
                            f'win_rate: {df.attrs["performance_report"]["win_rate"]:.2f} <br>' +
                            f'total_cost: {df.attrs["performance_report"]["total_cost"]:,.2f} <br>' +
                            f'pnl $: {df.attrs["performance_report"]["pnl_trading"]:,.2f} <br>' +
                            f'roi %: {df.attrs["performance_report"]["roi_trading"]:.2%} <br>' +
                            f'mdd $: {df.attrs["performance_report"]["mdd_dollar_trading"]:,.2f} <br>' +
                            f'mdd %: {df.attrs["performance_report"]["mdd_pct_trading"]:.2%} <br>' +
                            f'roi(trading-B&H) %: {(df.attrs["performance_report"]["roi_trading"]-df.attrs["performance_report"]["roi_bah"]):.2%} <br>' +
                            f'mdd(trading-B&H) %: {(df.attrs["performance_report"]["mdd_pct_trading"]-df.attrs["performance_report"]["mdd_pct_bah"]):.2%} <br>',
                hoverinfo='text',
            ))

        fig.update_layout(
            height=800,
            showlegend=False,
            hovermode='closest',
            paper_bgcolor='#F8EDE3',
            xaxis=dict(
                showspikes=True, spikemode='across', spikesnap='cursor', showline=True
            ),
            yaxis=dict(
                side='right',
                tickformat=',',
                showgrid=True,
                autorange=True,
                showspikes=True, spikemode='across', spikesnap='cursor', showline=True
            ),
        ) 

        return fig

    def plot(self):
        self.app.run_server(debug=True)
