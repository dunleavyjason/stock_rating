import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
from tensorflow import keras
import joblib
import datetime
import time
import plotly.graph_objects as go
import bokeh
from bokeh.plotting import figure, show
from bokeh.layouts import layout, grid, gridplot, row, column
from bokeh.models import HoverTool, ColumnDataSource, NumeralTickFormatter, Panel, Tabs
from sklearn.preprocessing import StandardScaler

st.title('Predicting Daily Stock Ratings')

with open('final_tickers.pkl', 'rb') as handle:
    final_tickers = pickle.load(handle)

ticker_company = st.selectbox(
     "Select a Ticker/Company",
     (final_tickers))

streamlit_df = pd.read_pickle("streamlit_df.pkl")

ticker = ticker_company.split(' - ')[0]

ticker_df = streamlit_df[streamlit_df.ticker == ticker]

def dtime(x):
    return np.array(x, dtype=np.datetime64)

source = ColumnDataSource(data={
    'date'      : dtime(ticker_df.date),
    'price' : ticker_df.close,
    'volume'    : ticker_df.volume,
})

p1 = figure(
    title= 'Closing Prices',
    x_axis_label='Date',
    y_axis_label='Closing Price ($)',
    x_axis_type="datetime",
    plot_width=500,
    plot_height=300)
p1.line(x='date', y='price', line_width=2, source=source)
p1.add_tools(HoverTool(tooltips=[("date", "@date{%F}"),("price", "@price"),("volume", "@volume")],
                      formatters={'@date': 'datetime'},
                      mode='vline'))


source2 = ColumnDataSource(data={
    'calendardate'      : dtime(ticker_df.calendardate),
    'revenue' : ticker_df.revenue,
    'ebitda' : ticker_df.ebitda,
    'gross_profit' : ticker_df.gp,
    'operating_income' : ticker_df.opinc,
    'net_income' : ticker_df.netinc,
    'eps' : ticker_df.eps
})
e1 = figure(
    title= 'Revenue',
    x_axis_label='Date',
    y_axis_label='Revenue ($)',
    x_axis_type="datetime")
e1.line(x='calendardate', y='revenue', line_width=2, source=source2)
e1.yaxis.formatter = NumeralTickFormatter(format='0.0a')
e1.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("revenue", "@revenue{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))
e2 = figure(
    title= 'Gross Profit',
    x_axis_label='Date',
    y_axis_label='Gross Profit ($)',
    x_axis_type="datetime")
e2.line(x='calendardate', y='gross_profit', line_width=2, source=source2)
e2.yaxis.formatter = NumeralTickFormatter(format='0.0a')
e2.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("gross_profit", "@gross_profit{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

e3 = figure(
    title= 'Operating Income',
    x_axis_label='Date',
    y_axis_label='Operating Income ($)',
    x_axis_type="datetime")
e3.line(x='calendardate', y='operating_income', line_width=2, source=source2)
e3.yaxis.formatter = NumeralTickFormatter(format='0.0a')
e3.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("operating_income", "@operating_income{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

e4 = figure(
    title= 'EBITDA',
    x_axis_label='Date',
    y_axis_label='EBITDA ($)',
    x_axis_type="datetime")
e4.line(x='calendardate', y='ebitda', line_width=2, source=source2)
e4.yaxis.formatter = NumeralTickFormatter(format='0.0a')
e4.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("ebitda", "@ebitda{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

e5 = figure(
    title= 'Net Income',
    x_axis_label='Date',
    y_axis_label='Net Income ($)',
    x_axis_type="datetime")
e5.line(x='calendardate', y='net_income', line_width=2, source=source2)
e5.yaxis.formatter = NumeralTickFormatter(format='0.0a')
e5.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("net_income", "@net_income{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

e6 = figure(
    title= 'Earnings Per Share',
    x_axis_label='Date',
    y_axis_label='EPS ($)',
    x_axis_type="datetime")
e6.line(x='calendardate', y='eps', line_width=2, source=source2)
e6.yaxis.formatter = NumeralTickFormatter(format='0.0a')
e6.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("eps", "@eps{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

ep1 = Panel(child=e1, title="Revenue")
ep2 = Panel(child=e2, title="Gross Profit")
ep3 = Panel(child=e3, title="Operating Income")
ep4 = Panel(child=e4, title="EBITDA")
ep5 = Panel(child=e5, title="Net Income")
ep6 = Panel(child=e6, title="EPS")


source3 = ColumnDataSource(data={
    'calendardate'      : dtime(ticker_df.calendardate),
    'assets' : ticker_df.assets,
    'cashneq' : ticker_df.cashneq,
    'inventory' : ticker_df.inventory,
    'receivables' : ticker_df.receivables,
    'ppe' : ticker_df.ppnenet,
    'liabilities' : ticker_df.liabilities,
    'payables' : ticker_df.payables,
    'deferredrev' : ticker_df.deferredrev,
    'debt' : ticker_df.debt,
    'equity' : ticker_df.equity
})
f1 = figure(
    title= 'Assets',
    x_axis_label='Date',
    y_axis_label='Assets ($)',
    x_axis_type="datetime")
f1.line(x='calendardate', y='assets', line_width=2, source=source3)
f1.yaxis.formatter = NumeralTickFormatter(format='0.0a')
f1.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("assets", "@assets{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))
f2 = figure(
    title= 'Cash and Equivalents',
    x_axis_label='Date',
    y_axis_label='Cash and Equivalents ($)',
    x_axis_type="datetime")
f2.line(x='calendardate', y='cashneq', line_width=2, source=source3)
f2.yaxis.formatter = NumeralTickFormatter(format='0.0a')
f2.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("cashneq", "@cashneq{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

f3 = figure(
    title= 'Inventory',
    x_axis_label='Date',
    y_axis_label='Inventory ($)',
    x_axis_type="datetime")
f3.line(x='calendardate', y='inventory', line_width=2, source=source3)
f3.yaxis.formatter = NumeralTickFormatter(format='0.0a')
f3.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("inventory", "@inventory{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

f4 = figure(
    title= 'Receivables',
    x_axis_label='Date',
    y_axis_label='Receivables ($)',
    x_axis_type="datetime")
f4.line(x='calendardate', y='receivables', line_width=2, source=source3)
f4.yaxis.formatter = NumeralTickFormatter(format='0.0a')
f4.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("receivables", "@receivables{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

f5 = figure(
    title= 'Property, Plant, and Equipment - Net',
    x_axis_label='Date',
    y_axis_label='PPE - Net ($)',
    x_axis_type="datetime")
f5.line(x='calendardate', y='ppe', line_width=2, source=source3)
f5.yaxis.formatter = NumeralTickFormatter(format='0.0a')
f5.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("ppe", "@ppe{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

f6 = figure(
    title= 'Liabilities',
    x_axis_label='Date',
    y_axis_label='Liabilities ($)',
    x_axis_type="datetime")
f6.line(x='calendardate', y='liabilities', line_width=2, source=source3)
f6.yaxis.formatter = NumeralTickFormatter(format='0.0a')
f6.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("liabilities", "@liabilites{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

f7 = figure(
    title= 'Payables',
    x_axis_label='Date',
    y_axis_label='Payables ($)',
    x_axis_type="datetime")
f7.line(x='calendardate', y='payables', line_width=2, source=source3)
f7.yaxis.formatter = NumeralTickFormatter(format='0.0a')
f7.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("payables", "@payables{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

f8 = figure(
    title= 'Deferred Revenue',
    x_axis_label='Date',
    y_axis_label='Deferred Revenue ($)',
    x_axis_type="datetime")
f8.line(x='calendardate', y='deferredrev', line_width=2, source=source3)
f8.yaxis.formatter = NumeralTickFormatter(format='0.0a')
f8.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("deferredrev", "@deferredrev{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

f9 = figure(
    title= 'Debt',
    x_axis_label='Date',
    y_axis_label='Debt ($)',
    x_axis_type="datetime")
f9.line(x='calendardate', y='debt', line_width=2, source=source3)
f9.yaxis.formatter = NumeralTickFormatter(format='0.0a')
f9.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("debt", "@debt{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

f10 = figure(
    title= 'Equity',
    x_axis_label='Date',
    y_axis_label='Equity ($)',
    x_axis_type="datetime")
f10.line(x='calendardate', y='equity', line_width=2, source=source3)
f10.yaxis.formatter = NumeralTickFormatter(format='0.0a')
f10.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("equity", "@equity{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

fp1 = Panel(child=f1, title="Assets")
fp2 = Panel(child=f2, title="Cash")
fp3 = Panel(child=f3, title="Inventory")
fp4 = Panel(child=f4, title="Receivables")
fp5 = Panel(child=f5, title="PPE")
fp6 = Panel(child=f6, title="Liabilities")
fp7 = Panel(child=f7, title="Payables")
fp8 = Panel(child=f8, title="Deferred Revenue")
fp9 = Panel(child=f9, title="Debt")
fp10 = Panel(child=f10, title="Equity")


source4 = ColumnDataSource(data={
    'calendardate'      : dtime(ticker_df.calendardate),
    'ncf' : ticker_df.ncf,
    'ncfo' : ticker_df.ncfo,
    'ncff' : ticker_df.ncff,
    'ncfi' : ticker_df.ncfi,
    'capex' : ticker_df.capex,
})
c1 = figure(
    title= 'Net Cash Flow',
    x_axis_label='Date',
    y_axis_label='NCF ($)',
    x_axis_type="datetime")
c1.line(x='calendardate', y='ncf', line_width=2, source=source4)
c1.yaxis.formatter = NumeralTickFormatter(format='0.0a')
c1.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("ncf", "@ncf{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))
c2 = figure(
    title= 'NCF - Operations',
    x_axis_label='Date',
    y_axis_label='NCF - Operations ($)',
    x_axis_type="datetime")
c2.line(x='calendardate', y='ncfo', line_width=2, source=source4)
c2.yaxis.formatter = NumeralTickFormatter(format='0.0a')
c2.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("ncfo", "@ncfo{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

c3 = figure(
    title= 'NCF - Financing',
    x_axis_label='Date',
    y_axis_label='NCF - Financing ($)',
    x_axis_type="datetime")
c3.line(x='calendardate', y='ncff', line_width=2, source=source4)
c3.yaxis.formatter = NumeralTickFormatter(format='0.0a')
c3.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("ncff", "@ncff{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

c4 = figure(
    title= 'NCF - Investing',
    x_axis_label='Date',
    y_axis_label='NCF - Investing ($)',
    x_axis_type="datetime")
c4.line(x='calendardate', y='ncfi', line_width=2, source=source4)
c4.yaxis.formatter = NumeralTickFormatter(format='0.0a')
c4.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("ncfi", "@ncfi{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

c5 = figure(
    title= 'CAPEX',
    x_axis_label='Date',
    y_axis_label='CAPEX ($)',
    x_axis_type="datetime")
c5.line(x='calendardate', y='capex', line_width=2, source=source4)
c5.yaxis.formatter = NumeralTickFormatter(format='0.0a')
c5.add_tools(HoverTool(tooltips=[("calendardate", "@calendardate{%F}"),("capex", "@capex{0.0a}")],
                      formatters={'@calendardate': 'datetime'},
                      mode='vline'))

cp1 = Panel(child=c1, title="NCF")
cp2 = Panel(child=c2, title="NCF - Operations")
cp3 = Panel(child=c3, title="NCF - Financing")
cp4 = Panel(child=c4, title="NCF - Investing")
cp5 = Panel(child=c5, title="CAPEX")


ratios_df = ticker_df[["marketcap", "ev", "pe", "pb", "ps", "assetturnover", "currentratio", "workingcapital", "de", "grossmargin", "netmargin", "payoutratio",
                     "roa", "roe", "ros"]]

ratios_df.rename(columns={"marketcap":"Market Cap", "ev":"Enterprise Value", "pe":"Price to Equity",
                        "pb":"Price to Book", "ps":"Price to Sales", "assetturnover":"Asset Turnover",
                        "currentratio":"Current Ratio", "de":"Debt to Equity", "grossmargin":"Gross Margin",
                        "netmargin":"Profit Margin", "payoutratio":"Payout Ratio", "roa":"Return on Assets",
                        "roe":"Return on Equity", "ros":"Return on Sales",
                        "workingcapital":"Working Capital"},
                        inplace=True)

day_ratios_df = pd.DataFrame(ratios_df.iloc[-1])

day_ratios_df.rename(columns={day_ratios_df.columns[0]:"TTM"}, inplace=True)

financials_df = ticker_df[["assets", "assetsc", "assetsnc", "cashneq", "cor", "consolinc", "debt", "debtc", "debtnc",
                            "deferredrev", "depamor", "deposits", "intangibles", "intexp", "invcap", "inventory",
                            "investments", "liabilities", "liabilitiesc", "liabilitiesnc", "ncf", "ncff",
                            "ncfi", "ncfo", "netinc", "opex", "opinc", "payables", "ppnenet", "receivables", "revenue",
                            "rnd", "sgna", "taxassets", "taxexp", "taxliabilities"]]

financials_df.rename(columns={"assets":"Assets", "assetsc":"Current Assets", "assetsnc":"Non-current Assets", "cashneq":"Cash and Equivalents",
                            "cor":"Cost of Revenue", "consolinc":"Consolidated Income", "debt":"Debt", "debtc":"Current Debt", "debtnc":"Non-current Debt",
                            "deferredrev":"Deferred Revenue", "depamor":"Depreciation & Amort", "deposits":"Deposits",
                            "intangibles":"Intangible Assets", "intexp":"Interest Expense", "invcap":"Invested Capital", "inventory":"Inventory",
                            "investments":"Investments", "liabilities":"Liabilities", "liabilitiesc":"Current Liabilities", "liabilitiesnc":"Non-current Liabilities",
                            "ncf":"Net Cash Flow", "ncff":"NCF - Financing", "taxliabilities":"Tax Liabilities",
                            "ncfi":"NCF - Investing", "ncfo":"NCF - Operations", "netinc":"Net Income", "opex":"Operating Expense",
                            "opinc":"Operating Income", "payables":"Payables", "ppnenet":"PPE - Net",
                            "receivables":"Receivables", "revenue":"Revenue","rnd":"R&D Expense",
                            "sgna":"SG&A Expense", "taxassets":"Tax Assets", "taxexp":"Tax Expense"},
                            inplace=True)

day_financials_df = pd.DataFrame(financials_df.iloc[-1])

day_financials_df.rename(columns={day_financials_df.columns[0]:"TTM"}, inplace=True)

predict_df_full = ticker_df.copy()
predict_df_full = pd.DataFrame(predict_df_full.iloc[-1]).T

predict_df = ticker_df.copy()

predict_df.drop([
 'ticker',
 'date',
 'calendardate',
 'name',
 'rating_cnt_strong_buys',
 'rating_cnt_mod_buys',
 'rating_cnt_holds',
 'rating_cnt_mod_sells',
 'rating_cnt_strong_sells',
 'rating_cnt_with',
 'rating_cnt_without',
 'rating_change',
 'rating_mean_recom',
 'quart',
 'year',
 'exchange',
 'sector',
 'industry'
 ], axis=1, inplace=True)

model = keras.models.load_model('/Users/dunleavyjason/Documents/Metis/stock_rating/ann_model_final')

predict_df_date = pd.DataFrame(predict_df.iloc[-1]).T

ss = joblib.load("std_scaler.joblib")
predict_df_date_scaled = ss.transform(predict_df_date)
rating_long = model.predict(predict_df_date_scaled).flat[0]
rating = round(rating_long,2)


layout = go.Layout(xaxis=dict(range=[1,5]))
fig = go.Figure(layout=layout)
fig.add_trace(go.Scatter(
    x=[rating], y=[0], mode='markers', marker_size=20, marker_symbol='triangle-down', marker_color='orange'
))
fig.update_xaxes(tick0=1, dtick = 1, nticks=5, showgrid=False, ticks='outside', ticklen=10, tickwidth=2)
fig.update_yaxes(showgrid=False,
                 zeroline=True, zerolinecolor='black', zerolinewidth=2,
                 showticklabels=False)
fig.update_layout(title={'text':'<b>Daily Stock Rating</b>', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                title_font_size=30, title_font_family='IBM Plex Sans', title_font_color="black",
                height=200, plot_bgcolor='white', xaxis= dict(tickmode = 'array',
                                                                tickvals=[1,2,3,4,5],
                                                                ticktext=['1 - Strong Buy', '2 - Buy', '3 - Hold', '4 - Sell', '5 - Strong Sell']))


#streamlit page

col1, col2 = st.beta_columns(2)
with col1:
    st.markdown('**Company:   **' + predict_df_full.name.unique()[0])
    st.markdown("**Date:   **" + str(predict_df_full.date.unique()[0])[0:10])
    st.markdown("**Closing Price:   **" + "$" + str(predict_df_full.close.unique()[0]))
    st.markdown("**Exchange:   **" + predict_df_full.exchange.unique()[0])
    st.markdown("**Sector:   **" + predict_df_full.sector.unique()[0])
    st.markdown("**Industry:   **" + predict_df_full.industry.unique()[0])
with col2:
    st.bokeh_chart(p1)

st.plotly_chart(fig)

col1, col2 = st.beta_columns(2)
with col1:
    st.subheader("Metrics and Ratios")
    st.dataframe(day_ratios_df)
with col2:
    st.subheader("Financial Statement Line Items")
    st.dataframe(day_financials_df)

st.subheader("Earnings")
st.bokeh_chart(Tabs(tabs=[ep1, ep2, ep3, ep4, ep5, ep6]))

st.subheader("Financial Position")
st.bokeh_chart(Tabs(tabs=[fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10]))

st.subheader("Cash Flows")
st.bokeh_chart(Tabs(tabs=[cp1, cp2, cp3, cp4, cp5]))
