import streamlit as st
import numpy as np
import pandas as pd
import pickle
from urllib.request import urlopen
import json

st.title('Stock Rating Recommendation')

with open('final_tickers.pkl', 'rb') as handle:
    final_tickers = pickle.load(handle)

ticker_company = st.selectbox(
     "Select a Ticker/Company",
     (final_tickers))

ticker = ticker_company.split(' - ')[0]

ticker_dict = {}
ticker_dict["ticker"] = ticker

def get_jsonparsed_data(url):
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

#get dcf data
url = ("https://financialmodelingprep.com/api/v3/discounted-cash-flow/"+ticker+"?apikey=8c48192b57789b1b85a59db736780f87")
dcf = (get_jsonparsed_data(url))[0]
ticker_dict["dcf"] = dcf["dcf"]

#get earning surprise data
url = ("https://financialmodelingprep.com/api/v3/earnings-surprises/"+ticker+"?apikey=8c48192b57789b1b85a59db736780f87")
earnings_surprises = (get_jsonparsed_data(url))[0]
ticker_dict["actualEarningResult"] = earnings_surprises["actualEarningResult"]
ticker_dict["estimatedEarning"] = earnings_surprises["estimatedEarning"]

#get market data
import quandl
import datetime
from datetime import timedelta, date
quandl.ApiConfig.api_key = "PCQqdJWPc-fYUJQ3JbGA"

sdate = datetime.datetime.now()   # start date
edate = sdate - datetime.timedelta(days=3*365)   # end date

delta = edate - sdate       # as timedelta

days = []

for i in range(-delta.days + 365):
    day = str((sdate - timedelta(days=i)).date())
    days.append(day)

dates = ", ".join(days)

sep = quandl.get_table('SHARADAR/SEP', ticker=ticker, date=dates)
sep_df = pd.DataFrame(sep)

#get fundamental data - SF1

quarter_ends = ["-03-31", "-06-30", "-09-30", "-12-31"]

syear = datetime.datetime.now().year

calendar_dates = []

for year in range(syear-3, syear+1):
    for quarter in quarter_ends:
        calendar_date = str(year) + quarter
        calendar_dates.append(calendar_date)

dates = ", ".join(calendar_dates)
sf1 = quandl.get_table('SHARADAR/SF1', calendardate=dates, ticker=ticker)
sf1_df = pd.DataFrame(sf1)

#get fundamental data - tickers
sf1 = quandl.get_table('SHARADAR/SF1', calendardate=dates, ticker=ticker)
sf1_df = pd.DataFrame(sf1)

sf1_df = sf1_df[sf1_df.dimension=="ART"]

st.dataframe(sep_df)
st.dataframe(sf1_df)
