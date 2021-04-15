# stock_rating

## Predicting Daily Stock Ratings

*Streamlined daily data points of financial statement, market, forecast, and analyst rating history for over 3,000 publically-traded companies. Using an artificial neural network, predicted a stock's mean daily analyst rating. Utilized Google Cloud Platform, to train and test multiple architectures. On a scale from 1 (Strong Buy) to 5 (Strong Sell), final model performed with a MAE of 0.73. Designed a Streamlit web app and deployed it via Heroku, enabling users to visualize stock performance and obtain that day's stock rating.*

---
### Links
[Medium - Blog Post](https://dunleavyjason.medium.com/predicating-daily-stock-ratings-via-artificial-neural-network-9adf4d7c5b44)

Project Presentation:

[![Project Presentation](https://www.youtube.com/watch?v=Q2mdbTExcGI)

Streamlit Demo:

[![Streamlit Demo](https://img.youtube.com/vi/w5SIXlhQuqY/0.jpg)](https://youtu.be/w5SIXlhQuqY)


---
### Motivation
My motivation for this data science project was to predict a stock’s analyst recommendation for a given stock, on a given day. Essentially, my goal was to create an algorithm that does the manual work analysts do — but with deep learning. While no algorithm can substitute the need for professional judgment in the investment decision-making process, this model can be used as a quick sanity check or temperature gauge.

---
### Models
1. Linear Regression w/ Lasso Regularization
2. Sequential Neural Network

---

### Data Used
[Zack's Analyst Ratings](https://www.quandl.com/databases/ZRH/documentation): This dataset contained two years of daily historical analyst ratings (from over 185 brokerage firms) for over 3,000 publicly traded companies.

[Shardar Equity Prices](https://www.quandl.com/databases/SEP/documentation): Updated daily, this database provides End-Of-Day (EOD) price data

[Core US Fundamentals Data](https://www.quandl.com/databases/SF1/documentation): Updated daily, this database provides up to 20 years of history, for 150 essential fundamental indicators (financial statement line items) and financial ratios, for more than 14,000 US public companies

[Financial Modeling Prep](https://financialmodelingprep.com/developer/docs/): Updated daily, this API provided daily discounted cash flow, and earnings surprise data

---

### Process
1. Fetch data from Quandl sources, calculate autoregressive features and financial ratios - [historical_data_quandl.ipynb](https://github.com/dunleavyjason/stock_rating/blob/main/historical_data_quandl.ipynb)
2. Fetch data from FinancialModelingPrep - [historical_data_financial_modeling.ipynb](https://github.com/dunleavyjason/stock_rating/blob/main/historical_data_financialmodelingprep.ipynb)
3. Merge DataFrames and prep for training - [merge_prep.ipynb](https://github.com/dunleavyjason/stock_rating/blob/main/merge_prep.ipynb)
4. Train/Test Linear Regression Model - [modeling_linear_regression.ipynb](https://github.com/dunleavyjason/stock_rating/blob/main/modeling_linear_regression.ipynb)
5. Train/Test Neural Network Model - [modeling_ann_final.ipynb](https://github.com/dunleavyjason/stock_rating/blob/main/modeling_ann_final.ipynb)
6. Create Streamlit App - [streamlit_app.py](https://github.com/dunleavyjason/stock_rating/blob/main/streamlit_app.py)

---

### Tools and Packages
1. Python
2. Pandas
3. NumPy
4. matplotlip
5. scikit-learn
6. Google Cloud Platform
7. TensorFlow
8. Keras
9. Bokeh
10. Plotly
11. Streamlit
12. Heroku

