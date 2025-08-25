import streamlit as st
import yfinance as yf
import pandas as pd

st.title("📊 My Trading Hub (Prototype)")

ticker = st.text_input("Enter a ticker symbol", "AAPL")

if ticker:
    data = yf.download(ticker, period="6mo")
    st.line_chart(data["Close"])
    st.write(data.tail())
