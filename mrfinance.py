from crewai import Agent, Task, Crew, Process
from queryworkflow import QueryWorkFlow
from groq import Groq
from langchain_groq import ChatGroq
import numpy as np
import pandas as pd
import time
import random
import streamlit as st
from technical_indicators import TechnicalIndicators
import os
from dotenv import load_dotenv

# Keyword Definitions
keywords_definitions = {
    "Simple Moving Average (SMA)": "A calculation that takes the arithmetic mean of a given set of prices over a specific number of days in the past.",
    "Exponential Moving Average (EMA)": "A type of moving average that places a greater weight and significance on the most recent data points.",
    "Moving Average Convergence Divergence (MACD)": "A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.",
    "Relative Strength Index (RSI)": "A momentum oscillator that measures the speed and change of price movements to identify overbought or oversold conditions.",
    "Stochastic Oscillator": "A momentum indicator that compares a particular closing price to a range of its prices over a certain period.",
    "Bollinger Bands": "A volatility indicator consisting of a set of three lines drawn in relation to a security’s price, including a simple moving average and two lines representing standard deviations above and below the moving average.",
    "Average True Range (ATR)": "A volatility indicator that measures the degree of price movement for a given market, showing how much an asset moves, on average, during a given time frame.",
    "On-Balance Volume (OBV)": "A technical trading momentum indicator that uses volume flow to predict changes in stock price.",
    "Volume-Weighted Average Price (VWAP)": "A trading benchmark that gives the average price a security has traded at throughout the day, based on both volume and price.",
    "Accumulation/Distribution Line (A/D Line)": "A cumulative indicator that measures the underlying flow of money into and out of a security.",
    "Chaikin Money Flow (CMF)": "A volume-weighted average of accumulation and distribution over a specified period, used to measure buying and selling pressure.",
    "Price": "The current value at which an asset is being bought or sold in the market.",
    "Volume": "The total number of shares or contracts traded for a security during a given period of time.",
    "Close": "The last price at which a security is traded on a given trading day.",
    "High": "The highest price at which a security traded during a specific period.",
    "Low": "The lowest price at which a security traded during a specific period.",
    "True Range": "A measure of volatility that considers the most significant price movement: either the current high-low range or the distance from the previous close to the current high or low.",
    "Money Flow Volume": "A measure of the trading volume attributed to a specific price direction, combining price and volume to understand the buying and selling pressure.",
    "Standard Deviation": "A statistical measure of the dispersion or spread of a set of data points, commonly used in finance to measure market volatility.",
    "Overbought": "A condition in which an asset is believed to be trading at a higher price than its intrinsic value, often due to excessive buying pressure.",
    "Oversold": "A condition in which an asset is believed to be trading at a lower price than its intrinsic value, often due to excessive selling pressure.",
    "Divergence": "A situation where the price of an asset and a technical indicator (or different indicators) move in opposite directions, often signaling a potential reversal.",
    "Signal Line": "In MACD, the signal line is a 9-day EMA of the MACD line, used to generate buy or sell signals."
}

# Technical - Indicators
technical_indicators = {
    "trend_indicators": [
        {
            "indicator_name": "Simple Moving Average (SMA)",
            "indicator_definition": "SMA is the average price of a security over a specified period. It smooths out price fluctuations to identify trends.",
            "indicator_formula": "SMA = (P1 + P2 + ... + Pn) / n, where P is the price at each point in time, and n is the number of time periods."
        },
        {
            "indicator_name": "Moving Average Convergence Divergence (MACD)",
            "indicator_definition": "MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.",
            "indicator_formula": "MACD = 12-period EMA - 26-period EMA; Signal Line = 9-period EMA of MACD."
        }
    ],
    "momentum_indicators": [
        {
            "indicator_name": "Relative Strength Index (RSI)",
            "indicator_definition": "RSI measures the speed and change of price movements to identify overbought or oversold conditions.",
            "indicator_formula": "RSI = 100 - [100 / (1 + RS)], where RS = Average gain of up periods / Average loss of down periods (typically over 14 periods)."
        },
        {
            "indicator_name": "Stochastic Oscillator",
            "indicator_definition": "The Stochastic Oscillator compares a particular closing price to a range of prices over a certain period to gauge momentum.",
            "indicator_formula": "%K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100; %D = 3-period moving average of %K."
        }
    ],
    "volatility_indicators": [
        {
            "indicator_name": "Bollinger Bands",
            "indicator_definition": "Bollinger Bands consist of a moving average and two standard deviation lines plotted above and below it, indicating volatility.",
            "indicator_formula": "Middle Band = 20-day SMA; Upper Band = Middle Band + 2 * 20-day standard deviation; Lower Band = Middle Band - 2 * 20-day standard deviation."
        },
        {
            "indicator_name": "Average True Range (ATR)",
            "indicator_definition": "ATR measures market volatility by calculating the average of the true range over a specified period.",
            "indicator_formula": "True Range = max[(High - Low), abs(High - Previous Close), abs(Low - Previous Close)]; ATR = n-period moving average of the True Range."
        }
    ],
    "volume_indicators": [
        {
            "indicator_name": "On-Balance Volume (OBV)",
            "indicator_definition": "OBV uses volume flow to predict changes in stock price, adding volume on up days and subtracting volume on down days.",
            "indicator_formula": "OBV = Previous OBV + Volume (if the closing price is higher than the previous close) or OBV = Previous OBV - Volume (if the closing price is lower)."
        },
        {
            "indicator_name": "Volume-Weighted Average Price (VWAP)",
            "indicator_definition": "VWAP is the average price a security has traded at throughout the day, based on both volume and price.",
            "indicator_formula": "VWAP = (Cumulative Price * Volume) / Cumulative Volume."
        }
    ],
    "market_strength_indicators": [
        {
            "indicator_name": "Accumulation/Distribution Line (A/D Line)",
            "indicator_definition": "The A/D Line measures the cumulative money flow into and out of a security, helping confirm the strength of a trend.",
            "indicator_formula": "A/D Line = Previous A/D + [(Close - Low) - (High - Close)] / (High - Low) * Volume."
        },
        {
            "indicator_name": "Chaikin Money Flow (CMF)",
            "indicator_definition": "CMF measures the amount of money flow volume over a specific period to assess buying and selling pressure.",
            "indicator_formula": "CMF = Sum of Money Flow Volume over n periods / Sum of Volume over n periods, where Money Flow Volume = [(Close - Low) - (High - Close)] / (High - Low) * Volume."
        }
    ]
}

def preprocessing_data(df):
    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort the DataFrame by 'Date'
    df = df.sort_values(by='Date')

    # Reset the index
    df = df.reset_index(drop=True)
    df = df.drop(columns=['Vol.'])
    df['Change %'] = df['Change %'].str.replace('%', '').astype(float)
    columns_to_convert = ['Price', 'Open', 'High', 'Low']

    for column in columns_to_convert:
        # Remove commas if they exist and convert the column to float
        df[column] = df[column].str.replace(',', '').astype(float)
    
    return df


# Language Model Inferencing
load_dotenv("credentials.txt")
groq_api_key = os.getenv("GROQ_API_KEY")
llama3_1 = ChatGroq(temperature=0, model_name="groq/llama-3.3-70b-versatile", api_key=groq_api_key)


def main():
    st.title("Mr. Finance - An Investing Assistant who knows Technical Indicators 🏦💰")
    st.divider()
    st.write("Load historical data of stock and get advices from Mr. Finance.")
    
    # Create a file uploader widget that accepts only CSV files
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Display the DataFrame in the Streamlit app
        st.write("Here's your CSV data:")
        st.dataframe(df)
        
        st.divider()
        st.header("Advices of Mr. Finance")
        
        cleaned_df = preprocessing_data(df)
        # Technical Indicators
        ti = TechnicalIndicators(cleaned_df)
        indicators_df = ti.get_all_indicators()
        prompt = "Estimate given stock's price"
        workflow = QueryWorkFlow(query_prompt=prompt)

        trading_agent =workflow.create_agent(
            role="Finance Trading Expert", 
            goal="As a finance trading expert you must give trading advices for given stock.", 
            backstory="You are a finance trading expert who can give advices whether selling or buying is better.", 
            llm=llama3_1
        )

        trading_advice_task =workflow.create_task(
            description=f"""With knowing {keywords_definitions} and {technical_indicators},
            Forecast the given stock price using {indicators_df} (this data has prices and also indicators in it)  give trading advices.
            ** 1/1/2024 is Monday You must first apply the days to given dates.
            ** You must give price estimatations for 3 days which starts with historical_datas ending date.
            This price list must be daily estimations.
            ** You must give advices. For example: It's better to sell now or wait sometime it will be more profitable.
            ** Do not forget If it is a weekend the price will not change that day. 
            """,
            agent=trading_agent,
            expected_output="Trading advices and forecasting stock price."
        )

        crew = workflow.create_crew()
        out = crew.kickoff()
        st.write(out.raw)
        
if __name__ == "__main__":
    main()