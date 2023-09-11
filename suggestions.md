# Suggestions

In this file, I write down suggestions and relevant information for improving the code. Often in the form of GPT questions and answer.

---

## Seasonality

**Question:** Do financial time series (such as NASDAQ stocks) show seasonality? If so, should seasonality be extracted for preprocessing purposes during financial forecasting?

**Answer:** Financial time series, including NASDAQ stocks, can exhibit seasonality in certain cases. Seasonality refers to patterns or fluctuations that occur at regular intervals within a given time series. These patterns may be influenced by factors such as calendar effects, economic cycles, or recurring events.

While seasonality is commonly associated with time series data like sales, weather, or economic indicators, it may not always be present in financial time series. The presence and significance of seasonality in financial data can vary depending on the specific stock, market conditions, and other factors.

When performing financial forecasting, it is important to consider whether seasonality exists and whether it should be extracted during preprocessing. Extracting seasonality can help remove repetitive patterns from the data, allowing the forecasting model to focus on capturing other relevant trends and patterns.

However, the decision to extract seasonality depends on the nature of the data and the forecasting task at hand. If seasonality is evident and has a significant impact on the target variable (e.g., stock prices), extracting it can improve the accuracy of the forecasting model. On the other hand, if seasonality is minimal or not influential, removing it may not be necessary.

It is recommended to analyze the specific financial time series data, conduct exploratory data analysis, and apply statistical techniques (e.g., autocorrelation analysis, Fourier transforms) to identify and assess the presence of seasonality. Based on the findings, appropriate preprocessing steps, including seasonality extraction, can be applied to enhance the forecasting process.