# The MACD Stock Market Indicator

The MACD (Moving Average Convergence Divergence) indicator is a technical analysis tool used to identify trends and buy/sell signals in financial markets.

The MACD indicator is based on the difference between two exponential moving averages (EMA) of different periods. The parameters used are EMA-12 (12-period moving average) and EMA-26 (26-period moving average). The difference between these two moving averages forms the MACD line. Additionally, a 9-period exponential moving average of the MACD line, called the signal line (SIGNAL), is used for analysis.

The MACD indicator generates buy or sell signals when the MACD line crosses the signal line. When the MACD line crosses the signal line from below, it indicates a buy signal. Conversely, when the MACD line crosses the signal line from above, it generates a sell signal.

# MACD Indicator Analysis

The analysis in the project utilized historical daily closing price data for Netflix Inc. shares over the period from February 1, 2017, to March 31, 2023. The MACD indicator was implemented in Python using the matplotlib and pandas packages.

Full report in *report.pdf* file.