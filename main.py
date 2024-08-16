import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ema(data, periods):
    alpha = 2 / (periods + 1)
    result = data.copy()
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result

df = pd.read_csv('data/NFLXUSUSD1440_1.csv', delimiter='\t', header=None, names=['date', 'open', 'high', 'low', 'close', 'volume'])
df['date'] = pd.to_datetime(df['date'])

# Calculating EMA12, EMA26, MACD, SIGNAL
df['EMA12'] = ema(df['close'], 12)
df['EMA26'] = ema(df['close'], 26)
df['MACD'] = df['EMA12'] - df['EMA26']
df['SIGNAL'] = ema(df['MACD'], 9)

# Finding BUY/SELL signals
#df['buy'] = np.where((df['MACD'] > df['SIGNAL']) & (df['MACD'].shift(1) < df['SIGNAL'].shift(1)), 1, 0)
#df['sell'] = np.where((df['MACD'] < df['SIGNAL']) & (df['MACD'].shift(1) > df['SIGNAL'].shift(1)), 1, 0)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12.2, 8))

ax1.plot(df['date'], df['close'], label='Close Price', alpha=0.5)
ax1.plot(df['date'], df['MACD'], label='MACD', alpha=0.5)
ax1.plot(df['date'], df['SIGNAL'], label='SIGNAL', alpha=0.5)
ax1.set_xlabel('Date')
ax1.set_ylabel('Close Price, MACD and SIGNAL')
ax1.set_title('Close Price & Buy/Sell')
ax1.legend()
ax1.grid()

ax2.plot(df['date'], df['MACD'], label='MACD', alpha=0.5)
ax2.plot(df['date'], df['SIGNAL'], label='SIGNAL', alpha=0.5)
ax2.set_xlabel('Date')
ax2.set_ylabel('MACD & SIGNAL')
ax2.set_title('MACD, SIGNAL & Buy/Sell')
ax2.legend()
ax2.grid()

plt.show()