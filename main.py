import numpy as np
import sqlite3
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

#percentage change function
def percent_change(prices):
    changes=[]
    for i in range(1,len(prices)):
        change= ((prices[i] - prices[i-1])/prices[i-1])*100
        changes.append(round(change,2))
    return changes


def moving_change(prices,window_size):
    averages = [None] * (window_size - 1)
    for i in range(len(prices) - window_size + 1):
        window = prices[i:i + window_size]
        avg = sum(window) / window_size
        averages.append(round(avg,2))
    
    return averages


def calculate_volatility(prices):
    changes = percent_change(prices)
    volatility = np.std(changes)  
    return round(volatility, 2)


def rolling_volatility(prices, window_size):
    volatilities = [None] * (window_size - 1)
    for i in range(len(prices) - window_size + 1):
        window = prices[i:i + window_size]
        pchanges = percent_change(window)
        vol = np.std(pchanges)
        volatilities.append(round(vol, 2))
    return volatilities


def trading_signals(prices):
    changes = percent_change(prices)
    signals=[]
    for change in changes:
        if change>2:               
            signals.append("BUY")
        elif change<-2:
            signals.append("SELL")
        else:
            signals.append("HOLD")
    return signals


def portfolio_return(assetA, assetB, wA=0.5, wB=0.2):
    returns = []
    for a, b in zip(assetA, assetB):
        r = wA * a + wB * b
        returns.append(round(r, 2))
    return returns


def task(symbol, table):
    conn = sqlite3.connect("db/crypto.db", check_same_thread=False)
    query = f"SELECT Unix, Close FROM {table} ORDER BY Unix;"  
    df = pd.read_sql(query, conn)
    conn.close()

    prices = df['Close'].tolist()
    dates = df['Unix'].tolist()

    # Calculate metrics
    pchanges = percent_change(prices)
    mchanges = moving_change(prices, 3)
    volatilities = rolling_volatility(prices, 7) 
    signals = trading_signals(prices)

    # Align with dates 
    daily_data = []
    for i in range(1, len(prices)):
        entry = {
            "symbol": symbol,
            "date": dates[i],               
            "percent_change": pchanges[i-1],
            "signal": signals[i-1],
            "moving_avg": mchanges[i-1],
            "volatility": volatilities[i-1]
        }
        daily_data.append(entry)

    daily_df = pd.DataFrame(daily_data)
    return daily_df


if __name__ =="__main__":
    
    coins = [("BTC", "btc_prices"), ("ETH", "eth_prices")]

    # Run coins in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda args: task(*args), coins))
        
    conn = sqlite3.connect("db/crypto.db")
    for df in results:
        df.to_sql("crypto_metrics", conn, if_exists="append", index=False)
        print(f"Inserted {len(df)} rows for {df['symbol'].iloc[0]}")
    conn.close()
    

    print("\n All daily metrics stored in crypto_metrics table using parallel tasking!")
