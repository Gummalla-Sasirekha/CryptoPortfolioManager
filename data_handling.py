import pandas as pd
import sqlite3

conn = sqlite3.connect("db/crypto.db")  
cursor = conn.cursor()

btc_df = pd.read_csv("data\Binance_BTCUSDT_d.csv")
btc_df.to_sql("btc_prices", conn, if_exists="replace", index=False)

eth_df = pd.read_csv("data\Binance_ETHUSDT_d.csv")
eth_df.to_sql("eth_prices", conn, if_exists="replace", index=False)


cursor.execute("SELECT COUNT(*) FROM btc_prices")
print(btc_df.shape)   
print(btc_df.head())  

cursor.execute("SELECT COUNT(*) FROM eth_prices")
print(eth_df.shape)
print(eth_df.head())

#create table for storing metrics
cursor.execute("""
    CREATE TABLE IF NOT EXISTS crypto_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    date INTEGER,              
    percent_change REAL,
    moving_avg REAL,
    signal TEXT,
    volatility REAL
);
""")

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in DB:")
for t in tables:
    print(t[0])
    
df = pd.read_sql_query("SELECT * FROM crypto_metrics", conn)   
print(df) 

# df.to_csv("data/crypto_metrics_daily.csv", index=False)

conn.close()
