import pandas as pd
import numpy as np
import requests
import sqlite3
from datetime import datetime
import time
import os

# ---- DB Connection ----
DB_PATH = "db/crypto.db"

# ---- Ensure DB Folder Exists ----
os.makedirs("db", exist_ok=True)


# ---- Step 0: Initialize DB (Fetch historical data if empty) ----
def init_db(coin_ids, vs_currency="usd", days=180):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for coin in coin_ids:
        safe_table = coin.replace("-", "_")  # sanitize for SQLite
        table_name = f"{safe_table}_prices"

        # Create table if not exists
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                Date TEXT PRIMARY KEY,
                Close REAL
            )
        """)
        conn.commit()

        # Check if table already has data
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"[DB] {coin.upper()} data already exists ({count} rows).")
            continue

        # Fetch from CoinGecko
        print(f"[FETCHING HISTORICAL DATA] {coin.upper()} ...")
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()["prices"]
            df = pd.DataFrame(data, columns=["timestamp", "Close"])
            df["Date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
            df = df[["Date", "Close"]].drop_duplicates("Date")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"[DB] {coin.upper()} data saved ({len(df)} rows).")
            time.sleep(1.5)
        except Exception as e:
            print(f"[ERROR] Could not fetch {coin}: {e}")

    conn.close()


# ---- Step 1: Load historical data from DB ----
def get_prices(coin_ids):
    conn = sqlite3.connect(DB_PATH)
    all_dfs = []

    for coin in coin_ids:
        safe_table = coin.replace("-", "_")
        table_name = f"{safe_table}_prices"
        query = f"SELECT Date, Close FROM {table_name} ORDER BY Date ASC"
        df = pd.read_sql(query, conn)
        df = df.rename(columns={"Close": coin, "Date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        all_dfs.append(df)

    conn.close()

    # Merge BTC and ETH by date
    prices_df = all_dfs[0]
    for df in all_dfs[1:]:
        prices_df = prices_df.merge(df, on="date", how="outer")

    prices_df = prices_df.sort_values("date").set_index("date")
    return prices_df


# ---- Step 2: Fetch Live Price ----
def fetch_live_price(coin_id, vs_currency="usd"):
    coin_map = {
        "btc": "bitcoin",
        "eth": "ethereum"
    }
    cg_id = coin_map.get(coin_id.lower(), coin_id)
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": cg_id, "vs_currencies": vs_currency}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    live_price = r.json()[cg_id][vs_currency]
    print(f"[LIVE PRICE] {coin_id.upper()}: {live_price} {vs_currency.upper()}")
    return live_price


# ---- Step 3: Sharpe + Weighting ----
def sharpe_weights(prices_df, risk_free_rate=0.0):
    returns = np.log(prices_df / prices_df.shift(1)).dropna()
    mean_returns = returns.mean()
    vol = returns.std()
    sharpe_ratios = (mean_returns - risk_free_rate) / vol
    sharpe_ratios = sharpe_ratios.clip(lower=0)

    print("\n[SHARPE RATIOS]")
    for asset, val in sharpe_ratios.items():
        print(f"{asset}: {val:.4f}")

    if sharpe_ratios.sum() > 0:
        weights = sharpe_ratios / sharpe_ratios.sum()
    else:
        weights = pd.Series([1 / len(sharpe_ratios)] * len(sharpe_ratios), index=sharpe_ratios.index)

    print("\n[WEIGHTS (Sharpe ratio normalized)]")
    for asset, val in weights.items():
        print(f"{asset}: {val:.4f}")

    return weights, returns


# ---- Step 4: Dynamic weights + portfolio return ----
def dynamic_weights_and_return(coin_ids, vs_currency="usd"):
    prices = get_prices(coin_ids)

    # Add today's live price
    latest_prices = {cid: fetch_live_price(cid, vs_currency) for cid in coin_ids}
    today = pd.DataFrame([latest_prices], index=[pd.Timestamp.today().normalize()])
    prices = pd.concat([prices, today], axis=0)

    weights, returns = sharpe_weights(prices)
    portfolio_return = np.dot(weights, returns.iloc[-1])
    print(f"\n[PORTFOLIO RETURN] Latest Day: {portfolio_return:.4%}")
    return dict(weights), portfolio_return, returns


# ---- Step 5: Stress Test ----
def stress_test(weights, n=1000):
    weights = {k.lower(): v for k, v in weights.items()}

    scenarios = {
        "Bull Market": pd.DataFrame({
            "btc": np.random.normal(0.04, 0.01, n),
            "eth": np.random.normal(0.03, 0.015, n)
        }),
        "Bear Market": pd.DataFrame({
            "btc": np.random.normal(-0.04, 0.015, n),
            "eth": np.random.normal(-0.035, 0.02, n)
        }),
        "Volatile Market": pd.DataFrame({
            "btc": np.random.normal(0.0, 0.08, n),
            "eth": np.random.normal(0.0, 0.09, n)
        })
    }

    results = {}
    for scenario, df in scenarios.items():
        wv = np.array([weights.get(col, 0) for col in df.columns])
        port_returns = df.dot(wv)
        results[scenario] = {
            "mean_return": float(port_returns.mean()),
            "volatility": float(port_returns.std()),
            "min_return": float(port_returns.min()),
            "max_return": float(port_returns.max())
        }
    return results


# ---- Step 6: Interpret Stress Test ----
def interpret_stress_test(results):
    insights = []
    bear = results["Bear Market"]
    if bear["mean_return"] < 0:
        insights.append(f"Bear Market → Avg: {bear['mean_return']:.2%}, Worst-case: {bear['min_return']:.2%}. "
                        "Portfolio faces downside but diversification cushions losses.")
    bull = results["Bull Market"]
    insights.append(f"Bull Market → Avg: {bull['mean_return']:.2%}, Best-case: {bull['max_return']:.2%}. "
                    "Portfolio captures upside effectively.")
    vol = results["Volatile Market"]
    insights.append(f"Volatile Market → Range: {vol['min_return']:.2%} to {vol['max_return']:.2%}. "
                    "Shows both strong growth opportunities and high risk.")
    return insights


# ---- Step 7: Store weights + portfolio return in DB ----
def store_weights(weights_dict, portfolio_return):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    columns_sql = ", ".join([f"{coin} REAL" for coin in weights_dict.keys()])
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS weights_history (
            date TEXT PRIMARY KEY,
            portfolio_return REAL,
            {columns_sql}
        )
    """)

    columns = ", ".join(["date", "portfolio_return"] + list(weights_dict.keys()))
    placeholders = ", ".join("?" for _ in range(len(weights_dict) + 2))
    values = [datetime.now().strftime("%Y-%m-%d"), portfolio_return] + list(weights_dict.values())

    c.execute(f"INSERT OR REPLACE INTO weights_history ({columns}) VALUES ({placeholders})", values)
    conn.commit()
    conn.close()
    print("\n[DB] Weights & Portfolio return stored successfully.")


# ---- Step 8: Run Everything ----
if __name__ == "__main__":
    # Use only BTC and ETH
    coin_ids = ["bitcoin", "ethereum"]
    init_db(coin_ids)

    short_ids = ["btc", "eth"]
    weights, port_return, returns = dynamic_weights_and_return(short_ids)

    results = stress_test(weights)
    print("\n[STRESS TEST RESULTS]")
    for scenario, stats in results.items():
        print(scenario, stats)

    insights = interpret_stress_test(results)
    print("\n[INSIGHTS]")
    for line in insights:
        print(line)

    store_weights(weights, port_return)
