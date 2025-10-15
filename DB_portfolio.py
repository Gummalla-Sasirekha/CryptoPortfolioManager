import sqlite3
import pandas as pd
from datetime import datetime, timezone
import json

def init_db():
    conn = sqlite3.connect("db/crypto.db")
    cursor = conn.cursor()

    # Portfolio summary table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        created_at TEXT,
        expected_return TEXT,
        expected_risk REAL
    );
    """)

    # Portfolio assets (weights for each currency)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id INTEGER,
        asset_name TEXT,
        weight REAL,
        FOREIGN KEY(portfolio_id) REFERENCES portfolio(id)
    );
    """)

    conn.commit()
    conn.close()


def store_portfolio(name, port_returns, expected_risk, weights_dict):
    
    conn = sqlite3.connect("db/crypto.db")
    cursor = conn.cursor()

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Insert portfolio summary
    cursor.execute(
        "INSERT INTO portfolio (name, created_at, expected_return, expected_risk) VALUES (?, ?, ?, ?)",
        (name, created_at, json.dumps(port_returns), expected_risk)
    )
    portfolio_id = cursor.lastrowid

    # Insert weights into portfolio_assets
    for asset, weight in weights_dict.items():
        cursor.execute(
            "INSERT INTO portfolio_assets (portfolio_id, asset_name, weight) VALUES (?, ?, ?)",
            (portfolio_id, asset, weight)
        )

    conn.commit()
    conn.close()
    print(f" Stored portfolio '{name}' with ID {portfolio_id}")


def fetch_portfolios():
    
    conn = sqlite3.connect("db/crypto.db")

    # Fetch portfolio summary
    portfolio_df = pd.read_sql("SELECT * FROM portfolio", conn)

    # Fetch portfolio assets
    assets_df = pd.read_sql("SELECT * FROM portfolio_assets", conn)

    conn.close()

    print("\n--- Portfolio Table ---")
    print(portfolio_df)

    print("\n--- Portfolio Assets Table ---")
    print(assets_df)

    return portfolio_df, assets_df
    
if __name__ == "__main__":
    init_db()
    fetch_portfolios()
    # conn = sqlite3.connect("db/crypto.db")
    # cursor = conn.cursor()
    # cursor.execute('DROP TABLE IF EXISTS portfolio')
    # cursor.execute('DROP TABLE IF EXISTS portfolio_assets')

    

