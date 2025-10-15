import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === Load data from DB ===
def load_data(table_name):
    conn = sqlite3.connect("db/crypto.db")
    query = f"SELECT Date, Open, Close, High, Low FROM {table_name} ORDER BY Date ASC"
    df = pd.read_sql_query(query, conn, parse_dates=["Date"])
    conn.close()
    df.set_index("Date", inplace=True)
    return df

# === Feature engineering for an asset ===
def create_features(df):
    df['Prev_Close'] = df['Close'].shift(1)
    df['Return'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close']
    df['MA3'] = df['Close'].rolling(3).mean()
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA14'] = df['Close'].rolling(14).mean()
    df.dropna(inplace=True)
    return df

# === Remove outliers - Removes extreme values beyond ±3σ to prevent them from distorting the model.===
def remove_outliers(df, col='Return', threshold=3):
    mean, std = df[col].mean(), df[col].std()
    return df[(df[col] > mean - threshold * std) & (df[col] < mean + threshold * std)]

# === Train model with Ridge Regression - Linear Regression with regularization ===
def train_predict(df, label, split_ratio=0.8, alpha=1.0):
    X = df[['Open', 'Prev_Close', 'High', 'Low', 'MA3', 'MA7', 'MA14']]
    y = df['Return']

    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=5)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # Time-based split
    split_index = int(len(X_scaled) * split_ratio)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Ridge regression
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    df['Predicted'] = np.nan
    df.iloc[split_index:, df.columns.get_loc('Predicted')] = y_test_pred

    print(f"\n=== {label} ===")
    print(f"Selected Features: {list(selected_features)}")
    print(f"Train R² Score: {model.score(X_train, y_train):.4f}")
    print(f"Test R² Score: {model.score(X_test, y_test):.4f}")
    print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.6f}")
    print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.6f}")
    print(f"Last Actual Return: {y.iloc[-1]:.4%}")
    print(f"Last Predicted Return: {df['Predicted'].iloc[-1]:.4%}")

    # Separate plot for each asset
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2)
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.title(f"{label} Predicted vs Actual Returns")
    plt.grid(True)
    plt.show()

    return df, y, model, selected_features, X

# === Main ===
def main():
    weights = {"BTC": 0.5, "ETH": 0.3, "USDC": 0.2}

    # BTC
    btc_df = create_features(load_data("btc_prices"))
    btc_df = remove_outliers(btc_df)
    btc_df, btc_y, btc_model, btc_features, btc_X = train_predict(btc_df, "BTC", alpha=0.5)

    # ETH
    eth_df = create_features(load_data("eth_prices"))
    eth_df = remove_outliers(eth_df)
    eth_df, eth_y, eth_model, eth_features, eth_X = train_predict(eth_df, "ETH", alpha=0.5)

    # === Portfolio ===
    portfolio_df = pd.DataFrame(index=btc_df.index)
    portfolio_df['Return'] = btc_y * weights["BTC"] + eth_y * weights["ETH"]
    portfolio_df['Open'] = btc_X['Open'] * weights["BTC"] + eth_X['Open'] * weights["ETH"]
    portfolio_df['Prev_Close'] = btc_X['Prev_Close'] * weights["BTC"] + eth_X['Prev_Close'] * weights["ETH"]
    portfolio_df['High'] = btc_X['High'] * weights["BTC"] + eth_X['High'] * weights["ETH"] 
    portfolio_df['Low'] = btc_X['Low'] * weights["BTC"] + eth_X['Low'] * weights["ETH"] 
    portfolio_df['MA3'] = btc_X['MA3'] * weights["BTC"] + eth_X['MA3'] * weights["ETH"] 
    portfolio_df['MA7'] = btc_X['MA7'] * weights["BTC"] + eth_X['MA7'] * weights["ETH"]
    portfolio_df['MA14'] = btc_X['MA14'] * weights["BTC"] + eth_X['MA14'] * weights["ETH"] 

    portfolio_df.dropna(inplace=True)
    split_ratio = 0.8
    split_index = int(len(portfolio_df) * split_ratio)
    X_portfolio = portfolio_df[['Open', 'Prev_Close', 'High', 'Low', 'MA3', 'MA7', 'MA14']]
    y_portfolio = portfolio_df['Return']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_portfolio)

    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y_portfolio.iloc[:split_index], y_portfolio.iloc[split_index:]

    model_portfolio = Ridge(alpha=0.5)
    model_portfolio.fit(X_train, y_train)

    y_train_pred = model_portfolio.predict(X_train)
    y_test_pred = model_portfolio.predict(X_test)

    portfolio_df['Predicted'] = np.nan
    portfolio_df.iloc[split_index:, portfolio_df.columns.get_loc('Predicted')] = y_test_pred

    print("\n=== PORTFOLIO ===")
    print(f"Train R² Score: {model_portfolio.score(X_train, y_train):.4f}")
    print(f"Test R² Score: {model_portfolio.score(X_test, y_test):.4f}")
    print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.6f}")
    print(f"Test MSE: {mean_squared_error(y_test, y_test_pred):.6f}")
    print(f"Last Actual Return: {y_portfolio.iloc[-1]:.4%}")
    print(f"Last Predicted Return: {portfolio_df['Predicted'].iloc[-1]:.4%}")

    # Separate portfolio plot
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_portfolio.min(), y_portfolio.max()], [y_portfolio.min(), y_portfolio.max()], color='red', linewidth=2)
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.title("Portfolio Predicted vs Actual Returns")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
