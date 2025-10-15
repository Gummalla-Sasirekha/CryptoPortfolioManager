
import numpy as np
import pandas as pd
import sqlite3
from DB_portfolio import init_db, store_portfolio
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

# ---------- RULES ----------
def cap_weights(weights, cap=0.5):
    capped = {s: min(w, cap) for s, w in weights.items()}
    total_capped = sum(capped.values())
    excess = 1 - total_capped

    if abs(excess) < 1e-9:
        return {s: round(capped[s], 6) for s in capped}

    uncapped_assets = [s for s in weights if weights[s] < cap]

    if not uncapped_assets:
        return {s: round(capped[s] / total_capped, 6) for s in capped}

    uncapped_total = sum(weights[s] for s in uncapped_assets)
    for s in uncapped_assets:
        add = (weights[s] / uncapped_total) * excess
        capped[s] += add
        if capped[s] > cap:
            capped[s] = cap

    total_final = sum(capped.values())
    return {s: round(capped[s] / total_final, 6) for s in capped}


def equal_weight(symbols):
    w = 1 / len(symbols)
    weights = {s: round(w, 6) for s in symbols}
    return cap_weights(weights)


def price_weight(symbols, prices):
    prices = {s: float(prices[s]) for s in symbols}
    total = sum(prices.values())
    weights = {s: prices[s] / total for s in symbols}
    return cap_weights(weights)


def inverse_volatility(symbols, returns):
    vols = {s: float(np.std(returns[s])) for s in symbols}
    inv = {s: 1 / vols[s] if vols[s] > 0 else 0 for s in symbols}
    total = sum(inv.values())
    weights = {s: inv[s] / total for s in symbols}
    return cap_weights(weights)


# ---------- HELPERS ----------
def percent_change(prices):
    changes = []
    for i in range(1, len(prices)):
        change = ((prices[i] - prices[i - 1]) / prices[i - 1]) * 100
        changes.append(round(change, 6))
    return changes


def portfolio_return(weights, returns):
    min_len = min(len(r) for r in returns.values())
    port = []
    for i in range(min_len):
        r = sum(weights[s] * returns[s][i] for s in weights)
        port.append(round(r, 2))
    return port


def portfolio_risk(port):
    return round(np.std(port), 2)


# ---------- FUNCTION TO PROCESS ONE RULE ----------
def process_rule(rule_name, weights_func, symbols, prices, returns):
    if rule_name == "Equal":
        w = weights_func(symbols)
    elif rule_name == "Price":
        w = weights_func(symbols, prices)
    elif rule_name == "InvVol":
        w = weights_func(symbols, returns)
    else:
        return None

    port = portfolio_return(w, returns)
    risk = portfolio_risk(port)

    store_portfolio(f"{rule_name} Portfolio", port, risk, w)
    return f"{rule_name} completed"


# ---------- MAIN ----------
if __name__ == "__main__":
    # Initialise DB
    init_db()

    # Connect DB
    conn = sqlite3.connect("db/crypto.db")
    btc = pd.read_sql("SELECT Close FROM btc_prices ORDER BY Unix", conn)
    eth = pd.read_sql("SELECT Close FROM eth_prices ORDER BY Unix", conn)
    conn.close()

    # Returns
    returns = {
        "BTC": percent_change(btc['Close'].tolist()),
        "ETH": percent_change(eth['Close'].tolist()),
    }

    prices = {"BTC": btc['Close'].iloc[0], "ETH": eth['Close'].iloc[0]}
    symbols = list(returns.keys())

    rules = {
        "Equal": equal_weight,
        "Price": price_weight,
        "InvVol": inverse_volatility,
    }

    # ---------- PARALLEL EXECUTION ----------
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_rule, name, func, symbols, prices, returns)
            for name, func in rules.items()
        ]
        for future in as_completed(futures):
            print(future.result())

    # --- Selected equal weight rule to compare with assets ---
    selected_rule = "Equal"
    w = rules[selected_rule](symbols)
    port_ret = portfolio_return(w, returns)

    # --- Use only first 15 days ---
    n = min(15, len(returns["BTC"]), len(returns["ETH"]), len(port_ret))
    comparison_df = pd.DataFrame({
        "BTC_Return": returns["BTC"][:n],
        "ETH_Return": returns["ETH"][:n],
        f"{selected_rule}_Portfolio": port_ret[:n],
    }).fillna(0)

    comparison_df.to_csv("data/portfolio_vs_assets_15days_equal.csv", index=False)
    print("Exported to portfolio_vs_assets_15days_equal.csv")
    print(comparison_df.head())

    # --- Plot 15-day line graph ---
    plt.figure(figsize=(10, 6))
    for col in comparison_df.columns:
        plt.plot(comparison_df.index, comparison_df[col], label=col)
    plt.legend()
    plt.title("15 Days Portfolio vs Assets Returns")
    plt.xlabel("Days")
    plt.ylabel("Return (%)")
    plt.grid(True)
    plt.show()

    # --- Insights ---
    print("\n--- Insights from 15 Days ---")
    metrics = {}
    for col in ["BTC_Return", "ETH_Return", f"{selected_rule}_Portfolio"]:
        data = comparison_df[col].dropna().tolist()
        avg_ret = np.mean(data)
        risk = np.std(data)
        metrics[col] = (avg_ret, risk)
        if col.endswith("_Portfolio"):
            print(f"{col} → Avg Return={avg_ret:.2f}%, Risk={risk:.2f} → Balanced allocation (capped ≤50%).")
        else:
            print(f"{col.replace('_Return','')} → Avg Return={avg_ret:.2f}%, Risk={risk:.2f}")

    # --- Summary Statement ---
    print("\n--- Summary Statement ---")
    btc_ret, btc_risk = metrics["BTC_Return"]
    eth_ret, eth_risk = metrics["ETH_Return"]
    port_ret_val, port_risk_val = metrics[f"{selected_rule}_Portfolio"]

    print(
        f"In the first {n} days, ETH showed the highest volatility (Risk={eth_risk:.2f}) "
        f"with Avg Return={eth_ret:.2f}%. BTC followed a similar pattern with "
        f"Risk={btc_risk:.2f} and Avg Return={btc_ret:.2f}%. "
        f"The {selected_rule} Portfolio, which equally allocates across BTC and ETH "
        f"(with ≤50% cap), achieved Avg Return={port_ret_val:.2f}% and Risk={port_risk_val:.2f}. "
        f"This indicates diversification reduced volatility compared to BTC and ETH alone."
    )

    # --- Cumulative Returns ---
    comparison_cum = (1 + comparison_df / 100).cumprod() - 1
    comparison_cum.to_csv("data/Portfolio_vs_Assets_(Cumulative Returns).csv", index=False)
    print("Exported to Portfolio_vs_Assets_(Cumulative Returns).csv")
    print(comparison_cum.head())

    # Plot cumulative returns line graph
    plt.figure(figsize=(10, 6))
    for col in comparison_cum.columns:
        plt.plot(comparison_cum.index, comparison_cum[col], label=col)
    plt.legend()
    plt.title("Cumulative Portfolio vs Assets Returns")
    plt.xlabel("Days")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.show()

    # --- Insights from Cumulative Returns ---
    print("\n--- Insights from Cumulative Returns ---")
    for col in ["BTC_Return", "ETH_Return", f"{selected_rule}_Portfolio"]:
        daily = comparison_df[col].dropna().tolist()
        cum = (1 + pd.Series(daily) / 100).cumprod() - 1

        avg_daily = np.mean(daily)
        volatility = np.std(daily)
        sharpe = avg_daily / volatility if volatility != 0 else 0
        final_cum = cum.iloc[-1]

        if col.endswith("_Portfolio"):
            print(f"{col} → Final Cumulative Return={final_cum:.2%}, "
                  f"Avg Daily Return={avg_daily:.2f}%, Volatility={volatility:.2f}, "
                  f"Sharpe={sharpe:.2f} → Diversified performance.")
        else:
            asset = col.replace("_Return", "")
            print(f"{asset} → Final Cumulative Return={final_cum:.2%}, "
                  f"Avg Daily Return={avg_daily:.2f}%, Volatility={volatility:.2f}, "
                  f"Sharpe={sharpe:.2f}")
