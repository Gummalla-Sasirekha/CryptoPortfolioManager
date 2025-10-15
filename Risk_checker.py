import sqlite3
import numpy as np
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ==== CONFIG ==== 
DB_PATH = "db/crypto.db"   # path to your db
ASSETS = ["btc_prices", "eth_prices"]

THRESHOLDS = {
    "volatility": 0.05,      # < 5%
    "sharpe": 1.0,           # ≥ 1
    "max_drawdown": -0.20,   # ≥ -20%
    "sortino": 1.0,          # ≥ 1
    "beta": 1.2,             # < 1.2
    "max_weight": 0.5        # < 50%
}

# Email setup
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "gummalla.cvv230163@cvv.ac.in"        # Replace with your sender email
EMAIL_PASS = "aiwt xhsu utyf vtbz"                 # Use app-specific password
ALERT_TO   = "gummallasasirekha@gmail.com"         # Recipient


def fetch_data():
    """Fetch Close prices from all asset tables and combine into DataFrame"""
    conn = sqlite3.connect(DB_PATH)
    data = {}
    for asset in ASSETS:
        query = f"SELECT Date, Close FROM {asset} ORDER BY Date ASC"
        df = pd.read_sql_query(query, conn, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        data[asset] = df["Close"]
    conn.close()
    prices = pd.concat(data, axis=1)
    prices = prices.dropna(how="any")
    return prices


def compute_metrics(prices):
    """Compute all risk metrics for an equal-weight portfolio"""
    returns = prices.pct_change().dropna()
    n_assets = len(ASSETS)
    weights = np.repeat(1/n_assets, n_assets)

    # Portfolio returns
    port_ret = returns.dot(weights)

    # Volatility (annualized)
    vol = port_ret.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (port_ret.mean() / port_ret.std()) * np.sqrt(252) if port_ret.std() != 0 else 0

    # Sortino Ratio
    downside = port_ret[port_ret < 0].std()
    sortino = (port_ret.mean() / downside) * np.sqrt(252) if downside != 0 else 0

    # Max Drawdown
    cumulative = (1 + port_ret).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    mdd = drawdown.min()

    # Beta (vs BTC as benchmark)
    market = returns["btc_prices"]
    cov = np.cov(port_ret, market)[0][1]
    var = np.var(market)
    beta = cov / var if var != 0 else np.nan

    # Max weight (equal-weight portfolio)
    max_weight = weights.max()

    return {
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "beta": beta,
        "max_weight": max_weight
    }


def store_metrics(metrics):
    """Store results in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop and recreate table to avoid schema mismatch errors
    cursor.execute("DROP TABLE IF EXISTS risk_metrics")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS risk_metrics (
            date TEXT,
            volatility REAL,
            sharpe REAL,
            sortino REAL,
            max_drawdown REAL,
            beta REAL,
            max_weight REAL
        )
    """)
    conn.commit()

    cursor.execute(
        "INSERT INTO risk_metrics (date, volatility, sharpe, sortino, max_drawdown, beta, max_weight) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         metrics["volatility"], metrics["sharpe"], metrics["sortino"],
         metrics["max_drawdown"], metrics["beta"], metrics["max_weight"])
    )
    conn.commit()
    conn.close()


def send_email_alert(subject, body):
    """Send an email alert if risk metrics are violated"""
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = ALERT_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, ALERT_TO, msg.as_string())
        print("Email alert sent successfully!")
    except Exception as e:
        print("Failed to send email:", e)


def main():
    prices = fetch_data()
    metrics = compute_metrics(prices)

    print("\n=== Portfolio Risk Metrics ===")
    for k, v in metrics.items():
        if k in ["volatility", "max_drawdown", "max_weight"]:
            print(f"{k.capitalize()}: {v:.2%}")
        else:
            print(f"{k.capitalize()}: {v:.2f}")

    store_metrics(metrics)

    # Check thresholds
    violations = []
    for k, v in THRESHOLDS.items():
        if k in ["sharpe", "sortino"]:  # higher is better
            if metrics[k] < v:
                violations.append(k)
        else:  # lower is better
            if metrics[k] > v:
                violations.append(k)

    if violations:
        subject = "🚨 Crypto Risk Alert"
        body = f"The following risk rules were violated: {', '.join(violations)}\n\nFull Metrics:\n"
        for k, v in metrics.items():
            if k in ["volatility", "max_drawdown", "max_weight"]:
                body += f"{k.capitalize()}: {v:.2%}\n"
            else:
                body += f"{k.capitalize()}: {v:.2f}\n"
        send_email_alert(subject, body)


if __name__ == "__main__":
    main()
