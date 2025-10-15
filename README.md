# Crypto Investment Manager

This project provides tools to manage and analyze cryptocurrency investments.
It automates data collection, portfolio construction, risk assessment, and performance analysis — helping users make informed investment decisions.

## Core Features

* **Automated Data Pipeline**
  The system automatically imports historical price data for cryptocurrencies like **BTC and ETH** from CSV files into a centralized SQLite database.
  This forms the foundation for all analysis and portfolio calculations.

* **Strategic Portfolio Construction**
  Multiple rule-based investment strategies are implemented to create diversified portfolios.
  These include:

  * **Equal Weighting** – Distributes capital equally among all assets.
  * **Price Weighting** – Allocates based on asset price levels.
  * **Inverse Volatility** – Gives more weight to less volatile assets to reduce risk.
  * **Sharpe Ratio-based** – Focuses on maximizing returns compared to risk.

* **Risk Management**
  A dedicated module continuously monitors the portfolio’s performance and risk.
  It calculates key metrics such as:

  * **Volatility** – How much the price changes over time.
  * **Sharpe & Sortino Ratios** – Risk-adjusted return measures.
  * **Maximum Drawdown** – The largest drop from a peak value.
  * **Beta** – Measures how volatile the portfolio is compared to the overall market.
    If any metric crosses a defined limit, the system sends an alert to the user.

* **Performance and Stress Testing**
  The project can backtest strategies using historical data and show performance through visual reports.
  It also runs stress tests to simulate portfolio behavior under different market situations (like bull or bear markets).

## Tech Stack

* **Language**: Python 3
* **Data Analysis**: Pandas, NumPy
* **Database**: SQLite
* **Visualization**: Matplotlib

## Workflow

1. **Initialization** – Database schema is created using `DB_portfolio.py`.
2. **Data Loading** – CSV price data is loaded into the database via `data_handling.py`.
3. **Metric Calculation** – `main.py` processes the raw data and computes indicators like volatility and moving averages.
4. **Strategy & Allocation** – `Investment_Rule.py` and `portfolio_math.py` calculate portfolio weights and apply the chosen investment strategy.
5. **Monitoring** – `Risk_checker.py` periodically checks portfolio risk and triggers alerts when necessary.

## Module Overview

| File                 | Description                                                                       |
| -------------------- | --------------------------------------------------------------------------------- |
| `main.py`            | Processes price data and calculates technical metrics.                            |
| `data_handling.py`   | Loads cryptocurrency price data into the SQLite database.                         |
| `DB_portfolio.py`    | Manages the database schema and provides helper functions.                        |
| `Investment_Rule.py` | Defines and applies portfolio strategies, including stress testing.               |
| `portfolio_math.py`  | Handles mathematical operations for portfolio weighting and performance analysis. |
| `Risk_checker.py`    | Tracks risk metrics and sends alerts when thresholds are exceeded.                |

## Getting Started

1. **Install Python 3 and pip**
2. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd CryptoPortfolioManager
   ```
3. **Install Dependencies**

   ```bash
   pip install pandas numpy matplotlib
   ```
4. **Setup the Database**

   ```bash
   python DB_portfolio.py
   ```
5. **Load Data**

   ```bash
   python data_handling.py
   ```
6. **Run the Modules**

   ```bash
   python main.py
   python Investment_Rule.py
   python portfolio_math.py
   ```

