import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Aksjeanalyse – Kjøp/Hold/Selg", layout="wide")

st.title("📈 Aksjeanalyse – Kjøp / Hold / Selg")

# --- Sidepanel ---
st.sidebar.header("⚙️ Innstillinger")
tickers_input = st.sidebar.text_input("Skriv inn tickere (komma-separert)", "AAPL, MSFT, TSLA, NVDA, NOK.OL")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Startdato", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Sluttdato", pd.Timestamp.today())

st.sidebar.info("Trykk 'Kjør analyse' for å hente data og trene modeller.")

@st.cache_data
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return df

def compute_indicators(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change() * 100
    df['Momentum'] = df['Return'].shift(1)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volatility'] = df['Return'].rolling(10).std()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df

def train_logistic(df):
    features = ['RSI', 'Momentum', 'Volatility']
    X = df[features]
    y = df['Target']
    # Tidsserie-splitt (ingen shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    df['Prob_Up'] = model.predict_proba(X)[:, 1]
    latest_prob = float(df['Prob_Up'].iloc[-1])
    return model, acc, latest_prob, df

def rating_from_prob(p):
    if p > 0.65:
        return "🟢 KJØP"
    elif p >= 0.5:
        return "🟡 HOLD"
    else:
        return "🔴 SELG"

if st.button("Kjør analyse"):
    rows = []
    plots = []
    for ticker in tickers:
        df_raw = fetch_data(ticker, start_date, end_date)
        if df_raw is None or df_raw.empty:
            st.warning(f"Ingen data for {ticker}.")
            continue
        df = compute_indicators(df_raw)
        if len(df) < 100:
            st.warning(f"For lite data for {ticker} etter beregning av indikatorer.")
            continue
        model, acc, latest_prob, df_out = train_logistic(df)
        rec = rating_from_prob(latest_prob)

        rows.append({
            "Ticker": ticker,
            "Treffsikkerhet": f"{acc*100:.2f}%",
            "Sannsynlighet (opp)": f"{latest_prob*100:.2f}%",
            "Anbefaling": rec
        })

        with st.expander(f"📊 {ticker} – detaljer"):
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(df_out.index, df_out['Close'], label='Pris')
            ax1.set_ylabel("Pris")
            ax1.set_xlabel("Dato")
            ax2 = ax1.twinx()
            ax2.plot(df_out.index, df_out['Prob_Up'], alpha=0.7, label='Sannsynlighet for stigning')
            ax2.axhline(0.5, linestyle='--', alpha=0.6)
            ax2.set_ylabel("Sannsynlighet")
            plt.title(f"{ticker} – Pris og sannsynlighet for stigning")
            st.pyplot(fig)

    if rows:
        st.subheader("📋 Oppsummering")
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"))
    else:
        st.info("Ingen resultater å vise ennå.")

st.caption("Merk: Dette er en forenklet modell for læring/illustrasjon. Ikke investeringsråd.")