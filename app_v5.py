
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
import io

st.set_page_config(page_title="Aksjeanalyse – Pro v5", layout="wide")
st.title("📈 Aksjeanalyse – Pro v5 (presets, egendefinerte lister, historikk)")

# =====================
# Preset-grupper (kan redigeres)
# =====================
PRESETS = {
    "OBX (Norge)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL","AKRBP.OL",
        "TGS.OL","SUBC.OL","SALM.OL","AUTO.OL","ELK.OL","NOD.OL","PGS.OL","ADE.OL","BONHR.OL",
        "OTELLO.OL","KID.OL","NRC.OL","VAR.OL","KIT.OL","SCHB.OL","AGS.OL","NEL.OL"
    ],
    "OBX25 (eksempel)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL","AKRBP.OL","TGS.OL",
        "SUBC.OL","SALM.OL","AUTO.OL","ELK.OL","PGS.OL","ADE.OL","NOD.OL","BONHR.OL","VAR.OL","KID.OL",
        "KIT.OL","SCHB.OL","OTELLO.OL","NEL.OL","ODL.OL"
    ],
    "OBX40 (eksempel)": [
        "EQNR.OL","DNB.OL","MOWI.OL","NHY.OL","TEL.OL","ORK.OL","YAR.OL","KOG.OL","AKRBP.OL","TGS.OL","SUBC.OL",
        "SALM.OL","AUTO.OL","ELK.OL","NOD.OL","PGS.OL","ADE.OL","BONHR.OL","OTELLO.OL","KID.OL","NRC.OL","VAR.OL",
        "KIT.OL","SCHB.OL","AGS.OL","NEL.OL","EMGS.OL","ODL.OL","HEX.OL","GOGL.OL","BWLPG.OL","HUNT.OL","BORR.OL",
        "AKSO.OL","LSG.OL","AFG.OL","MPCC.OL","SATS.OL","AUSS.OL","TOM.OL"
    ],
    "USA – Megacaps": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM","XOM",
        "UNH","V","JNJ","WMT","PG","MA","AVGO","HD","MRK","PEP"
    ],
    "Europa – Blue chips": [
        "NESN.SW","NOVN.SW","ROG.SW","SAP.DE","SIE.DE","MBG.DE","ASML.AS","AD.AS","AIR.PA","OR.PA",
        "MC.PA","SAN.PA","ULVR.L","HSBA.L","SHEL.L","BP.L","BATS.L","RIO.L","AZN.L","GSK.L"
    ],
    "S&P 100 (utvalg)": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","JPM","XOM","UNH","V","JNJ","WMT","PG","MA","HD","MRK","PEP","AVGO","COST",
        "ABBV","ADBE","KO","CRM","CVX","NFLX","TMO","ACN","CSCO","MCD","AMD","ABT","LIN","WFC","DHR","TXN","PM","BMY","IBM","HON",
        "DIS","INTC","GE","MS","QCOM","CMCSA","AMAT","CAT","ORCL","PFE","GS","NOW","RTX","BLK","ADI","AMGN","ISRG","SBUX","INTU",
        "MU","MDT","LMT","SPGI","PLD","LOW","CI","GILD","DE","SYK","ELV","MMC","ZTS","TJX","SCHW","BDX","AMT","PNC","TGT","C","REGN","MO",
        "USB","BKNG","BSX","VRTX","CB","SO","PYPL","ADP","ICE","HUM","EQIX","EOG","NKE","COP","FDX","CSX","MDLZ","PGR","GM"
    ],
    "DAX40 (utvalg)": [
        "SAP.DE","SIE.DE","MBG.DE","DTE.DE","BMW.DE","ALV.DE","BAS.DE","MUV2.DE","BAYN.DE","VOW3.DE",
        "IFX.DE","ADS.DE","HEI.DE","HEN3.DE","RWE.DE","LIN.DE","PUM.DE","FME.DE","ENR.DE","DB1.DE"
    ],
    "CAC40 (utvalg)": [
        "OR.PA","MC.PA","AIR.PA","BNP.PA","SU.PA","ENGI.PA","GLE.PA","DG.PA","AI.PA","UG.PA",
        "SAN.PA","ORA.PA","KER.PA","SGO.PA","CAP.PA","STLAP.PA","ACA.PA","RNO.PA","EDEN.PA","PUB.PA"
    ],
    "FTSE100 (utvalg)": [
        "SHEL.L","BP.L","HSBA.L","AZN.L","ULVR.L","RIO.L","BATS.L","GSK.L","DGE.L","GLEN.L",
        "VOD.L","LSEG.L","BARC.L","AV.L","NG.L","BA.L","AAL.L","PHNX.L","REL.L","BTI"
    ],
    "OMX30 (utvalg)": [
        "VOLV-B.ST","ERIC-B.ST","SAND.ST","ATCO-A.ST","ATCO-B.ST","ESSITY-B.ST","SWED-A.ST","SEB-A.ST","ALFA.ST","TELIA.ST",
        "ABB.ST","HEXA-B.ST","SKF-B.ST","BOL.ST","INVE-B.ST","EVO.ST","KINV-B.ST","NDA-SE.ST","MTG-B.ST","SCA-B.ST"
    ],
    "Råvarer": ["CL=F","BZ=F","NG=F","GC=F","SI=F","HG=F","ZC=F","ZW=F","ZS=F"],
    "Valuta (Forex)": ["EURUSD=X","USDJPY=X","GBPUSD=X","AUDUSD=X","USDCAD=X","USDCHF=X","EURNOK=X","USDNOK=X","EURGBP=X"],
    "Krypto": ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD"]
}

# =====================
# Sidepanel – kontroller
# =====================
st.sidebar.header("⚙️ Innstillinger")

preset_choice = st.sidebar.selectbox("Velg preset", list(PRESETS.keys()))
if st.sidebar.button("📥 Bytt til valgt preset"):
    st.session_state['tickers_text'] = ", ".join(PRESETS[preset_choice])

# Flere presets kan aktiveres i tillegg og merges
add_presets = st.sidebar.multiselect("Legg til flere presets (merges)", list(PRESETS.keys()))
if st.sidebar.button("➕ Legg til valgte presets"):
    merged = st.session_state.get('tickers_text', "")
    extra = []
    for p in add_presets:
        extra += PRESETS[p]
    if merged.strip():
        merged_list = [t.strip().upper() for t in merged.split(",") if t.strip()]
        merged_list += extra
    else:
        merged_list = extra
    # unik + sortert
    merged_list = sorted(list(dict.fromkeys([x.upper() for x in merged_list])))
    st.session_state['tickers_text'] = ", ".join(merged_list)

# Egendefinerte lister (lagres i session_state; legg flere ved behov)
st.sidebar.subheader("📚 Egendefinerte lister")
if 'custom_lists' not in st.session_state:
    st.session_state['custom_lists'] = {}

new_list_name = st.sidebar.text_input("Navn på ny liste")
new_list_tickers = st.sidebar.text_area("Tickere i ny liste (komma/linjer)", "", height=100)
if st.sidebar.button("💾 Lagre egendefinert liste"):
    if new_list_name.strip() and new_list_tickers.strip():
        tks = [t.strip().upper() for chunk in new_list_tickers.split("\n") for t in chunk.split(",") if t.strip()]
        st.session_state['custom_lists'][new_list_name.strip()] = sorted(list(dict.fromkeys(tks)))
        st.sidebar.success(f"Lagret liste: {new_list_name} ({len(tks)} tickere)")

# Velg egendefinerte lister å bruke nå
if st.sidebar.checkbox("Bruk egendefinerte lister"):
    chosen_custom = st.sidebar.multiselect("Velg lister", list(st.session_state['custom_lists'].keys()))
    if st.sidebar.button("📥 Bytt til valgt(e) egendefinert(e)"):
        combined = []
        for name in chosen_custom:
            combined += st.session_state['custom_lists'][name]
        combined = sorted(list(dict.fromkeys(combined)))
        st.session_state['tickers_text'] = ", ".join(combined)

default_text = "AAPL, MSFT, TSLA, NVDA, EQNR.OL"
tickers_input = st.text_area("🎯 Aktive tickere (komma/linje-separert)",
                             st.session_state.get('tickers_text', default_text),
                             height=120)
tickers = [t.strip().upper() for chunk in tickers_input.split("\n") for t in chunk.split(",") if t.strip()]

# Datoperiode og terskler
colA, colB = st.columns(2)
with colA:
    start_date = st.date_input("Startdato", pd.to_datetime("2020-01-01"))
with colB:
    end_date = st.date_input("Sluttdato", pd.Timestamp.today())

c1, c2, c3 = st.columns(3)
with c1:
    buy_thr = st.slider("KJØP hvis sannsynlighet >", 0.50, 0.90, 0.65, 0.01)
with c2:
    sell_thr = st.slider("SELG hvis sannsynlighet <", 0.10, 0.50, 0.50, 0.01)
with c3:
    top_n = st.slider("Vis topp/bunn N", 3, 100, 15, 1)

st.subheader("Indikatorer (påvirker modell)")
ci1, ci2, ci3 = st.columns(3)
with ci1:
    use_ema = st.checkbox("EMA(20, 50)", value=True)
with ci2:
    use_bbands = st.checkbox("Bollinger (20, 2σ)", value=False)
with ci3:
    use_macd = st.checkbox("MACD (12,26,9)", value=False)

st.subheader("Andre valg")
c4, c5 = st.columns(2)
with c4:
    intraday_on = st.checkbox("Inkluder intradag-analysen per ticker (saktere)", value=False)
with c5:
    intraday_interval = st.selectbox("Intradag-intervall", ["30m", "1h"])

want_excel = st.checkbox("Lag Excel-rapport (flere ark)", value=True)

# =====================
# Hjelpefunksjoner
# =====================
@st.cache_data
def fetch_data(ticker, start_date, end_date):
    return yf.download(ticker, start=start_date, end=end_date, progress=False)

def compute_indicators(df: pd.DataFrame, use_ema, use_bbands, use_macd) -> pd.DataFrame:
    df = df.copy()
    df['Return_%'] = df['Close'].pct_change() * 100.0
    df['Momentum'] = df['Return_%'].shift(1)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volatility'] = df['Return_%'].rolling(10).std()

    if use_ema:
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    if use_bbands:
        ma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['BB_upper'] = ma20 + 2*std20
        df['BB_lower'] = ma20 - 2*std20
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / ma20
    if use_macd:
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['MACD'] = macd
        df['MACD_signal'] = signal
        df['MACD_hist'] = macd - signal

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    return df

def train_and_score(df: pd.DataFrame, use_ema, use_bbands, use_macd):
    features = ['RSI', 'Momentum', 'Volatility']
    if use_ema:
        features += ['EMA20','EMA50']
    if use_bbands:
        features += ['BB_width']
    if use_macd:
        features += ['MACD','MACD_signal','MACD_hist']

    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X)[:, 1]
    df = df.copy()
    df['Prob_Up'] = proba
    acc = accuracy_score(y_test, model.predict(X_test))
    return df, acc, features

@st.cache_data
def fetch_intraday(ticker: str, interval: str):
    try:
        df = yf.download(ticker, period="60d", interval=interval, progress=False)
        return df
    except Exception:
        return pd.DataFrame()

def intraday_time_of_day(df_intraday: pd.DataFrame):
    if df_intraday is None or df_intraday.empty:
        return None
    tzinfo = getattr(df_intraday.index, "tz", None)
    gb = df_intraday['Close'].groupby(df_intraday.index.time).mean()
    if gb.empty:
        return None
    t_high = gb.idxmax(); t_low = gb.idxmin()
    return {"avg_by_time": gb, "time_high": t_high, "time_low": t_low, "tz": str(tzinfo) if tzinfo else "naive/UTC-ukjent"}

def composite_score(prob, acc, w_prob=0.7, w_acc=0.3):
    if np.isnan(prob) or np.isnan(acc):
        return np.nan
    return w_prob*prob + w_acc*acc

# =====================
# Skann og ranger
# =====================
run_scan = st.button("🔎 Skann og ranger gruppen")

if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=["Date","Group","Ticker","Prob_Up","Accuracy","Recommendation","Composite"])

if run_scan:
    results = []
    progress = st.progress(0)
    status = st.empty()

    group_label = preset_choice
    if 'custom_lists' in st.session_state and st.session_state['custom_lists']:
        group_label = f"{group_label} + Custom" if group_label else "Custom"

    for i, ticker in enumerate(tickers, start=1):
        status.text(f"Henter og analyserer: {ticker} ({i}/{len(tickers)})")
        df_raw = fetch_data(ticker, start_date, end_date)
        if df_raw is None or df_raw.empty:
            results.append({"Ticker": ticker, "Prob_Up": np.nan, "Accuracy": np.nan, "Recommendation": "N/A"})
            progress.progress(i/len(tickers)); continue
        df_ind = compute_indicators(df_raw, use_ema, use_bbands, use_macd)
        if len(df_ind) < 120:
            results.append({"Ticker": ticker, "Prob_Up": np.nan, "Accuracy": np.nan, "Recommendation": "Too few samples"})
            progress.progress(i/len(tickers)); continue
        df_model, acc, used_feats = train_and_score(df_ind, use_ema, use_bbands, use_macd)
        latest_prob = float(df_model['Prob_Up'].iloc[-1])
        rec = "BUY" if latest_prob>buy_thr else ("SELL" if latest_prob<sell_thr else "HOLD")
        comp = composite_score(latest_prob, acc)
        results.append({"Ticker": ticker, "Prob_Up": latest_prob, "Accuracy": acc, "Recommendation": rec, "Composite": comp})
        progress.progress(i/len(tickers))

    status.empty(); progress.empty()

    res_df = pd.DataFrame(results)

    st.subheader("📋 Rangering – Høyest sannsynlighet for oppgang")
    st.dataframe(res_df.sort_values("Prob_Up", ascending=False).head(top_n).style.format({"Prob_Up":"{:.2%}","Accuracy":"{:.2%}"}))

    st.subheader("📋 Rangering – Høyest sannsynlighet for nedgang")
    st.dataframe(res_df.sort_values("Prob_Up", ascending=True).head(top_n).style.format({"Prob_Up":"{:.2%}","Accuracy":"{:.2%}"}))

    st.subheader("⭐ Rangering – Komposittscore (prob + accuracy)")
    st.dataframe(res_df.sort_values("Composite", ascending=False).head(top_n).style.format({"Prob_Up":"{:.2%}","Accuracy":"{:.2%}","Composite":"{:.3f}"}))

    # ===== Eksport =====
    csv_out = res_df.copy()
    csv_out['Prob_Up'] = (csv_out['Prob_Up']*100).round(2)
    csv_out['Accuracy'] = (csv_out['Accuracy']*100).round(2)
    st.download_button("⬇️ Last ned resultatliste (CSV)",
                       data=csv_out.to_csv(index=False).encode('utf-8'),
                       file_name=f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv")

    if want_excel:
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
            res_df.to_excel(writer, index=False, sheet_name="Full_list")
            res_df.sort_values("Prob_Up", ascending=False).head(top_n).to_excel(writer, index=False, sheet_name="Top_Up")
            res_df.sort_values("Prob_Up", ascending=True).head(top_n).to_excel(writer, index=False, sheet_name="Top_Down")
            res_df.sort_values("Composite", ascending=False).head(top_n).to_excel(writer, index=False, sheet_name="Top_Composite")
        st.download_button("⬇️ Last ned Excel-rapport",
                           data=xbuf.getvalue(),
                           file_name=f"scan_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ===== Historikk (lagre topp/bunn) =====
    st.markdown("### 🗓️ Historikk – lagre dagens rangering")
    colh1, colh2, colh3 = st.columns(3)
    today_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    top_up = res_df.sort_values("Prob_Up", ascending=False).head(top_n).copy()
    top_down = res_df.sort_values("Prob_Up", ascending=True).head(top_n).copy()

    def _append_history(df_part, label):
        h = df_part.copy()
        h['Date'] = today_str
        h['Group'] = f"{group_label} | {label}"
        return h[['Date','Group','Ticker','Prob_Up','Accuracy','Recommendation','Composite']]

    if colh1.button("💾 Lagre TOPP (oppgang) til historikk"):
        st.session_state['history'] = pd.concat([st.session_state['history'], _append_history(top_up, "Top_Up")], ignore_index=True)
        st.success("Lagt til historikk (Top_Up).")

    if colh2.button("💾 Lagre BUNN (nedgang) til historikk"):
        st.session_state['history'] = pd.concat([st.session_state['history'], _append_history(top_down, "Top_Down")], ignore_index=True)
        st.success("Lagt til historikk (Top_Down).")

    if colh3.button("🧹 Tøm historikk i minnet"):
        st.session_state['history'] = st.session_state['history'].iloc[0:0]
        st.warning("Historikk tømt (kun i denne økten).")

# ===== Historikkseksjon =====
st.markdown("---")
st.header("📜 Historikk")
hist = st.session_state.get('history', pd.DataFrame(columns=["Date","Group","Ticker","Prob_Up","Accuracy","Recommendation","Composite"]))
if hist.empty:
    st.info("Ingen historikk lagret ennå. Kjør en skann og bruk knappene over for å lagre.")
else:
    st.dataframe(hist.sort_values(["Date","Group","Composite"], ascending=[False, True, False]).style.format({"Prob_Up":"{:.2%}","Accuracy":"{:.2%}","Composite":"{:.3f}"}))
    st.download_button("⬇️ Last ned historikk (CSV)",
                       data=hist.to_csv(index=False).encode('utf-8'),
                       file_name="historikk_rangeringer.csv",
                       mime="text/csv")

st.caption("Datakilde: Yahoo Finance via yfinance. Historikk lagres i nettleserøkten (Session State) og kan lastes ned til fil. Modellen er forenklet – ikke investeringsråd.")
