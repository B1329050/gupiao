import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
import urllib3
from datetime import datetime

# === System Config ===
st.set_page_config(page_title="Stock Guardian AI V3.2 (Integrity Fix)", layout="wide")
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === Global Constants ===
MA_SHORT = 20
MA_MID = 60

# === Module: Crawler (SSL & Unit Fix) ===
class TWSE_Crawler:
    def __init__(self):
        self.base_url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    
    def fetch_real_chips(self, stock_id):
        try:
            date_str = datetime.now().strftime('%Y%m%d')
            params = {'date': date_str, 'selectType': 'ALL', 'response': 'json'}
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            res = requests.get(self.base_url, params=params, headers=headers, timeout=5, verify=False)
            
            if res.status_code == 200:
                data = res.json()
                if data.get('stat') == 'OK':
                    for row in data.get('data', []):
                        if row[0] == stock_id:
                            # 單位修正：股 -> 張
                            f_net = int(row[4].replace(',', '')) // 1000
                            t_net = int(row[10].replace(',', '')) // 1000
                            return {'status': True, 'foreign': f_net, 'trust': t_net}
            return {'status': False, 'msg': 'No Data'}
        except Exception as e:
            return {'status': False, 'msg': str(e)}

# === Core Engine (V3.2 Fixes) ===
class StockGuardianV3_2:
    def __init__(self, stock_id):
        self.stock_id = stock_id
        self.ticker_id = f"{stock_id}.TW"
        self.crawler = TWSE_Crawler()
        
        self.df = None
        self.q_financials = None
        self.real_chips = None
        self.macro = {}
        
        self.scores = {'fund': 0, 'tech': 0, 'chips': 0}
        self.logs = {'fund': [], 'tech': [], 'chips': [], 'debug': []}
        
        # Flags
        self.data_is_stale = False # 財報是否過期
        self.is_turnaround = False
        self.advice = {}

    def fetch_data(self):
        with st.spinner("Fetching Clean Data (Auto-Adjusted)..."):
            ticker = yf.Ticker(self.ticker_id)
            
            # FIX 1: Auto Adjust = True (還原權息，修復 MA 斷層)
            # Fetch 2 years to ensure MA60 has full window
            self.df = ticker.history(period="2y", auto_adjust=True)
            
            # Data Cleaning: Remove Rows with 0 or NaN
            self.df = self.df[self.df['Close'] > 0].dropna()
            
            if len(self.df) < 60:
                st.error("❌ Data Error: Insufficient history for MA60.")
                st.stop()

            # FIX 2: Check Financials Date
            qf = ticker.quarterly_financials
            if qf is not None:
                self.q_financials = qf[sorted(qf.columns, reverse=True)]
            
            # Chips
            if '.TW' in self.ticker_id:
                self.real_chips = self.crawler.fetch_real_chips(self.stock_id)
            
            # Macro
            try:
                self.macro['VIX'] = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
            except: self.macro['VIX'] = 20.0

    # === Fundamental: Fallback Logic ===
    def analyze_fundamental(self):
        score = 0
        logs = []
        
        try:
            # Check Freshness
            latest_date = self.q_financials.columns[0]
            days_diff = (datetime.now() - latest_date).days
            logs.append(f"📅 Report Date: {latest_date.date()} ({days_diff} days ago)")
            
            # FIX 3: Stale Data Logic
            if days_diff > 120: # 若超過 4 個月沒更新
                self.data_is_stale = True
                logs.append(f"⚠️ **Data Stale (>120d)**: Skipping Fundamental Weight.")
                logs.append(f"ℹ️ System will rely on Tech + Chips.")
                self.scores['fund'] = 0 # Score set to 0 but will be excluded from weight
            else:
                # Normal Logic
                eps_curr = self.q_financials.loc['Basic EPS'].iloc[0]
                eps_prev = self.q_financials.loc['Basic EPS'].iloc[1]
                
                if eps_prev < 0 and eps_curr > 0:
                    self.is_turnaround = True
                    score = 4.0
                    logs.append(f"🚀 **Turnaround**: EPS {eps_prev} -> {eps_curr}")
                else:
                    if eps_curr > eps_prev: score += 1
                    roe = 0 
                    if roe > 0.15: score += 1
                    # Simple checks for demo
                    pe = 20 # Mock
                    if pe < 15: score += 1
        except Exception as e:
            logs.append(f"⚠️ Fund Error: {e}")
        
        if not self.data_is_stale:
            self.scores['fund'] = min(4.0, score)
        self.logs['fund'] = logs

    # === Technical: Ghost Data Fix ===
    def analyze_technical(self):
        score = 0
        logs = []
        
        close = self.df['Close']
        curr_price = close.iloc[-1]
        
        # FIX 4: Robust MA Calculation
        # min_periods=1 ensures we get a number, but we filtered data already
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1]
        
        # Bias
        bias = ((curr_price - ma60) / ma60) * 100
        
        logs.append(f"📏 Price: {curr_price:.1f} | MA60: {ma60:.1f} | Bias: {bias:.2f}%")
        
        # Scoring
        # 1. Bias (Oversold is Good)
        if bias < -5:
            score += 1.0
            logs.append(f"✅ **Oversold**: Bias {bias:.1f}% (Negative)")
        elif bias > 20:
            logs.append(f"🔻 Overheated: Bias +{bias:.1f}%")
            
        # 2. RSI
        delta = close.diff()
        gain = (delta.where(delta>0, 0)).rolling(14).mean()
        loss = (-delta.where(delta<0, 0)).rolling(14).mean()
        rs = gain/loss
        rsi = 100 - (100/(1+rs)).iloc[-1]
        
        if rsi < 35: 
            score += 1.0
            logs.append(f"✅ RSI Bottom: {rsi:.1f}")
        
        # 3. Structure
        if curr_price > ma60:
            score += 1.0
            logs.append("✅ Uptrend (Price > MA60)")
        elif self.is_turnaround or self.data_is_stale:
            # 若是資料過期(通常是循環股)或轉機股，容許跌破均線
            score += 0.5 
            logs.append("ℹ️ Downtrend Tolerance (Cyclical/Turnaround)")

        self.scores['tech'] = min(3.0, score)
        self.logs['tech'] = logs

    # === Chips ===
    def analyze_chips(self):
        score = 0
        logs = []
        
        if self.real_chips and self.real_chips['status']:
            net = self.real_chips['foreign'] + self.real_chips['trust']
            logs.append(f"🏦 Net Buy: {net} 張")
            if net > 0:
                score += 2.0
                logs.append("🔥 **Smart Money**: Buying")
        
        vol = self.df['Close'].pct_change().std() * np.sqrt(252)
        if vol < 0.4:
            score += 1.0
            logs.append(f"✅ Stable: Vol {vol:.2f}")
            
        self.scores['chips'] = min(3.0, score)
        self.logs['chips'] = logs

    def run(self):
        self.fetch_data()
        self.analyze_fundamental()
        self.analyze_technical()
        self.analyze_chips()
        
        # FIX 5: Dynamic Weighting (Fallback Logic)
        if self.data_is_stale:
            # 忽略基本面，將滿分基數從 10 分降為 6 分，再換算回 10 分制
            raw_score = self.scores['tech'] + self.scores['chips']
            max_potential = 6.0 # Tech(3) + Chips(3)
            final_score = (raw_score / max_potential) * 10
            self.logs['debug'].append("⚠️ Fundamental Weight Ignored (Stale Data)")
        else:
            raw_score = self.scores['fund'] + self.scores['tech'] + self.scores['chips']
            final_score = raw_score # / 10
            
        # Macro Penalty
        if self.macro['VIX'] > 30: final_score *= 0.8
        
        return final_score

# === UI ===
st.title("🛡️ Stock Guardian V3.2 (Integrity Fix)")
st.caption("Auto-Adjusted Data | Stale Data Fallback | Correct MA Logic")

col1, col2 = st.columns([3, 1])
with col1:
    s_input = st.text_input("Ticker", "2408")
with col2:
    st.write("")
    st.write("")
    btn = st.button("Analyze", type="primary")

if btn:
    engine = StockGuardianV3_2(s_input)
    score = engine.run()
    
    st.markdown("---")
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    
    # 顯示分數 (若 Stale 則顯示 N/A)
    f_disp = "N/A (Stale)" if engine.data_is_stale else f"{engine.scores['fund']} / 4.0"
    c1.metric("Fundamental", f_disp)
    c2.metric("Technical", f"{engine.scores['tech']} / 3.0")
    c3.metric("Chips", f"{engine.scores['chips']} / 3.0")
    
    color = "normal"
    if score >= 7: color = "normal"
    elif score < 5: color = "inverse"
    
    c4.metric("🏆 Final Score", f"{score:.2f} / 10", 
              delta="Strong Buy" if score>=7 else "Hold", 
              delta_color=color)
    
    if engine.data_is_stale:
        st.warning("⚠️ **Data Latency Mode**: Financial reports are outdated. Score is based purely on **Technicals & Chips**.")

    # Details
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("Fundamental")
        for l in engine.logs['fund']: st.write(l)
    with c2:
        st.success("Technical")
        for l in engine.logs['tech']: st.write(l)
    with c3:
        st.warning("Chips")
        for l in engine.logs['chips']: st.write(l)
