import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
import urllib3
from datetime import datetime

# === System Config ===
st.set_page_config(page_title="Stock Guardian AI V3.0 (Full-Spectrum)", layout="wide")
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === Global Constants ===
MA_SHORT = 20  # 月線
MA_MID = 60    # 季線

# === Module: Data Access Layer (Crawler) ===
class TWSE_Crawler:
    """
    負責抓取真實的法人籌碼數據。
    """
    def __init__(self):
        self.base_url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    
    def fetch_real_chips(self, stock_id):
        try:
            # 抓取最新交易日 (簡易版：抓當日)
            date_str = datetime.now().strftime('%Y%m%d')
            params = {'date': date_str, 'selectType': 'ALL', 'response': 'json'}
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            
            # verify=False 解決 SSL 問題
            res = requests.get(self.base_url, params=params, headers=headers, timeout=5, verify=False)
            
            if res.status_code == 200:
                data = res.json()
                if data.get('stat') == 'OK':
                    for row in data.get('data', []):
                        if row[0] == stock_id:
                            # API 回傳單位為「股」，轉換為「張」 (除以 1000)
                            # col 4 = 外資, col 10 = 投信
                            f_net = int(row[4].replace(',', '')) // 1000
                            t_net = int(row[10].replace(',', '')) // 1000
                            return {'status': True, 'foreign': f_net, 'trust': t_net}
            return {'status': False, 'msg': '無盤後數據 (Market Closed/No Data)'}
        except Exception as e:
            return {'status': False, 'msg': str(e)}

# === Module: Core Analysis Engine V3.0 ===
class StockGuardianV3:
    def __init__(self, stock_id):
        self.stock_id = stock_id
        self.ticker_id = f"{stock_id}.TW"
        self.crawler = TWSE_Crawler()
        
        # Data Containers
        self.df = None
        self.info = {}
        self.q_financials = None
        self.balance_sheet = None
        self.real_chips = None
        self.macro = {}
        
        # Scoring & Reporting
        self.scores = {'fund': 0, 'tech': 0, 'chips': 0} # Raw Scores
        self.weights = {'fund': 4.0, 'tech': 3.0, 'chips': 3.0} # Max Points
        self.logs = {'fund': [], 'tech': [], 'chips': [], 'macro': [], 'risk': []}
        
        # Flags
        self.is_turnaround = False
        self.macro_coef = 1.0 # 環境係數
        self.veto = False
        self.advice = {}

    def _detect_market(self):
        for suffix in ['.TW', '.TWO']:
            t = yf.Ticker(f"{self.stock_id}{suffix}")
            if not t.history(period='5d').empty:
                self.ticker_id = f"{self.stock_id}{suffix}"
                return t
        st.error(f"Ticker {self.stock_id} not found.")
        st.stop()

    def fetch_data(self):
        with st.spinner("Initializing V3.0 Engine... Fetching Live Data..."):
            ticker = self._detect_market()
            
            # 1. Price (1y for MA60/RSI calculations)
            self.df = ticker.history(period="1y")
            
            # 2. Financials (CRITICAL FIX: Ensure Descending Sort)
            # 強制將財報按日期由新到舊排序，防止抓到舊資料
            qf = ticker.quarterly_financials
            if qf is not None:
                self.q_financials = qf[sorted(qf.columns, reverse=True)]
                
            bs = ticker.quarterly_balance_sheet
            if bs is not None:
                self.balance_sheet = bs[sorted(bs.columns, reverse=True)]
                
            self.info = ticker.info
            
            # 3. Chips
            if '.TW' in self.ticker_id and '.TWO' not in self.ticker_id:
                self.real_chips = self.crawler.fetch_real_chips(self.stock_id)
            
            # 4. Macro Indicators
            try:
                self.macro['VIX'] = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
                self.macro['TNX'] = yf.Ticker("^TNX").history(period="5d")['Close'].iloc[-1]
            except:
                self.macro['VIX'] = 20.0; self.macro['TNX'] = 4.0

    # === Module A: Macro Filter (市場環境) ===
    def analyze_macro(self):
        vix = self.macro['VIX']
        tnx = self.macro['TNX']
        logs = []
        
        # Logic: Adjust Coefficient based on VIX
        if vix > 30:
            self.macro_coef = 0.8
            logs.append(f"🌪️ **Panic Mode (VIX {vix:.1f})**: Risk-Off. Score Discounted x0.8")
        elif vix < 15:
            logs.append(f"☀️ **Greed Mode (VIX {vix:.1f})**: Market is complacent.")
        else:
            logs.append(f"☁️ **Neutral (VIX {vix:.1f})**: Normal conditions.")
            
        logs.append(f"📉 US 10Y Yield: {tnx:.2f}%")
        self.logs['macro'] = logs

    # === Module B: Fundamental (40%) ===
    def analyze_fundamental(self):
        score = 0 # Max 4.0
        logs = []
        
        try:
            # Check Data Freshness
            d1 = self.q_financials.columns[0]
            d2 = self.q_financials.columns[1]
            logs.append(f"📅 **Report Date**: {d1.date()} (Latest) vs {d2.date()}")

            # Extract Data
            eps_curr = self.q_financials.loc['Basic EPS'].iloc[0]
            eps_prev = self.q_financials.loc['Basic EPS'].iloc[1]
            
            # GM Calculation
            try:
                gm_curr = self.q_financials.loc['Gross Profit'].iloc[0] / self.q_financials.loc['Total Revenue'].iloc[0]
                gm_prev = self.q_financials.loc['Gross Profit'].iloc[1] / self.q_financials.loc['Total Revenue'].iloc[1]
            except: gm_curr, gm_prev = 0, 0

            # --- Scoring Logic (Max 4) ---
            
            # 1. Turnaround (Priority)
            if eps_prev < 0 and eps_curr > 0:
                self.is_turnaround = True
                score = 4.0 # Max Score for Turnaround
                logs.append(f"🚀 **Turnaround Detected**: EPS {eps_prev} -> {eps_curr}")
            else:
                # Normal Scoring
                # EPS Growth (1.0)
                if eps_curr > eps_prev: 
                    score += 1.0
                    logs.append(f"✅ EPS Growth: {eps_prev} -> {eps_curr}")
                
                # ROE (1.0)
                roe = self.info.get('returnOnEquity', 0)
                if roe > 0.15: 
                    score += 1.0
                    logs.append(f"✅ ROE Quality: {roe*100:.1f}%")
                
                # PEG (1.0)
                pe = self.info.get('trailingPE', 0)
                gr = self.info.get('earningsGrowth', 0)
                if gr > 0 and 0 < (pe/(gr*100)) < 1.5:
                    score += 1.0
                    logs.append("✅ PEG Undervalued")
                    
                # GM Trend (1.0) - Critical for Cyclical
                if gm_curr > gm_prev:
                    score += 1.0
                    logs.append(f"✅ GM Expanding: {gm_prev*100:.1f}% -> {gm_curr*100:.1f}%")
                else:
                    logs.append(f"🔻 GM Contracting: {gm_prev*100:.1f}% -> {gm_curr*100:.1f}%")

            self.gm_expanding = (gm_curr > gm_prev) # Flag for Inventory Check

        except Exception as e:
            logs.append(f"⚠️ Fund. Data Error: {e}")
        
        self.scores['fund'] = min(4.0, score)
        self.logs['fund'] = logs

    # === Module C: Technical (30%) - Fixed Math ===
    def analyze_technical(self):
        score = 0 # Max 3.0
        logs = []
        
        close = self.df['Close']
        curr_price = close.iloc[-1]
        
        # 1. Moving Averages
        ma20 = close.rolling(MA_SHORT).mean().iloc[-1]
        ma60 = close.rolling(MA_MID).mean().iloc[-1]
        
        # 2. Bias Calculation (Fixed Formula)
        # 確保使用最新股價與最新均線
        bias = ((curr_price - ma60) / ma60) * 100
        logs.append(f"📏 Price: {curr_price:.1f} | MA60: {ma60:.1f} | Bias: {bias:.2f}%")
        
        # --- Scoring Logic (Max 3) ---
        
        # 1. Bias/Oversold (1.0)
        # 若是負乖離過大，視為超賣機會 (加分)
        if bias < -15:
            score += 1.0
            logs.append(f"✅ **Deep Value**: Bias {bias:.1f}% (Oversold)")
        elif -15 <= bias <= 20:
            score += 0.5
            logs.append("ℹ️ Bias Normal")
        else:
            logs.append(f"🔻 Overheated: Bias +{bias:.1f}%")

        # 2. RSI Oscillator (1.0)
        delta = close.diff()
        gain = (delta.where(delta>0, 0)).rolling(14).mean()
        loss = (-delta.where(delta<0, 0)).rolling(14).mean()
        rs = gain/loss
        rsi = 100 - (100/(1+rs)).iloc[-1]
        
        if rsi < 30:
            score += 1.0
            logs.append(f"✅ **RSI Oversold ({rsi:.1f})**: Potential Rebound")
        elif rsi > 70:
            logs.append(f"🔻 RSI Overbought ({rsi:.1f})")
        else:
            score += 0.5
            logs.append(f"ℹ️ RSI Neutral ({rsi:.1f})")

        # 3. Structure (1.0)
        # 轉機股特例：若均線空頭排列但出量止跌，給予觀察分
        if curr_price > ma60:
            score += 1.0
            logs.append("✅ Structure: Price > MA60")
        elif self.is_turnaround:
             score += 0.5
             logs.append("ℹ️ Turnaround: Tolerating MA60 Breakdown")
        
        self.scores['tech'] = min(3.0, score)
        self.advice['buy'] = ma20
        self.advice['stop'] = ma60 * 0.95
        self.logs['tech'] = logs

    # === Module D: Chips (30%) - Stability & Flow ===
    def analyze_chips(self):
        score = 0 # Max 3.0
        logs = []
        
        # 1. Real Institutional Flow (Priority)
        real_buy = False
        if self.real_chips and self.real_chips['status']:
            net_buy = self.real_chips['foreign'] + self.real_chips['trust']
            logs.append(f"🏦 Foreign: {self.real_chips['foreign']}張 | Trust: {self.real_chips['trust']}張")
            
            if net_buy > 0:
                score += 2.0 # Major Weight
                real_buy = True
                logs.append(f"🔥 **Smart Money**: Net Buy +{net_buy} (Accumulation)")
            else:
                logs.append(f"🔻 Smart Money: Net Sell {net_buy}")
        else:
            logs.append("🔸 Using Proxy (No Real Data)")
            
        # 2. Concentration Proxy (Volatility)
        # 低波動 + 股價穩 = 籌碼安定
        volatility = self.df['Close'].pct_change().std() * np.sqrt(252)
        if volatility < 0.4:
            score += 1.0
            logs.append(f"✅ **Concentration**: High (Vol {volatility:.2f})")
        else:
            logs.append(f"ℹ️ Concentration: Loose (Vol {volatility:.2f})")

        self.scores['chips'] = min(3.0, score)
        self.logs['chips'] = logs

    # === Final Scoring & Veto ===
    def calculate_final_score(self):
        self.fetch_data()
        self.analyze_macro()
        self.analyze_fundamental() # 40%
        self.analyze_technical()   # 30%
        self.analyze_chips()       # 30%
        
        # Inventory Veto Logic
        try:
            inv_curr = self.balance_sheet.loc['Inventory'].iloc[0]
            cost_curr = self.q_financials.loc['Cost Of Revenue'].iloc[0]
            days_curr = (inv_curr / cost_curr) * 90
            
            inv_prev = self.balance_sheet.loc['Inventory'].iloc[1]
            cost_prev = self.q_financials.loc['Cost Of Revenue'].iloc[1]
            days_prev = (inv_prev / cost_prev) * 90
            
            diff = (days_curr - days_prev) / days_prev
            
            inv_log = f"Inventory: {days_curr:.0f}d (Prev {days_prev:.0f}d, Chg {diff*100:+.0f}%)"
            
            if diff > 0.5 and not self.gm_expanding:
                self.veto = True
                self.logs['risk'].append(f"❌ **VETO**: {inv_log} + GM Contracting")
            else:
                self.logs['risk'].append(f"✅ {inv_log} (Controlled/Expanding GM)")
                
        except: pass

        # Weighted Sum
        raw_score = self.scores['fund'] + self.scores['tech'] + self.scores['chips']
        
        # Apply Multipliers
        final_score = raw_score * self.macro_coef
        
        if self.veto: final_score = 0
        
        return final_score

# === Streamlit UI V3.0 ===
st.title("🛡️ Stock Guardian AI (V3.0 Full-Spectrum)")
st.caption("Macro Filter | RSI Divergence | Chip Concentration | Turnaround Engine")

col1, col2 = st.columns([3, 1])
with col1:
    s_input = st.text_input("Enter Stock Ticker (e.g., 2408, 2330)", "2408")
with col2:
    st.write("")
    st.write("")
    btn = st.button("🚀 Analyze V3", type="primary")

if btn:
    engine = StockGuardianV3(s_input)
    final_score = engine.calculate_final_score()
    
    # 1. Macro Header
    st.markdown("---")
    m_col1, m_col2 = st.columns([3, 1])
    with m_col1:
        for l in engine.logs['macro']: st.caption(l)
    
    # 2. Main Score Dashboard
    st.markdown("### 🎯 Final Analysis Result")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Fundamental (40%)", f"{engine.scores['fund']} / 4.0")
    sc2.metric("Technical (30%)", f"{engine.scores['tech']} / 3.0")
    sc3.metric("Chips (30%)", f"{engine.scores['chips']} / 3.0")
    
    delta_color = "normal"
    if engine.is_turnaround: delta_color = "off"
    
    sc4.metric("🏆 FINAL SCORE", f"{final_score:.2f} / 10", 
               delta="Buy" if final_score >= 7 else "Watch",
               delta_color=delta_color)

    # 3. Badges & Veto
    if engine.veto:
        st.error("❌ **VETO TRIGGERED**: High Risk detected.")
        for l in engine.logs['risk']: st.write(l)
    
    if engine.is_turnaround:
        st.success("🔥 **TURNAROUND STOCK**: Valuation constraints removed. Focus on Growth.")

    # 4. Detailed Report
    tab1, tab2, tab3 = st.tabs(["📊 Fundamental", "📈 Technical", "🏦 Chips"])
    
    with tab1:
        st.markdown("#### Engine Performance")
        for l in engine.logs['fund']: st.markdown(l)
        
    with tab2:
        st.markdown("#### Timing & Signals")
        for l in engine.logs['tech']: st.markdown(l)
        st.info(f"💡 **Buy Zone**: {engine.advice.get('buy', 0):.2f} | **Stop Loss**: {engine.advice.get('stop', 0):.2f}")
        
    with tab3:
        st.markdown("#### Smart Money Flow")
        for l in engine.logs['chips']: st.markdown(l)

    # 5. Risk Log
    with st.expander("Risk Management Logs"):
        for l in engine.logs['risk']: st.markdown(l)
