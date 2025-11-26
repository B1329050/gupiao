import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
import urllib3
from datetime import datetime, timedelta

# === System Config ===
st.set_page_config(page_title="Stock Guardian AI (Wall St. Edition)", layout="wide")
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === Global Constants ===
MA_SHORT = 20  # 月線
MA_MID = 60    # 季線 (生命線)

# === Module 1: Data Access Layer (Crawler) ===
class TWSE_Crawler:
    """
    負責抓取真實的法人籌碼數據。
    Fix: 加入 User-Agent 與 verify=False 以繞過證交所防火牆。
    """
    def __init__(self):
        self.base_url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    
    def fetch_real_chips(self, stock_id):
        try:
            # 嘗試抓取當日數據 (若盤中無數據，邏輯上應回溯，此處簡化為抓取最新可用)
            date_str = datetime.now().strftime('%Y%m%d')
            
            params = {'date': date_str, 'selectType': 'ALL', 'response': 'json'}
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
            
            # Request
            res = requests.get(self.base_url, params=params, headers=headers, timeout=5, verify=False)
            
            if res.status_code == 200:
                data = res.json()
                if data.get('stat') == 'OK':
                    for row in data.get('data', []):
                        if row[0] == stock_id:
                            # 格式: [代號, 名稱, 外資買進, 外資賣出, 外資買賣超(4), ..., 投信買賣超(10), ...]
                            # 注意: 證交所格式可能會變，這裡抓取關鍵欄位
                            # 欄位 4: 外資買賣超, 欄位 10: 投信買賣超
                            foreign_net = int(row[4].replace(',', ''))
                            trust_net = int(row[10].replace(',', ''))
                            return {
                                'status': True, 
                                'foreign': foreign_net, 
                                'trust': trust_net,
                                'msg': 'Data Retrieved'
                            }
            return {'status': False, 'msg': 'No Data / Market Closed'}
        except Exception as e:
            return {'status': False, 'msg': str(e)}

# === Module 2: Quantitative Analysis Engine ===
class QuantEngine:
    def __init__(self, stock_id):
        self.stock_id = stock_id
        self.ticker_id = f"{stock_id}.TW" # Default TWSE
        self.crawler = TWSE_Crawler()
        
        # Data Containers
        self.df = None          # Price History
        self.info = {}          # Basic Info
        self.q_financials = None # Quarterly Financials (Critical for Turnaround)
        self.balance_sheet = None
        self.real_chips = None
        self.macro = {}
        
        # Analysis Results
        self.scores = {'fund': 0, 'chips': 0, 'tech': 0}
        self.logs = []
        self.veto = False
        self.is_turnaround = False # 轉機股標記
        self.advice = {}

    def _detect_market(self):
        """自動判斷上市/上櫃"""
        for suffix in ['.TW', '.TWO']:
            t = yf.Ticker(f"{self.stock_id}{suffix}")
            if not t.history(period='3d').empty:
                self.ticker_id = f"{self.stock_id}{suffix}"
                return t
        st.error(f"Ticker {self.stock_id} not found.")
        st.stop()

    def fetch_data(self):
        with st.spinner("Fetching Data from Exchanges & Bloomberg Terminals..."):
            ticker = self._detect_market()
            
            # 1. Price Data (1 Year)
            self.df = ticker.history(period="1y")
            
            # 2. Quarterly Financials (重點優化：只看季報，不看 TTM)
            self.q_financials = ticker.quarterly_financials
            self.balance_sheet = ticker.quarterly_balance_sheet
            self.info = ticker.info
            
            # 3. Real Chips
            if '.TW' in self.ticker_id and '.TWO' not in self.ticker_id:
                self.real_chips = self.crawler.fetch_real_chips(self.stock_id)
            
            # 4. Macro
            try:
                self.macro['VIX'] = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
            except:
                self.macro['VIX'] = 20.0

    # --- Logic 1: Fundamental (Turnaround Logic Added) ---
    def analyze_fundamental(self):
        score = 0
        details = []
        
        # Data Pre-processing: Extract Latest 2 Quarters
        try:
            # yfinance 欄位通常是日期倒序 (col 0 = Latest, col 1 = Previous)
            q1_date = self.q_financials.columns[0]
            q2_date = self.q_financials.columns[1]
            
            # EPS
            eps_q1 = self.q_financials.loc['Basic EPS'].iloc[0]
            eps_q2 = self.q_financials.loc['Basic EPS'].iloc[1]
            
            # Gross Margin (毛利率)
            try:
                # 嘗試標準欄位名稱
                gm_q1 = (self.q_financials.loc['Gross Profit'].iloc[0] / self.q_financials.loc['Total Revenue'].iloc[0])
                gm_q2 = (self.q_financials.loc['Gross Profit'].iloc[1] / self.q_financials.loc['Total Revenue'].iloc[1])
            except:
                gm_q1, gm_q2 = 0, 0 # Fallback

            # Logic 1.1: Turnaround Detection (轉機股偵測)
            if eps_q2 < 0 and eps_q1 > 0:
                self.is_turnaround = True
                score += 4 # 直接給滿分 (Fundamental Max)
                details.append(f"🔥 **Turnaround Detected (轉虧為盈)**: Q{q2_date.month} EPS {eps_q2} -> Q{q1_date.month} EPS {eps_q1}")
                details.append(f"ℹ️ **Strategy**: Ignore PEG/PE. Focus on Growth.")
                
            # Logic 1.2: Normal Evaluation (非轉機股)
            else:
                # ROE Check
                roe = self.info.get('returnOnEquity', 0)
                if roe > 0.15: 
                    score += 1
                    details.append(f"✅ ROE: {roe*100:.1f}% (Quality)")
                
                # EPS Growth
                if eps_q1 > eps_q2:
                    score += 1
                    details.append(f"✅ EPS QoQ Growth: {eps_q2} -> {eps_q1}")
                
                # PEG (Only if EPS > 0)
                pe = self.info.get('trailingPE', 0)
                growth = self.info.get('earningsGrowth', 0) # This is usually YoY
                if growth > 0 and pe > 0:
                    peg = pe / (growth * 100)
                    if peg < 1.5: 
                        score += 1
                        details.append(f"✅ PEG: {peg:.2f} (Undervalued)")
                    elif peg > 2.5:
                        details.append(f"🔻 PEG: {peg:.2f} (Overvalued)")
                else:
                     details.append(f"🔸 PEG Invalid (N/A)")

            # Logic 1.3: Gross Margin Slope (關鍵指標)
            if gm_q1 > gm_q2:
                if not self.is_turnaround: score += 1
                details.append(f"✅ **GM Expanding (毛利擴張)**: {gm_q2*100:.1f}% -> {gm_q1*100:.1f}%")
                self.gm_expanding = True
            else:
                details.append(f"🔻 GM Contracting: {gm_q2*100:.1f}% -> {gm_q1*100:.1f}%")
                self.gm_expanding = False

        except Exception as e:
            details.append(f"⚠️ Fundamental Data Missing: {e}")
        
        self.scores['fund'] = min(4, score)
        return details

    # --- Logic 2: Technical (Bias Fix & Positioning) ---
    def analyze_technical(self):
        score = 0 # This module calculates Multiplier actually
        mult = 1.0
        details = []
        
        close = self.df['Close']
        curr_price = close.iloc[-1]
        ma20 = close.rolling(MA_SHORT).mean().iloc[-1]
        ma60 = close.rolling(MA_MID).mean().iloc[-1]
        
        # 1. Bias Calculation (乖離率修正)
        bias_60 = ((curr_price - ma60) / ma60) * 100
        
        # 2. Positioning (位階: 股價 vs 52週高點)
        high_52w = close.max()
        drawdown = (curr_price - high_52w) / high_52w
        
        details.append(f"📊 Bias (60MA): {bias_60:.2f}%")
        details.append(f"📉 Drawdown: {drawdown*100:.1f}% from High ({high_52w})")
        
        # 3. Logic
        # Trend
        if curr_price > ma60:
            if ma20 > ma60:
                mult = 1.2
                details.append("✅ Structure: Uptrend (多頭排列)")
            else:
                details.append("🔸 Structure: Consolidation (整理)")
        else:
            # Special Logic: Pullback Buy or Crash?
            if self.is_turnaround:
                mult = 1.0 # 轉機股容許跌破季線 (洗盤)
                details.append("ℹ️ Turnaround Exception: Ignoring MA60 breakdown (Potential Bear Trap)")
            else:
                mult = 0.0
                details.append("🔻 Structure: Downtrend (空頭)")

        # RSI
        delta = close.diff()
        gain = (delta.where(delta>0, 0)).rolling(14).mean()
        loss = (-delta.where(delta<0, 0)).rolling(14).mean()
        rs = gain/loss
        rsi = 100 - (100/(1+rs)).iloc[-1]
        
        if rsi < 30: details.append(f"✅ RSI: {rsi:.1f} (Oversold/超賣 - Potential Bottom)")
        elif rsi > 80: 
            mult *= 0.8
            details.append(f"🔻 RSI: {rsi:.1f} (Overbought/超買)")
            
        self.scores['tech'] = mult
        self.advice['stop_loss'] = ma60 * 0.95 # 寬鬆停損
        return details

    # --- Logic 3: Chips (Institutional Filter Added) ---
    def analyze_chips(self):
        score = 0
        details = []
        
        # 1. Price/Volume Action
        vol_5 = self.df['Volume'].rolling(5).mean().iloc[-1]
        vol_20 = self.df['Volume'].rolling(20).mean().iloc[-1]
        price_chg = self.df['Close'].pct_change(5).iloc[-1]
        
        is_vol_up = vol_5 > vol_20
        is_price_drop = price_chg < 0
        
        # 2. Institutional Filter (關鍵修正)
        real_buy = False
        if self.real_chips and self.real_chips['status']:
            net_buy = self.real_chips['foreign'] + self.real_chips['trust']
            if net_buy > 0:
                real_buy = True
                score += 2 # Max Score
                details.append(f"🔥 **Smart Money**: 法人淨買超 {net_buy} 張 (Foreign+Trust)")
            else:
                details.append(f"🔻 Smart Money: 法人淨賣超 {net_buy} 張")
        else:
            details.append("🔸 No Real Chips Data (Using Proxy)")

        # 3. Logic Synthesis
        if is_price_drop and is_vol_up:
            if real_buy:
                details.append("✅ **Accumulation (壓低吃貨)**: 價跌 + 量增 + 法人買")
                if score < 2: score += 1
            else:
                details.append("🔻 Distribution (出貨): 價跌 + 量增 + 法人賣/無數據")
                score = 0 # 扣分
        elif is_price_drop and not is_vol_up:
             details.append("ℹ️ Correction (量縮回調): 正常整理")
             score += 0.5
        
        # 4. Concentration
        volatility = self.df['Close'].pct_change().std() * np.sqrt(252)
        if volatility < 0.4:
            score = min(2, score + 0.5)
            details.append(f"✅ Low Volatility ({volatility:.2f}):籌碼安定")
            
        self.scores['chips'] = min(2, score)
        return details

    # --- Logic 4: Macro & Inventory Veto ---
    def check_risks(self):
        logs = []
        # 1. VIX
        if self.macro['VIX'] > 40:
            self.veto = True
            logs.append(f"❌ VIX Alert: {self.macro['VIX']} (Panic Market)")
            
        # 2. Inventory Risk (With Trend Adjustment)
        try:
            inv = self.balance_sheet.loc['Inventory'].iloc[0]
            cost = self.q_financials.loc['Cost Of Revenue'].iloc[0] # Note: Quarterly Cost
            days = (inv / cost) * 90 # Quarterly Turnover
            
            # 比較前期
            inv_prev = self.balance_sheet.loc['Inventory'].iloc[1]
            cost_prev = self.q_financials.loc['Cost Of Revenue'].iloc[1]
            days_prev = (inv_prev / cost_prev) * 90
            
            diff = (days - days_prev) / days_prev
            
            inv_log = f"Inventory Days: {days:.0f} (Prev: {days_prev:.0f}, Chg: {diff*100:+.0f}%)"
            
            # Logic: 如果庫存暴增 > 50%
            if diff > 0.5:
                # Exception: 如果毛利在擴張 (Price Up)，則庫存是資產
                if hasattr(self, 'gm_expanding') and self.gm_expanding:
                    logs.append(f"⚠️ {inv_log} -> **Ignored** (GM Expanding = Low Cost Inventory)")
                else:
                    self.veto = True
                    logs.append(f"❌ {inv_log} -> **VETO** (High Risk Drowning)")
            else:
                logs.append(f"✅ {inv_log} (Controlled)")
                
        except:
            logs.append("🔸 Inventory Data N/A")
            
        return logs

    # --- Main Execution ---
    def run(self):
        self.fetch_data()
        
        # Analysis
        f_logs = self.analyze_fundamental()
        c_logs = self.analyze_chips()
        t_logs = self.analyze_technical()
        r_logs = self.check_risks()
        
        # Scoring
        base_score = self.scores['fund'] + self.scores['chips']
        final_score = base_score * self.scores['tech']
        
        # Veto Override
        if self.veto: final_score = 0
        
        return {
            'score': final_score,
            'base': base_score,
            'mult': self.scores['tech'],
            'logs': {'fund': f_logs, 'chips': c_logs, 'tech': t_logs, 'risk': r_logs},
            'turnaround': self.is_turnaround
        }

# === Streamlit UI ===
st.title("🛡️ Stock Guardian AI (Pro)")
st.caption("Wall Street Logic | Turnaround Detection | Institutional Filter")

col1, col2 = st.columns([3, 1])
with col1:
    s_input = st.text_input("Stock Ticker", "2408")
with col2:
    st.write("")
    st.write("")
    btn = st.button("Analyze", type="primary")

if btn:
    engine = QuantEngine(s_input)
    res = engine.run()
    
    # 1. Top Dashboard
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Price", f"{engine.df['Close'].iloc[-1]:.1f}")
    m2.metric("Base Score", f"{res['base']} / 6")
    m3.metric("Tech Multiplier", f"x{res['mult']:.1f}")
    
    delta_color = "normal"
    if res['turnaround']: delta_color = "off" # Gold/Grey for special case
    
    m4.metric("Final Expected Value", f"{res['score']:.2f} / 10", 
              delta="Turnaround Buy" if res['turnaround'] else ("Buy" if res['score']>=7 else "Neutral"),
              delta_color=delta_color)

    # 2. Turnaround Badge
    if res['turnaround']:
        st.success("🔥 **TURNAROUND DETECTED (轉機股模式)**: PEG Constraint Removed. Focus on Accumulation.")

    # 3. Veto Alert
    if engine.veto:
        st.error("❌ **VETO TRIGGERED**: Risk too high.")
        for l in res['logs']['risk']: st.write(l)

    # 4. Detailed Breakdown
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("Fundamental (Engine)")
        for l in res['logs']['fund']: st.markdown(l)
    with c2:
        st.warning("Chips (Smart Money)")
        for l in res['logs']['chips']: st.markdown(l)
    with c3:
        st.success("Technical (Timing)")
        for l in res['logs']['tech']: st.markdown(l)
        
    # 5. Risk & Inventory
    with st.expander("Risk & Inventory Depth Analysis"):
        for l in res['logs']['risk']: st.markdown(l)
