import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
import urllib3
from datetime import datetime, timedelta

# === System Config ===
st.set_page_config(page_title="Stock Guardian AI (Wall St. Edition v2.1)", layout="wide")
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# === Global Constants ===
MA_SHORT = 20  # 月線
MA_MID = 60    # 季線

# === Module 1: Data Access Layer (Crawler) ===
class TWSE_Crawler:
    def __init__(self):
        self.base_url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    
    def fetch_real_chips(self, stock_id):
        try:
            date_str = datetime.now().strftime('%Y%m%d')
            params = {'date': date_str, 'selectType': 'ALL', 'response': 'json'}
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'}
            
            # 加入 verify=False 解決 SSL 問題
            res = requests.get(self.base_url, params=params, headers=headers, timeout=5, verify=False)
            
            if res.status_code == 200:
                data = res.json()
                if data.get('stat') == 'OK':
                    for row in data.get('data', []):
                        if row[0] == stock_id:
                            # 格式修正：API 回傳的是「股數」，需轉換為「張數」
                            # int(row[4]) // 1000
                            foreign_net = int(row[4].replace(',', '')) // 1000
                            trust_net = int(row[10].replace(',', '')) // 1000
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
        self.ticker_id = f"{stock_id}.TW"
        self.crawler = TWSE_Crawler()
        
        self.df = None
        self.info = {}
        self.q_financials = None 
        self.balance_sheet = None
        self.real_chips = None
        self.macro = {}
        
        self.scores = {'fund': 0, 'chips': 0, 'tech': 0}
        self.logs = {'fund': [], 'chips': [], 'tech': [], 'risk': []} # 初始化結構
        self.veto = False
        self.is_turnaround = False
        self.advice = {}
        self.gm_expanding = False

    def _detect_market(self):
        for suffix in ['.TW', '.TWO']:
            t = yf.Ticker(f"{self.stock_id}{suffix}")
            hist = t.history(period='5d')
            if not hist.empty:
                self.ticker_id = f"{self.stock_id}{suffix}"
                return t
        st.error(f"Ticker {self.stock_id} not found.")
        st.stop()

    def fetch_data(self):
        with st.spinner("Fetching Real-time Data..."):
            ticker = self._detect_market()
            
            # 1. Price (Ensure sorted by date)
            self.df = ticker.history(period="1y")
            
            # 2. Financials (Fix: Sort Columns by Date Descending)
            # 這能解決抓到舊年份資料的問題
            self.q_financials = ticker.quarterly_financials
            if self.q_financials is not None:
                self.q_financials = self.q_financials[sorted(self.q_financials.columns, reverse=True)]
            
            self.balance_sheet = ticker.quarterly_balance_sheet
            if self.balance_sheet is not None:
                self.balance_sheet = self.balance_sheet[sorted(self.balance_sheet.columns, reverse=True)]
                
            self.info = ticker.info
            
            # 3. Chips
            if '.TW' in self.ticker_id and '.TWO' not in self.ticker_id:
                self.real_chips = self.crawler.fetch_real_chips(self.stock_id)
            
            # 4. Macro
            try:
                self.macro['VIX'] = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
            except:
                self.macro['VIX'] = 20.0

    # --- Logic 1: Fundamental (Fix: Date Freshness Check) ---
    def analyze_fundamental(self):
        score = 0
        details = []
        
        try:
            # 顯示數據日期，確保透明度
            latest_date = self.q_financials.columns[0]
            prev_date = self.q_financials.columns[1]
            # 格式化日期顯示
            d1_str = latest_date.strftime('%Y-%m')
            d2_str = prev_date.strftime('%Y-%m')
            
            details.append(f"📅 Data Period: **{d1_str}** vs {d2_str}")

            # EPS
            eps_q1 = self.q_financials.loc['Basic EPS'].iloc[0]
            eps_q2 = self.q_financials.loc['Basic EPS'].iloc[1]
            
            # Gross Margin
            try:
                rev_q1 = self.q_financials.loc['Total Revenue'].iloc[0]
                gp_q1 = self.q_financials.loc['Gross Profit'].iloc[0]
                gm_q1 = gp_q1 / rev_q1
                
                rev_q2 = self.q_financials.loc['Total Revenue'].iloc[1]
                gp_q2 = self.q_financials.loc['Gross Profit'].iloc[1]
                gm_q2 = gp_q2 / rev_q2
            except:
                gm_q1, gm_q2 = 0, 0

            # Logic: Turnaround
            if eps_q2 < 0 and eps_q1 > 0:
                self.is_turnaround = True
                score += 4
                details.append(f"🔥 **Turnaround**: EPS {eps_q2} -> {eps_q1}")
            else:
                roe = self.info.get('returnOnEquity', 0)
                if roe > 0.15: 
                    score += 1
                    details.append(f"✅ ROE: {roe*100:.1f}%")
                
                if eps_q1 > eps_q2:
                    score += 1
                    details.append("✅ EPS Growth")

                # PEG
                pe = self.info.get('trailingPE', 0)
                growth = self.info.get('earningsGrowth', 0)
                if growth > 0 and pe > 0:
                    peg = pe / (growth * 100)
                    if peg < 1.5: 
                        score += 1
                        details.append(f"✅ PEG: {peg:.2f}")

            # GM Slope (Critical Fix)
            if gm_q1 > gm_q2:
                if not self.is_turnaround: score += 1
                details.append(f"✅ **GM Expanding**: {gm_q2*100:.1f}% -> {gm_q1*100:.1f}%")
                self.gm_expanding = True
            else:
                details.append(f"🔻 GM Contracting: {gm_q2*100:.1f}% -> {gm_q1*100:.1f}%")
                self.gm_expanding = False

        except Exception as e:
            details.append(f"⚠️ Fund. Error: {e}")
        
        self.scores['fund'] = min(4, score)
        self.logs['fund'] = details

    # --- Logic 2: Technical (Fix: Bias Logic & MA Calculation) ---
    def analyze_technical(self):
        mult = 1.0
        details = []
        
        close = self.df['Close']
        curr_price = close.iloc[-1]
        
        # 確保 MA 計算準確
        ma20 = close.rolling(MA_SHORT).mean().iloc[-1]
        ma60 = close.rolling(MA_MID).mean().iloc[-1]
        
        # 1. Bias Fix (乖離率)
        # 公式: (價格 - 均線) / 均線
        bias_60 = ((curr_price - ma60) / ma60) * 100
        
        # Debug 資訊顯示：證明軟體看到的數字是對的
        details.append(f"📏 Price: {curr_price:.1f} | MA60: {ma60:.1f}")
        
        if bias_60 > 20:
            details.append(f"🔻 High Bias: +{bias_60:.1f}% (Overheated)")
            mult *= 0.8
        elif bias_60 < -20:
            details.append(f"✅ Deep Discount: {bias_60:.1f}% (Oversold)")
        else:
            details.append(f"ℹ️ Bias: {bias_60:.1f}% (Normal)")
        
        # 2. Trend Structure
        if curr_price > ma60:
            if ma20 > ma60:
                mult = 1.2
                details.append("✅ Structure: Uptrend")
            else:
                details.append("🔸 Structure: Consolidation")
        else:
            if self.is_turnaround:
                mult = 1.0 
                details.append("ℹ️ Turnaround: Ignoring MA60 breakdown")
            else:
                mult = 0.0
                details.append("🔻 Structure: Downtrend")

        # RSI
        delta = close.diff()
        gain = (delta.where(delta>0, 0)).rolling(14).mean()
        loss = (-delta.where(delta<0, 0)).rolling(14).mean()
        rs = gain/loss
        rsi = 100 - (100/(1+rs)).iloc[-1]
        
        if rsi < 30: details.append(f"✅ RSI: {rsi:.1f} (Bottom)")
        elif rsi > 80: 
            mult *= 0.8
            details.append(f"🔻 RSI: {rsi:.1f} (Top)")
            
        self.scores['tech'] = mult
        self.advice['stop_loss'] = ma60 * 0.95
        self.logs['tech'] = details

    # --- Logic 3: Chips (Fix: Unit Conversion in Crawler) ---
    def analyze_chips(self):
        score = 0
        details = []
        
        # 1. Volume Price
        vol_5 = self.df['Volume'].rolling(5).mean().iloc[-1]
        vol_20 = self.df['Volume'].rolling(20).mean().iloc[-1]
        price_chg = self.df['Close'].pct_change(5).iloc[-1]
        
        is_vol_up = vol_5 > vol_20
        is_price_drop = price_chg < 0
        
        # 2. Institutional (Unit Fixed)
        real_buy = False
        if self.real_chips and self.real_chips['status']:
            foreign = self.real_chips['foreign']
            trust = self.real_chips['trust']
            net_buy = foreign + trust
            
            # 顯示修正後的張數
            details.append(f"🏦 Foreign: {foreign}張 | Trust: {trust}張")
            
            if net_buy > 0:
                real_buy = True
                score += 2
                details.append(f"🔥 **Smart Money**: Net Buy +{net_buy} Lots")
            else:
                details.append(f"🔻 Smart Money: Net Sell {net_buy} Lots")
        else:
            details.append("🔸 Chips Data N/A")

        # 3. Synthesis
        if is_price_drop and is_vol_up:
            if real_buy:
                details.append("✅ **Accumulation**: Price Drop + Vol Up + Insti Buy")
                if score < 2: score += 1
            else:
                details.append("🔻 Distribution: Price Drop + Vol Up + Insti Sell")
                score = 0
        
        # 4. Concentration
        volatility = self.df['Close'].pct_change().std() * np.sqrt(252)
        if volatility < 0.4:
            score = min(2, score + 0.5)
            details.append(f"✅ Low Vol: {volatility:.2f}")
            
        self.scores['chips'] = min(2, score)
        self.logs['chips'] = details

    # --- Logic 4: Risk ---
    def check_risks(self):
        logs = []
        if self.macro['VIX'] > 40:
            self.veto = True
            logs.append(f"❌ VIX: {self.macro['VIX']}")
            
        # Inventory
        try:
            inv = self.balance_sheet.loc['Inventory'].iloc[0]
            cost = self.q_financials.loc['Cost Of Revenue'].iloc[0]
            days = (inv / cost) * 90
            
            inv_prev = self.balance_sheet.loc['Inventory'].iloc[1]
            cost_prev = self.q_financials.loc['Cost Of Revenue'].iloc[1]
            days_prev = (inv_prev / cost_prev) * 90
            
            diff = (days - days_prev) / days_prev
            
            log_str = f"Inventory: {days:.0f}d (Prev: {days_prev:.0f}d, Chg: {diff*100:+.0f}%)"
            
            if diff > 0.5:
                if self.gm_expanding:
                    logs.append(f"⚠️ {log_str} -> Ignored (GM Up)")
                else:
                    self.veto = True
                    logs.append(f"❌ {log_str} -> VETO")
            else:
                logs.append(f"✅ {log_str} (Controlled)")
        except:
            logs.append("🔸 Inventory N/A")
            
        self.logs['risk'] = logs

    def run(self):
        self.fetch_data()
        self.analyze_fundamental()
        self.analyze_chips()
        self.analyze_technical()
        self.check_risks()
        
        base = self.scores['fund'] + self.scores['chips']
        final = base * self.scores['tech']
        if self.veto: final = 0
        
        return {'score': final, 'base': base, 'mult': self.scores['tech'], 'logs': self.logs, 'turnaround': self.is_turnaround}

# === UI ===
st.title("🛡️ Stock Guardian AI (Pro v2.1)")
st.caption("Fixed: Fundamental Dates | Chips Units | Technical Bias")

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
    
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base Score", f"{res['base']} / 6")
    m2.metric("Tech Mult", f"x{res['mult']:.1f}")
    
    final_color = "normal"
    if res['turnaround']: final_color = "off"
    m3.metric("Final Score", f"{res['score']:.2f} / 10", delta_color=final_color)
    m4.metric("Action", "BUY" if res['score']>=7 else "HOLD/SELL")

    if res['turnaround']:
        st.success("🔥 **TURNAROUND MODE**: Ignoring PEG constraints.")
    
    if engine.veto:
        st.error("❌ **VETO TRIGGERED**")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("Fundamental")
        for l in res['logs']['fund']: st.markdown(l)
    with c2:
        st.warning("Chips")
        for l in res['logs']['chips']: st.markdown(l)
    with c3:
        st.success("Technical")
        for l in res['logs']['tech']: st.markdown(l)
        
    with st.expander("Risk & Inventory"):
        for l in res['logs']['risk']: st.markdown(l)
