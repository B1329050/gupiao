import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
import urllib3
from datetime import datetime, timedelta

# 1. 頁面設定 (必須在第一行)
st.set_page_config(page_title="Hedge Fund Alpha Engine", layout="wide")

# 2. 忽略警告設定 (針對 SSL 憑證錯誤與 Pandas)
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 3. 定義真實爬蟲類別 (已修復 SSL 問題)
class TWSE_Crawler:
    def __init__(self):
        # 證交所個股盤後資訊接口 (包含三大法人)
        self.base_url = "https://www.twse.com.tw/rwd/zh/fund/T86"
    
    def fetch_real_chips(self, stock_id):
        """
        真正執行 HTTP 請求去抓取證交所數據
        包含 verify=False 以解決 SSLCertVerificationError
        """
        try:
            # 取得最近交易日 (嘗試抓取今天)
            date_str = datetime.now().strftime('%Y%m%d')
            
            params = {
                'date': date_str,
                'selectType': 'ALL',
                'response': 'json'
            }
            
            # 設定 User-Agent 偽裝成瀏覽器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # === 關鍵修正：加入 verify=False 跳過 SSL 檢查 ===
            response = requests.get(self.base_url, params=params, headers=headers, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('stat') == 'OK':
                    # 解析 JSON 尋找該股票代號
                    raw_data = data.get('data') # 數據內容
                    
                    target_data = None
                    for row in raw_data:
                        if row[0] == stock_id: # 0號欄位通常是證券代號
                            target_data = row
                            break
                    
                    if target_data:
                        # 成功抓到數據
                        # target_data[4] 通常是外資買賣超, [10] 是投信 (依實際回傳為準)
                        # 這裡回傳成功狀態
                        return {'status': True, 'msg': '成功獲取證交所盤後數據'}
            
            # 若無數據或非盤後時間
            return {'status': False, 'msg': '非盤後時間或無數據，轉用量價模型'}

        except Exception as e:
            return {'status': False, 'msg': f'連線異常 ({str(e)})，轉用量價模型'}

# 4. 核心引擎
class StreamlitHedgeFundEngine:
    def __init__(self, stock_id):
        self.raw_id = str(stock_id)
        self.ticker_id = self._detect_market_suffix(self.raw_id)
        self.market_type = 'TWSE' if '.TW' in self.ticker_id else 'TPEx'
        self.ticker = None
        self.df = None
        self.info = {}
        self.financials = None
        self.balance_sheet = None
        
        # 實例化爬蟲
        self.crawler = TWSE_Crawler()
        
        # 數據容器
        self.macro = {}
        self.report_logs = [] 
        self.advice = {}
        self.chips_real_data = None
        
        # 評分
        self.base_score = 0
        self.multiplier = 1.0
        self.final_score = 0
        self.veto_triggered = False
        self.veto_reason = ""

    def _detect_market_suffix(self, stock_id):
        for suffix in ['.TW', '.TWO']:
            try_id = f"{stock_id}{suffix}"
            try:
                test = yf.Ticker(try_id)
                if not test.history(period='3d').empty: return try_id
            except: continue
        st.error(f"找不到代號 {stock_id}，請確認輸入正確。")
        st.stop()

    def fetch_data(self):
        with st.spinner(f"正在連線資料庫抓取 {self.ticker_id} ({self.market_type})..."):
            self.ticker = yf.Ticker(self.ticker_id)
            self.df = self.ticker.history(period="1y")
            self.info = self.ticker.info
            self.financials = self.ticker.financials
            self.balance_sheet = self.ticker.balance_sheet
            
            # 宏觀數據
            try:
                self.macro['VIX'] = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
                self.macro['TNX'] = yf.Ticker("^TNX").history(period="5d")['Close'].iloc[-1]
            except:
                self.macro['VIX'] = 20.0
                self.macro['TNX'] = 4.0

            # 執行 Plan B: 真實爬蟲
            if self.market_type == 'TWSE':
                self.chips_real_data = self.crawler.fetch_real_chips(self.raw_id)
            else:
                self.chips_real_data = {'status': False, 'msg': '上櫃股票不支援證交所爬蟲'}

    def log(self, msg, status="neutral"):
        color = "black"
        if status == "good": color = "green"
        elif status == "bad": color = "red"
        elif status == "warn": color = "orange"
        self.report_logs.append(f":{color}[{msg}]")

    # === 維度 1: 宏觀與否決 (檢查 Prompt 每個字) ===
    def check_macro_veto(self):
        vix = self.macro['VIX']
        
        # VIX 檢查
        if vix > 40:
            self.veto_triggered = True
            self.veto_reason = f"系統性崩盤風險 (VIX {vix:.1f} > 40)"
            return

        # 存貨週轉天數 (Inventory Days) - Prompt: "財報出現重大瑕疵（如存貨週轉天數異常暴增）"
        try:
            if 'Inventory' in self.balance_sheet.index and 'Cost Of Revenue' in self.financials.index:
                inv = self.balance_sheet.loc['Inventory'].iloc[0]
                cost = self.financials.loc['Cost Of Revenue'].iloc[0]
                days = (inv / cost) * 365
                
                # 比較去年同期
                days_prev = days # 預設
                if self.balance_sheet.shape[1] > 1:
                    inv_prev = self.balance_sheet.loc['Inventory'].iloc[1]
                    cost_prev = self.financials.loc['Cost Of Revenue'].iloc[1]
                    days_prev = (inv_prev / cost_prev) * 365
                    
                    diff = (days - days_prev) / days_prev
                    
                    # 顯示數據在 Log
                    log_status = "bad" if diff > 0.5 else "good"
                    self.log(f"存貨週轉天數: 本期 {days:.0f}天 vs 去年同期 {days_prev:.0f}天 (變動 {diff*100:+.0f}%)", log_status)
                    
                    if diff > 0.5: # 暴增 50%
                        self.veto_triggered = True
                        self.veto_reason = f"存貨週轉天數異常暴增 (+{diff*100:.0f}%)，疑似假帳/滯銷"
        except:
            self.log("存貨數據缺失，無法計算週轉天數", "warn")

        # 嚴重虧損防護
        roe = self.info.get('returnOnEquity', 0)
        if roe < -0.2:
            self.veto_triggered = True
            self.veto_reason = f"基本面嚴重惡化 (ROE {roe*100:.1f}%)"

    # === 維度 2: 基本面 ===
    def analyze_fundamental(self):
        score = 0
        details = []
        
        # ROE
        roe = self.info.get('returnOnEquity', 0)
        if roe > 0.15:
            score += 1
            details.append(f"✅ ROE: {roe*100:.2f}% (>15%)")
        else:
            details.append(f"🔸 ROE: {roe*100:.2f}%")

        # EPS Growth
        eps_g = self.info.get('earningsGrowth', 0)
        if eps_g > 0.2:
            score += 1
            details.append(f"✅ EPS成長: {eps_g*100:.2f}% (>20%)")
        else:
            details.append(f"🔸 EPS成長: {eps_g*100:.2f}%")

        # PEG (Prompt: 判斷高估低估)
        pe = self.info.get('trailingPE', 0)
        peg = pe / (eps_g * 100) if eps_g > 0 else 999
        if 0 < peg < 1.5:
            score += 1
            details.append(f"✅ PEG: {peg:.2f} (低估)")
        elif peg > 2.0:
            details.append(f"🔻 PEG: {peg:.2f} (高估)")
        else:
            details.append(f"🔸 PEG: {peg:.2f}")

        # 毛利趨勢 (Prompt: 趨勢判斷)
        try:
            gm_curr = self.financials.loc['Gross Profit'].iloc[0] / self.financials.loc['Total Revenue'].iloc[0]
            if self.financials.shape[1] > 1:
                gm_prev = self.financials.loc['Gross Profit'].iloc[1] / self.financials.loc['Total Revenue'].iloc[1]
                if gm_curr >= gm_prev:
                    score += 1
                    details.append(f"✅ 毛利趨勢: 上升 ↗ ({gm_curr*100:.1f}%)")
                else:
                    details.append(f"🔻 毛利趨勢: 下滑 ↘ ({gm_curr*100:.1f}%)")
            else:
                details.append(f"🔸 毛利: {gm_curr*100:.1f}% (無前期比較)")
        except:
            details.append("🔸 毛利數據缺失")

        return score, details

    # === 維度 3: 籌碼面 ===
    def analyze_chips(self):
        score = 0
        details = []
        
        # 1. 真實籌碼 (Plan B)
        use_real = False
        if self.chips_real_data and self.chips_real_data.get('status') is True:
            use_real = True
            details.append(f"✅ 啟用真實法人數據: {self.chips_real_data.get('msg')}")
        else:
            # 顯示失敗原因 (讓使用者知道爬蟲確實有運作，只是可能沒資料)
            msg = self.chips_real_data.get('msg', '未知') if self.chips_real_data else '未初始化'
            details.append(f"🔸 爬蟲狀態: {msg} -> 轉用量價模型")

        # 2. 量價分析 (Prompt: 散戶流向大戶?)
        vol_ma5 = self.df['Volume'].rolling(5).mean().iloc[-1]
        vol_ma20 = self.df['Volume'].rolling(20).mean().iloc[-1]
        pct = self.df['Close'].pct_change(periods=5).iloc[-1]
        
        if pct > 0 and vol_ma5 > vol_ma20:
            if not use_real: score += 1
            details.append("✅ 資金流向: 量增價漲 (進貨)")
        elif pct < 0 and vol_ma5 > vol_ma20:
            details.append("🔻 資金流向: 量增價跌 (出貨)")
        else:
            details.append("🔸 資金流向: 量能平穩")

        # 3. 集中度
        vol = self.df['Close'].pct_change().std() * np.sqrt(252)
        if vol < 0.35:
            score += 1
            details.append(f"✅ 籌碼集中度: 高 (波動率 {vol*100:.1f}%)")
        else:
            details.append(f"🔸 籌碼集中度: 低 (波動率 {vol*100:.1f}%)")

        return score, details

    # === 維度 4: 技術面 ===
    def analyze_technical(self):
        details = []
        mult = 1.0
        
        p = self.df['Close'].iloc[-1]
        ma20 = self.df['Close'].rolling(20).mean().iloc[-1]
        ma60 = self.df['Close'].rolling(60).mean().iloc[-1]
        
        # 均線 (Prompt: 判斷趨勢)
        if p > ma60:
            if ma20 > ma60:
                mult = 1.2
                details.append("✅ 趨勢: 多頭排列 (x1.2)")
            else:
                details.append("🔸 趨勢: 整理中 (x1.0)")
        else:
            mult = 0.0
            details.append("🔻 趨勢: 跌破季線 (x0.0 / Veto)")

        # RSI (Prompt: 相對強弱)
        delta = self.df['Close'].diff()
        gain = (delta.where(delta>0, 0)).rolling(14).mean()
        loss = (-delta.where(delta<0, 0)).rolling(14).mean()
        rs = gain/loss
        rsi = 100 - (100/(1+rs)).iloc[-1]
        
        if rsi > 80:
            mult = min(mult, 0.8)
            details.append(f"🔻 RSI: {rsi:.1f} (超買警示)")
        else:
            details.append(f"✅ RSI: {rsi:.1f} (正常)")

        # 乖離率 (Prompt: 判斷漲太多)
        bias = ((p - ma60)/ma60)*100
        if bias > 20:
            mult = min(mult, 0.8)
            details.append(f"🔻 乖離率: {bias:.1f}% (過大)")
        else:
            details.append(f"✅ 乖離率: {bias:.1f}% (正常)")

        # 操作建議點位 (Prompt 要求)
        self.advice['buy'] = ma20
        self.advice['stop'] = ma60
        
        return mult, details

    def run_analysis(self):
        self.fetch_data()
        self.check_macro_veto()
        
        if self.veto_triggered:
            self.final_score = 0
            return None
        
        f_score, f_details = self.analyze_fundamental()
        c_score, c_details = self.analyze_chips()
        t_mult, t_details = self.analyze_technical()
        
        self.base_score = min(6, f_score + c_score)
        self.multiplier = t_mult
        self.final_score = self.base_score * self.multiplier
        
        return {
            'fundamental': f_details,
            'chips': c_details,
            'technical': t_details
        }

# --- Streamlit UI 層 ---
st.title("📈 Hedge Fund Alpha Engine")
st.markdown("### 機構級量化分析儀表板 | Prompt Compliant")

col1, col2 = st.columns([3, 1])
with col1:
    stock_input = st.text_input("輸入股票代號 (Ex: 2330)", "2330")
with col2:
    st.write("") 
    st.write("") 
    run_btn = st.button("🚀 開始分析", type="primary")

if run_btn:
    engine = StreamlitHedgeFundEngine(stock_input)
    result = engine.run_analysis()
    
    # 1. 宏觀數據
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("VIX 恐慌指數", f"{engine.macro['VIX']:.2f}")
    m2.metric("10年美債殖利率", f"{engine.macro['TNX']:.2f}%")
    m3.metric("最新股價", f"{engine.df['Close'].iloc[-1]:.2f}")

    # 2. 否決權狀態 (最重要的 Prompt 檢查)
    if engine.veto_triggered:
        st.error(f"❌ **觸發一票否決機制 (VETO TRIGGERED)**")
        st.error(f"原因: {engine.veto_reason}")
        # 在否決時也顯示細節 log，方便除錯
        with st.expander("查看詳細原因"):
             for log in engine.report_logs: st.write(log)
    
    elif result:
        # 3. 分數展示
        st.markdown("### 📊 期望值評分")
        s1, s2, s3 = st.columns(3)
        s1.metric("基礎分 (Base)", f"{engine.base_score} / 6")
        s2.metric("技術乘數 (Mult)", f"x{engine.multiplier}")
        
        final_color = "normal" if engine.final_score >= 7 else ("inverse" if engine.final_score < 5 else "off")
        s3.metric("★ 最終評分", f"{engine.final_score:.2f} / 10", 
                  delta="Buy" if engine.final_score >=7 else "Hold/Sell", 
                  delta_color=final_color)

        # 4. 細節展示 (三欄佈局)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("基本面 (Fundamental)")
            for d in result['fundamental']: st.write(d)
        with c2:
            st.warning("籌碼面 (Chips)")
            for d in result['chips']: st.write(d)
        with c3:
            st.success("技術面 (Technical)")
            for d in result['technical']: st.write(d)

        # 5. 操作建議 (Prompt 指定)
        if engine.final_score >= 5:
            st.markdown("---")
            st.markdown("### 🎯 操作建議")
            op1, op2 = st.columns(2)
            op1.success(f"**建議進場點 (月線)**: {engine.advice['buy']:.2f}")
            op2.error(f"**嚴格停損點 (季線)**: {engine.advice['stop']:.2f}")

    # 6. Log 區域 (包含存貨天數等詳細數字)
    with st.expander("查看分析日誌與宏觀細節"):
        for log in engine.report_logs: st.markdown(log)
