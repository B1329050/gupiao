import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- 頁面設定 ---
st.set_page_config(page_title="Stock Guardian AI v3.0", page_icon="🛡️", layout="wide")

# --- 核心分析類別 (修改為適配 Streamlit) ---
class StockAnalystAI:
    def __init__(self, ticker):
        self.ticker_symbol = f"{ticker}.TW" if not ticker.endswith('.TW') and not ticker.isdigit() == False else f"{ticker}.TW"
        # 簡單處理輸入，若輸入 2408 自動變 2408.TW
        if ticker.isdigit():
             self.ticker_symbol = f"{ticker}.TW"
        else:
             self.ticker_symbol = ticker

    def fetch_data(self):
        """抓取歷史數據"""
        try:
            stock = yf.Ticker(self.ticker_symbol)
            df = stock.history(period="1y")
            if df.empty:
                st.error(f"❌ 找不到股票代號: {self.ticker_symbol}，請確認輸入正確。")
                return None
            return df
        except Exception as e:
            st.error(f"連線錯誤: {e}")
            return None

    def calculate_technicals(self, df):
        """計算技術指標"""
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # 乖離率計算
        df['Bias_60'] = (df['Close'] - df['MA60']) / df['MA60'] * 100
        
        # RSI 計算
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def run_analysis(self, df, eps_status, gm_status, chips_vol):
        """執行評分邏輯"""
        data = df.iloc[-1] # 最新一筆
        price = data['Close']
        ma60 = data['MA60']
        bias_60 = data['Bias_60']
        rsi = data['RSI']
        
        score = 0
        report_logs = []
        
        # --- 1. 基本面 (40%) ---
        fund_score = 0
        if eps_status == 'Turnaround (轉虧為盈)':
            fund_score += 2.0
            report_logs.append("✅ [基本面] EPS 轉虧為盈 (強烈買進訊號)")
        elif eps_status == 'Growth (成長)':
            fund_score += 1.5
            report_logs.append("✅ [基本面] EPS 持續成長")
        
        if gm_status == 'Up (上升)':
            fund_score += 2.0
            report_logs.append("✅ [基本面] 毛利率回升 (護城河變寬)")
        else:
            report_logs.append("🔻 [基本面] 毛利率下滑 (扣分)")
            
        score += fund_score
        
        # --- 2. 技術面 (30%) ---
        tech_score = 0
        
        # 乖離率邏輯
        if bias_60 < -10:
            tech_score += 1.5
            report_logs.append(f"✅ [技術面] 負乖離過大 ({bias_60:.2f}%)，超賣有反彈空間")
        elif abs(bias_60) < 5:
            tech_score += 1.0
            report_logs.append(f"ℹ️ [技術面] 股價貼近季線 ({bias_60:.2f}%)，方向待變")
        elif bias_60 > 20:
            tech_score -= 1.0
            report_logs.append(f"⚠️ [技術面] 正乖離過大 ({bias_60:.2f}%)，過熱警告")
        else:
            if price > ma60:
                tech_score += 0.5
                report_logs.append("✅ [技術面] 股價位於季線上方 (多頭)")
            else:
                report_logs.append("🔻 [技術面] 股價位於季線下方 (整理)")

        # RSI
        if rsi < 30:
            tech_score += 1.0
            report_logs.append(f"✅ [技術面] RSI ({rsi:.1f}) 超賣區 (底部訊號)")
        elif rsi > 70:
            report_logs.append(f"⚠️ [技術面] RSI ({rsi:.1f}) 超買區 (追高風險)")
            
        score += tech_score

        # --- 3. 籌碼面 (30%) ---
        chip_score = 0
        formatted_vol = f"{int(chips_vol):,}"
        
        if chips_vol > 5000:
            chip_score += 3.0
            report_logs.append(f"🔥 [籌碼面] 主力大舉買超 (+{formatted_vol} 張)")
        elif chips_vol > 0:
            chip_score += 1.5
            report_logs.append(f"✅ [籌碼面] 法人小幅吸籌 (+{formatted_vol} 張)")
        else:
            report_logs.append(f"🔻 [籌碼面] 法人賣超 ({formatted_vol} 張)")
            
        # 壓低吃貨偵測
        prev_close = df.iloc[-2]['Close']
        if price < prev_close and chips_vol > 0:
             report_logs.append("✨ [籌碼面] 偵測到「壓低吃貨」行為 (價跌量增+法人買)")
             chip_score += 0.5 # 加分
             
        score += chip_score
        
        return score, report_logs, data

# --- UI 介面 ---
st.title("🛡️ Stock Guardian AI v3.0 (Analyst Edition)")
st.markdown("### 全方位即時股票分析系統 (yfinance + 手動校正)")

# 側邊欄：輸入區
with st.sidebar:
    st.header("1. 股票設定")
    ticker_input = st.text_input("輸入台股代號", value="2408")
    
    st.header("2. 手動注入 (Manual Injection)")
    st.info("解決 API 財報滯後問題，請手動輸入最新狀況")
    
    eps_opt = st.selectbox("EPS 狀態 (最新一季)", 
                           ['Turnaround (轉虧為盈)', 'Growth (成長)', 'Decline (衰退)'])
    
    gm_opt = st.radio("毛利率趨勢", ['Up (上升)', 'Down (下降)'])
    
    chips_input = st.number_input("今日法人買賣超 (張)", value=0, step=100, help="正數為買超，負數為賣超")
    
    run_btn = st.button("🚀 啟動高階分析", type="primary")

# 主畫面邏輯
if run_btn:
    bot = StockAnalystAI(ticker_input)
    
    with st.spinner(f"正在連線全球節點抓取 {ticker_input} 最新數據..."):
        raw_df = bot.fetch_data()
    
    if raw_df is not None:
        # 計算指標
        df_processed = bot.calculate_technicals(raw_df)
        
        # 執行分析
        final_score, logs, latest_data = bot.run_analysis(
            df_processed, eps_opt, gm_opt, chips_input
        )
        
        # --- 顯示結果區域 ---
        
        # 1. 關鍵指標卡 (Metrics)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最新收盤價", f"{latest_data['Close']:.1f}")
        col2.metric("季線 (MA60)", f"{latest_data['MA60']:.1f}")
        col3.metric("乖離率 (Bias)", f"{latest_data['Bias_60']:.2f}%", 
                    delta_color="inverse") # 乖離率越小越好(綠色)
        col4.metric("RSI 強弱", f"{latest_data['RSI']:.1f}")
        
        # 2. 評分結果
        st.divider()
        st.subheader("🏆 最終評分與建議")
        
        score_col, advice_col = st.columns([1, 2])
        
        with score_col:
            st.metric("綜合得分", f"{final_score:.1f} / 10.0")
        
        with advice_col:
            if final_score >= 7.5:
                st.success("### ⭐ STRONG BUY (強力買進)\n基本面好轉 + 技術面配合 + 籌碼進駐")
            elif final_score >= 5.0:
                st.warning("### ⚖️ HOLD / ACCUMULATE (分批承接)\n關注轉機，適合區間操作")
            else:
                st.error("### 🛑 SELL / WAIT (觀望/賣出)\n數據疲弱，建議避開")

        # 3. 詳細分析日誌
        with st.expander("查看詳細分析邏輯 (Logic Logs)", expanded=True):
            for log in logs:
                st.write(log)

        # 4. 互動圖表 (證明 MA60 是對的)
        st.divider()
        st.subheader("📈 趨勢驗證圖表")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_processed.index, y=df_processed['Close'], 
                                 mode='lines', name='收盤價'))
        fig.add_trace(go.Scatter(x=df_processed.index, y=df_processed['MA60'], 
                                 mode='lines', name='季線 (60MA)', line=dict(color='orange')))
        
        fig.update_layout(title=f"{ticker_input} 股價 vs 季線", xaxis_title="日期", yaxis_title="價格")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("無法取得數據，請稍後再試。")
