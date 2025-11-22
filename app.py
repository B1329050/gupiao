import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------
# 1. 系統設定
# ---------------------------------------------------------
st.set_page_config(page_title="Stock Guardian Pro", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .status-danger { 
        color: #D32F2F; font-weight: bold; font-size: 1.3rem; 
        background-color: #FFEBEE; padding: 15px; border-radius: 8px; 
        border-left: 6px solid #D32F2F; margin-bottom: 10px;
    }
    .status-safe { 
        color: #2E7D32; font-weight: bold; font-size: 1.3rem; 
        background-color: #E8F5E9; padding: 15px; border-radius: 8px; 
        border-left: 6px solid #2E7D32; margin-bottom: 10px;
    }
    .status-neutral { 
        color: #EF6C00; font-weight: bold; font-size: 1.3rem; 
        background-color: #FFF3E0; padding: 15px; border-radius: 8px; 
        border-left: 6px solid #EF6C00; margin-bottom: 10px;
    }
    .explanation-text { 
        font-size: 1rem; color: #444; margin-left: 5px; line-height: 1.5;
    }
    /* 指標卡片樣式 */
    .metric-card {
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 資料獲取
# ---------------------------------------------------------
@st.cache_data(ttl=900)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        if df.empty: return None, None
        info = stock.info

        # 技術指標
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=9, smooth_window=3)
        df['K'] = stoch.stoch()
        
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        return df, info
    except:
        return None, None

# ---------------------------------------------------------
# 3. 產業判斷
# ---------------------------------------------------------
def detect_industry_type(info):
    sector = info.get('sector', '')
    industry = info.get('industry', '')
    summary = info.get('longBusinessSummary', '')
    
    cycle_keywords = [
        'Semiconductors', 'Memory', 'DRAM', 'Flash',
        'Marine', 'Shipping', 'Freight', 'Transport',
        'Steel', 'Iron', 'Metal',
        'Chemical', 'Oil', 'Petroleum',
        'Panel', 'Display', 'LCD'
    ]
    
    primary_check = (str(sector) + " " + str(industry)).lower()
    for kw in cycle_keywords:
        if kw.lower() in primary_check: return kw
    
    summary_check = str(summary).lower()
    for kw in cycle_keywords:
        if kw.lower() in summary_check: return kw
            
    return None

# ---------------------------------------------------------
# 4. AI 分析邏輯
# ---------------------------------------------------------
def analyze_logic(df, info, buy_price, stop_loss_pct, strategy_mode, use_trailing):
    current_close = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    k_val = df['K'].iloc[-1]
    pb_ratio = info.get('priceToBook', None)
    
    report = {
        "score": 50,
        "action": "觀望 / 持有",
        "details": [],
        "atr_stop_price": current_close - (2.0 * atr),
        "trailing_stop_price": 0.0
    }

    if strategy_mode == "Trend":
        if current_close < ma20:
            report['score'] += 20
            report['details'].append(("[警告] 跌破月線 (MA20)", "過去一個月買的人都賠錢了，短期支撐破裂。"))
        if current_close < ma60:
            report['score'] += 30
            report['details'].append(("[危險] 跌破季線 (MA60)", "過去三個月的趨勢轉壞，中期保護傘失效。"))
        if current_close < report['atr_stop_price']:
            report['score'] += 40
            report['details'].append(("[賣出訊號] 跌破 ATR 安全線", "股價跌破主力防守價位，這是最客觀的離場訊號。"))
        if rsi > 80:
            report['score'] += 10
            report['details'].append(("[風險] RSI 過熱 (>80)", "漲太兇了，隨時有人想獲利了結。"))

    elif strategy_mode == "Cycle":
        report['action'] = "觀察循環位階"
        if pb_ratio:
            if pb_ratio < 1.0:
                report['score'] = 10
                report['action'] = "建議分批佈局 (價值區)"
                report['details'].append(("[機會] P/B < 1.0 (便宜)", "股價比清算價值還便宜，通常是歷史底部。"))
            elif pb_ratio < 1.5:
                report['score'] = 40
                report['action'] = "續抱 / 觀望"
                report['details'].append(("[中性] P/B 正常", "價格合理，不貴也不便宜。"))
            else:
                report['score'] = 70
                report['details'].append(("[注意] P/B 過高 (貴)", "雖然是循環股，但現在價格偏貴。"))

        if k_val < 20:
            report['score'] -= 10
            report['details'].append(("[訊號] KD低檔鈍化", "股價殺過頭了 (超賣)，隨時可能反彈。"))

    user_stop_price = buy_price * (1 - stop_loss_pct / 100)
    if current_close <= user_stop_price:
        report['score'] = 100
        report['details'].append(("[強制停損] 觸及虧損極限", f"虧損已達設定的 {stop_loss_pct}%，請執行紀律。"))

    if use_trailing:
        recent_high = df['High'].tail(60).max()
        if buy_price > recent_high: recent_high = buy_price
        report['trailing_stop_price'] = recent_high * 0.90
        if current_close < report['trailing_stop_price']:
            report['score'] = 100
            report['details'].append(("[停利訊號] 觸發移動停利", "股價從最高點回檔超過 10%，請鎖住獲利。"))

    report['score'] = min(100, max(0, report['score']))
    return report

# ---------------------------------------------------------
# 5. 介面顯示
# ---------------------------------------------------------
def main():
    st.title("🛡️ 股票決策輔助系統 (新手友善版)")
    
    with st.expander("🔰 給新手：這個介面要怎麼看？ (點擊展開)", expanded=False):
        st.markdown("""
        * **P/B (股價淨值比)**：判斷**貴不貴**。 < 1 代表便宜 (適合買)。
        * **ATR (波動安全線)**：判斷**該不該跑**。 跌破這條線，代表主力在出貨。
        * **RSI (相對強弱)**：判斷**有沒有過熱**。 > 80 代表大家都在搶買，容易買在最高點。
        """)
    st.divider()

    # 側邊欄
    st.sidebar.header("第一步：輸入資料")
    ticker_input = st.sidebar.text_input("股票代號", "2408", help="例如 2330")
    ticker = f"{ticker_input}.TW" if not ticker_input.endswith(".TW") else ticker_input
    buy_price = st.sidebar.number_input("買入成本 (元)", value=60.0, help="買進單價")
    shares_held = st.sidebar.number_input("持有股數 (股)", value=1000, step=1000, help="一張 = 1000 股")
    stop_loss_pct = st.sidebar.number_input("最大容忍虧損 (%)", value=10, help="認賠殺出的比例")
    
    df, info = get_stock_data(ticker)
    if df is None:
        st.error("查無資料，請檢查代號。")
        return

    detected = detect_industry_type(info)
    st.sidebar.markdown("---")
    st.sidebar.header("第二步：確認模式")
    
    mode_index = 1 if detected else 0
    if detected:
        st.sidebar.success(f"🔍 偵測到：**{detected}**\n\n這是「景氣循環股」，已切換為**「循環抄底模式」**。")
    else:
        st.sidebar.info("🔍 偵測到：**一般趨勢股**\n\n已使用**「趨勢風控模式」**。")

    strategy_mode = st.sidebar.radio("目前模式", ("Trend (趨勢)", "Cycle (循環)"), index=mode_index, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    use_trailing = st.sidebar.checkbox("🚀 啟用「移動停利」", value=False, help="獲利時強烈建議開啟，回檔 10% 自動賣出。")

    report = analyze_logic(df, info, buy_price, stop_loss_pct, strategy_mode.split()[0], use_trailing)
    
    current_price = df['Close'].iloc[-1]
    pl_amount = (current_price - buy_price) * shares_held
    pl_pct = (pl_amount / (buy_price * shares_held)) * 100 if buy_price > 0 else 0
    
    # --- 1. 主要損益看板 ---
    col1, col2, col3 = st.columns(3)
    col1.metric("當前股價", f"{current_price:.2f}")
    col2.metric("您的總損益", f"{int(pl_amount):,} 元", f"{pl_pct:.2f}%")
    col3.metric("風險評分", f"{report['score']} / 100", help="> 80 分建議賣出")

    st.markdown("---")

    # --- 2. 關鍵指標儀表板 (新增功能) ---
    # 這裡直接顯示您要求的三個指標
    rsi_val = df['RSI'].iloc[-1]
    pb_val = info.get('priceToBook', 0)
    atr_stop = report['atr_stop_price']
    
    st.subheader("📊 關鍵指標體檢")
    m1, m2, m3 = st.columns(3)
    
    m1.metric("RSI 強弱指標", f"{rsi_val:.1f}", help="> 80 過熱 (危險)，< 20 超賣 (機會)")
    
    pb_display = f"{pb_val:.2f}" if isinstance(pb_val, float) else "N/A"
    m2.metric("P/B 股價淨值比", pb_display, help="< 1 便宜 (適合循環股)，> 4 昂貴")
    
    m3.metric("ATR 安全防線", f"{atr_stop:.2f}", help="如果收盤價低於這個數字，代表跌破安全線，建議賣出。")
    
    st.markdown("---")

    # --- 3. AI 分析報告 ---
    st.subheader("📋 AI 分析報告")
    if report['score'] >= 80:
        st.markdown(f"<div class='status-danger'>🛑 危險訊號 (賣出/減碼)</div>", unsafe_allow_html=True)
        st.write("目前情況危險，建議不要再抱了，請考慮離場。")
    elif report['score'] <= 30:
        st.markdown(f"<div class='status-safe'>✅ 安全訊號 ({report['action']})</div>", unsafe_allow_html=True)
        st.write("目前股價處於安全或低估區間，可以安心。")
    else:
        st.markdown(f"<div class='status-neutral'>⚠️ 中性觀察</div>", unsafe_allow_html=True)
        st.write("目前方向不明確，建議多看少做。")

    st.write("")
    
    if not report['details']:
        st.info("目前走勢正常，無特殊訊號。")
    
    for title, explanation in report['details']:
        with st.container():
            st.markdown(f"**{title}**")
            st.markdown(f"<div class='explanation-text'>💡 白話解釋：{explanation}</div>", unsafe_allow_html=True)
            st.divider()

    if pl_amount < 0:
        deposit_rate = 0.017
        total_cost = buy_price * shares_held
        loss_years = abs(pl_amount) / (total_cost * deposit_rate) if total_cost > 0 else 0
        st.error(f"💸 **現實換算**：這筆虧損金額，相當於賠掉了本金存銀行 **{loss_years:.1f} 年** 的利息。")

    # --- 4. 大圖表 ---
    st.markdown("### 📈 走勢圖 (清晰版)")
    
    fig = go.Figure()
    
    # K線
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name='股價', increasing_line_color='#EF5350', decreasing_line_color='#26A69A'
    ))
    
    # 均線
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA20'], line=dict(color='#FFA726', width=2), name='月線 (短期支撐)'
    ))
    
    # ATR 停損線
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close']-(2*df['ATR']), 
        line=dict(color='red', width=2, dash='dot'), name=f'AI 安全底線 ({atr_stop:.2f})'
    ))
    
    # 成本線
    fig.add_hline(y=buy_price, line_dash="dash", line_color="blue", line_width=2, annotation_text="您的成本")
    
    # 移動停利線
    if use_trailing and report['trailing_stop_price'] > 0:
            fig.add_hline(y=report['trailing_stop_price'], line_color="purple", line_width=3, annotation_text="移動停利線")
    
    fig.update_layout(
        xaxis_rangeslider_visible=False, height=600,
        margin=dict(t=30, b=20), font=dict(size=14),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
