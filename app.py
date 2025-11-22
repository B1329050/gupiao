import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------
# 系統設定
# ---------------------------------------------------------
st.set_page_config(page_title="Stock Guardian", layout="wide")

# CSS 設定
st.markdown("""
    <style>
    .status-danger { 
        color: #D32F2F; font-weight: bold; font-size: 1.2rem; 
        background-color: #FFEBEE; padding: 10px; border-radius: 5px; border-left: 5px solid #D32F2F;
    }
    .status-safe { 
        color: #2E7D32; font-weight: bold; font-size: 1.2rem; 
        background-color: #E8F5E9; padding: 10px; border-radius: 5px; border-left: 5px solid #2E7D32;
    }
    .status-neutral { 
        color: #EF6C00; font-weight: bold; font-size: 1.2rem; 
        background-color: #FFF3E0; padding: 10px; border-radius: 5px; border-left: 5px solid #EF6C00;
    }
    .explanation-text { font-size: 0.95rem; color: #555; margin-left: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 資料獲取
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
# 產業判斷
# ---------------------------------------------------------
def detect_industry_type(info):
    sector = info.get('sector', '')
    industry = info.get('industry', '')
    summary = info.get('longBusinessSummary', '')
    cycle_keywords = ['Semiconductors', 'Memory', 'DRAM', 'Flash', 'Marine', 'Shipping', 'Freight', 'Steel', 'Iron', 'Panel', 'LCD']
    
    primary_check = (str(sector) + " " + str(industry)).lower()
    for kw in cycle_keywords:
        if kw.lower() in primary_check: return kw
    
    summary_check = str(summary).lower()
    for kw in cycle_keywords:
        if kw.lower() in summary_check: return kw
            
    return None

# ---------------------------------------------------------
# 核心分析邏輯
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
            report['details'].append(("[警告] 跌破月線 (MA20)", "這代表過去一個月買的人都賠錢了，短期支撐破裂，股價容易繼續跌。"))
        if current_close < ma60:
            report['score'] += 30
            report['details'].append(("[危險] 跌破季線 (MA60)", "代表過去三個月的趨勢已經轉壞，中期保護傘失效。"))
        if current_close < report['atr_stop_price']:
            report['score'] += 40
            report['details'].append(("[賣出訊號] 跌破 ATR 安全線", "股價波動超出正常範圍，代表主力正在大量出貨，這是最客觀的離場訊號。"))
        if rsi > 80:
            report['score'] += 10
            report['details'].append(("[風險] RSI 過熱 (>80)", "代表這幾天漲太兇了，隨時會有人想獲利了結，不要追高。"))

    elif strategy_mode == "Cycle":
        report['action'] = "觀察循環位階"
        if pb_ratio:
            if pb_ratio < 1.0:
                report['score'] = 10
                report['action'] = "建議分批佈局 (價值區)"
                report['details'].append(("[機會] 股價淨值比 P/B < 1.0", "股價已經比公司的清算價值還便宜，這在景氣循環股中通常是歷史底部。"))
            elif pb_ratio < 1.5:
                report['score'] = 40
                report['action'] = "續抱 / 觀望"
                report['details'].append(("[中性] 股價淨值比 P/B 正常", "股價處於合理範圍，不貴也不便宜，可以耐心等待。"))
            else:
                report['score'] = 70
                report['details'].append(("[注意] 股價淨值比 P/B 過高", "雖然是循環股，但現在價格偏貴，風險正在增加。"))
        if k_val < 20:
            report['score'] -= 10
            report['details'].append(("[訊號] KD指標低檔鈍化", "股價已經殺過頭了 (超賣)，隨時可能出現跌深反彈，現在賣容易賣在最低點。"))

    user_stop_price = buy_price * (1 - stop_loss_pct / 100)
    if current_close <= user_stop_price:
        report['score'] = 100
        report['details'].append(("[強制停損] 觸及虧損極限", f"虧損已達您設定的 {stop_loss_pct}%。這是最後一道防線，請務必執行紀律，保留本金。"))

    if use_trailing:
        recent_high = df['High'].tail(60).max()
        if buy_price > recent_high: recent_high = buy_price
        report['trailing_stop_price'] = recent_high * 0.90
        if current_close < report['trailing_stop_price']:
            report['score'] = 100
            report['details'].append(("[停利訊號] 觸發移動停利", "股價從最高點回檔超過 10%，代表這波漲勢結束了，請先把賺到的錢放口袋。"))

    report['score'] = min(100, max(0, report['score']))
    return report

# ---------------------------------------------------------
# UI 介面
# ---------------------------------------------------------
def main():
    st.title("🛡️ 股票決策輔助系統 (新手友善版)")
    
    with st.expander("🔰 給新手：這個介面要怎麼看？ (點擊展開說明)", expanded=False):
        st.markdown("""
        ### 1. 為什麼要用這個軟體？
        這套系統是您的「風險煞車」。它不會預測明天漲跌，但會在**危險發生時**亮紅燈提醒您，防止大賠。

        ### 2. 關鍵數值說明書 (Dictionary)
        * **P/B (股價淨值比)**：判斷東西**貴不貴**。 < 1 代表便宜 (適合買)。
        * **ATR (波動安全線)**：判斷**該不該跑**。 跌破這條線，代表主力在出貨。
        * **RSI (相對強弱)**：判斷**有沒有過熱**。 > 80 代表大家都在搶買，容易買在最高點。
        """)
    st.divider()

    st.sidebar.header("第一步：輸入資料")
    ticker_input = st.sidebar.text_input("股票代號", "2408", help="請輸入台股代號，例如 2330")
    ticker = f"{ticker_input}.TW" if not ticker_input.endswith(".TW") else ticker_input
    buy_price = st.sidebar.number_input("買入成本 (元)", value=60.0, help="您當時買進一股是多少錢？")
    shares_held = st.sidebar.number_input("持有股數 (股)", value=1000, step=1000, help="一張股票是 1000 股。")
    stop_loss_pct = st.sidebar.number_input("最大容忍虧損 (%)", value=10, help="如果賠超過這個比例，您願意認賠殺出嗎？")
    
    df, info = get_stock_data(ticker)
    if df is None:
        st.error("查無資料，請檢查代號。")
        return

    detected = detect_industry_type(info)
    st.sidebar.markdown("---")
    st.sidebar.header("第二步：確認模式")
    
    mode_index = 1 if detected else 0
    if detected:
        st.sidebar.success(f"🔍 偵測到：**{detected}**\n\n這是「景氣循環股」，系統已切換為**「循環抄底模式」**。")
    else:
        st.sidebar.info("🔍 偵測到：**一般趨勢股**\n\n系統使用**「趨勢風控模式」**。")

    strategy_mode = st.sidebar.radio("目前模式", ("Trend (趨勢)", "Cycle (循環)"), index=mode_index, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    use_trailing = st.sidebar.checkbox("🚀 啟用「移動停利」", value=False, help="【強烈建議獲利時開啟】\n當股價從最高點回跌 10% 時，系統會強制叫您賣出。")

    report = analyze_logic(df, info, buy_price, stop_loss_pct, strategy_mode.split()[0], use_trailing)
    
    current_price = df['Close'].iloc[-1]
    pl_amount = (current_price - buy_price) * shares_held
    pl_pct = (pl_amount / (buy_price * shares_held)) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("當前股價", f"{current_price:.2f}")
    col2.metric("您的總損益", f"{int(pl_amount):,} 元", f"{pl_pct:.2f}%")
    col3.metric("風險評分", f"{report['score']} / 100", help="分數越高越危險。超過 80 分建議賣出。")

    st.subheader("📋 AI 分析報告")
    if report['score'] >= 80:
        st.markdown(f"<div class='status-danger'>🛑 危險訊號 (賣出/減碼)</div>", unsafe_allow_html=True)
        st.write("目前情況非常危險，建議不要再抱了。")
    elif report['score'] <= 30:
        st.markdown(f"<div class='status-safe'>✅ 安全訊號 ({report['action']})</div>", unsafe_allow_html=True)
        st.write("目前股價處於安全或低估區間，可以安心。")
    else:
        st.markdown(f"<div class='status-neutral'>⚠️ 中性觀察</div>", unsafe_allow_html=True)
        st.write("目前方向不明確，建議多看少做。")

    st.write("")
    st.markdown("#### 🧐 為什麼這樣判斷？ (白話翻譯)")
    
    if not report['details']:
        st.info("目前沒有出現特殊的買賣訊號，股價走勢正常。")
    
    for title, explanation in report['details']:
        with st.container():
            st.markdown(f"**{title}**")
            st.markdown(f"<div class='explanation-text'>💡 白話解釋：{explanation}</div>", unsafe_allow_html=True)
            st.divider()

    if pl_amount < 0:
        deposit_rate = 0.017
        loss_years = abs(pl_amount) / (buy_price * shares_held * deposit_rate)
        st.error(f"💸 **現實換算**：這筆虧損金額，相當於賠掉了本金存銀行 **{loss_years:.1f} 年** 的利息。")

    # --- 重點修改：大圖清晰版 ---
    st.markdown("### 📊 走勢圖")
    tab1, tab2 = st.tabs(["K線圖 (含停損線)", "基本面數據 (P/B)"])
    
    with tab1:
        fig = go.Figure()
        
        # 1. K線圖 (設定台股顏色：紅漲綠跌)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
            name='股價',
            increasing_line_color='#EF5350',  # 紅色 (漲)
            decreasing_line_color='#26A69A'   # 綠色 (跌)
        ))
        
        # 2. 均線 (加粗 width=3)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA20'], 
            line=dict(color='#FFA726', width=3), # 橘色加粗
            name='月線 (短期支撐)'
        ))
        
        # 3. ATR 停損線 (紅色虛線，加粗)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close']-(2*df['ATR']), 
            line=dict(color='red', width=3, dash='dot'), # 紅色虛線加粗
            name='AI 安全底線'
        ))
        
        # 4. 成本線 (藍色虛線)
        fig.add_hline(y=buy_price, line_dash="dash", line_color="blue", line_width=2, annotation_text="您的成本")
        
        # 5. 移動停利線 (紫色實線，加粗)
        if use_trailing and report['trailing_stop_price'] > 0:
             fig.add_hline(y=report['trailing_stop_price'], line_color="purple", line_width=3, annotation_text="移動停利線")
        
        # 6. 版面設定 (加大高度, 字體變大)
        fig.update_layout(
            xaxis_rangeslider_visible=False, 
            height=650, # 加大高度
            margin=dict(t=30,b=20),
            font=dict(size=16), # 字體加大
            template="plotly_white", # 背景乾淨白色
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center") # 圖例放在正上方
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        pb = info.get('priceToBook', 'N/A')
        pe = info.get('trailingPE', 'N/A')
        col_a, col_b = st.columns(2)
        col_a.metric("股價淨值比 (P/B)", f"{pb:.2f}" if isinstance(pb, float) else pb, help="< 1 代表便宜，> 4 代表貴")
        col_b.metric("本益比 (P/E)", f"{pe:.2f}" if isinstance(pe, float) else pe, help="回本需要的年數。")
        st.caption("P/B 解釋：如果這家公司今天倒閉清算，股東能拿回多少錢。數值 0.8 代表妳用 0.8 元買到價值 1 元的東西。")

if __name__ == "__main__":
    main()
