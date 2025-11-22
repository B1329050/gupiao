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
st.set_page_config(page_title="Stock Analysis System", layout="wide")

# CSS 設定：建立清晰的紅綠文字風格，不使用圖示
st.markdown("""
    <style>
    .status-danger { color: #D32F2F; font-weight: bold; }
    .status-safe { color: #388E3C; font-weight: bold; }
    .status-neutral { color: #F57C00; font-weight: bold; }
    .metric-value { font-size: 1.2rem; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 資料獲取與計算核心
# ---------------------------------------------------------
@st.cache_data(ttl=900) # 15分鐘更新一次
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 抓取足夠的歷史資料以計算長期均線
        df = stock.history(period="2y")
        
        if df.empty:
            return None, None

        # 基本面資訊
        info = stock.info

        # --- 技術指標計算 ---
        # 移動平均線 (MA)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['MA120'] = df['Close'].rolling(window=120).mean()
        
        # KD 指標 (Stochastic Oscillator)
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=9, smooth_window=3)
        df['K'] = stoch.stoch()
        df['D'] = stoch.stoch_signal()
        
        # RSI 相對強弱指標
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # ATR 平均真實波幅 (用於計算波動風險)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        return df, info
    except Exception as e:
        st.error(f"System Error: {e}")
        return None, None

# ---------------------------------------------------------
# 產業判斷邏輯
# ---------------------------------------------------------
def detect_industry_type(info):
    """
    判斷股票屬性：景氣循環股 vs 一般趨勢股
    """
    sector = info.get('sector', '')
    industry = info.get('industry', '')
    summary = info.get('longBusinessSummary', '')
    
    # 循環型產業關鍵字清單
    cycle_keywords = [
        'Semiconductors', 'Memory', 'DRAM', 'Flash', # 記憶體
        'Steel', 'Iron', 'Metal', # 鋼鐵
        'Marine', 'Shipping', 'Transport', # 航運
        'Chemical', 'Oil', 'Petroleum', # 塑化
        'Panel', 'Display', 'LCD' # 面板
    ]
    
    detected_keyword = None
    
    # 搜尋配對
    text_to_search = (str(sector) + " " + str(industry) + " " + str(summary)).lower()
    
    for kw in cycle_keywords:
        if kw.lower() in text_to_search:
            detected_keyword = kw
            break
            
    return detected_keyword

# ---------------------------------------------------------
# 核心分析邏輯
# ---------------------------------------------------------
def analyze_logic(df, info, buy_price, stop_loss_pct, strategy_mode, use_trailing):
    # 取得最新一筆數據
    current_close = df['Close'].iloc[-1]
    
    # 技術指標數值
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    k_val = df['K'].iloc[-1]
    
    # 基本面數值
    pb_ratio = info.get('priceToBook', None)
    
    report = {
        "score": 50, # 0(安全) ~ 100(危險)
        "action": "觀望 / 持有",
        "details": [],
        "atr_stop_price": 0.0,
        "trailing_stop_price": 0.0,
        "is_danger": False
    }

    # 計算 ATR 動態停損價 (收盤價 - 2倍 ATR)
    report['atr_stop_price'] = current_close - (2.0 * atr)

    # -------------------------------------------------------
    # 策略 A: 趨勢風控模式 (Trend Mode)
    # -------------------------------------------------------
    if strategy_mode == "Trend":
        # 1. 均線檢測
        if current_close < ma20:
            report['details'].append(f"[警告] 收盤價({current_close:.2f}) 跌破月線 MA20({ma20:.2f})，短線轉弱。")
            report['score'] += 20
        
        if current_close < ma60:
            report['details'].append(f"[危險] 收盤價({current_close:.2f}) 跌破季線 MA60({ma60:.2f})，中線趨勢轉空。")
            report['score'] += 30

        # 2. ATR 波動停損檢測
        if current_close < report['atr_stop_price']:
            report['details'].append(f"[賣出訊號] 跌破 ATR 動態支撐位({report['atr_stop_price']:.2f})。")
            report['score'] += 40

        # 3. RSI 高檔過熱
        if rsi > 80:
            report['details'].append(f"[風險] RSI 指標({rsi:.2f}) 進入過熱區(>80)，隨時可能回檔。")
            report['score'] += 10

    # -------------------------------------------------------
    # 策略 B: 循環抄底模式 (Cycle Mode)
    # -------------------------------------------------------
    elif strategy_mode == "Cycle":
        report['action'] = "觀察循環位階"
        
        # 1. 檢測 P/B (股價淨值比)
        if pb_ratio:
            if pb_ratio < 1.0:
                report['score'] = 10
                report['action'] = "建議分批佈局 (價值區)"
                report['details'].append(f"[機會] 股價淨值比 P/B({pb_ratio:.2f}) 小於 1.0，屬於歷史低估區間。")
            elif pb_ratio < 1.5:
                report['score'] = 40
                report['action'] = "續抱 / 觀望"
                report['details'].append(f"[中性] 股價淨值比 P/B({pb_ratio:.2f}) 處於合理區間。")
            else:
                report['score'] = 70
                report['details'].append(f"[注意] 股價淨值比 P/B({pb_ratio:.2f}) 已高，風險增加。")
        else:
            report['details'].append("[錯誤] 無法取得 P/B 數據，無法執行循環策略判斷。")

        # 2. KD 低檔鈍化檢查
        if k_val < 20:
            report['details'].append(f"[訊號] KD指標 K值({k_val:.2f}) 小於 20，處於超賣區，不建議此時殺低。")
            report['score'] -= 10

    # -------------------------------------------------------
    # 通用模組：停損與停利
    # -------------------------------------------------------
    
    # 硬性停損 (使用者設定)
    user_stop_price = buy_price * (1 - stop_loss_pct / 100)
    if current_close <= user_stop_price:
        report['details'].append(f"[強制停損] 觸及您設定的虧損極限 (-{stop_loss_pct}%)，價格低於 {user_stop_price:.2f}。")
        report['score'] = 100
        report['is_danger'] = True

    # 移動停利 (Trailing Stop)
    if use_trailing:
        recent_high = df['High'].tail(60).max()
        if buy_price > recent_high:
            recent_high = buy_price
            
        trailing_stop_price = recent_high * 0.90
        report['trailing_stop_price'] = trailing_stop_price
        
        if current_close < trailing_stop_price:
            report['details'].append(f"[停利訊號] 股價已從波段高點({recent_high:.2f}) 回檔超過 10%，建議獲利了結。")
            report['score'] = 100
            report['is_danger'] = True
        else:
            report['details'].append(f"[監控中] 移動停利點為 {trailing_stop_price:.2f} (高點 {recent_high:.2f} 之 90%)。")

    report['score'] = min(100, max(0, report['score']))
    return report

# ---------------------------------------------------------
# UI 介面
# ---------------------------------------------------------
def main():
    st.title("投資決策輔助系統 (Stock Decision Support)")
    st.markdown("本系統僅提供數據分析運算結果，不含任何主觀預測。請依據下方數據進行決策。")
    st.divider()

    # --- 側邊欄輸入區 ---
    st.sidebar.header("參數設定 (Parameters)")
    
    # 1. 買什麼股票
    ticker_input = st.sidebar.text_input("股票代號 (例如: 2408)", "2408")
    ticker = f"{ticker_input}.TW" if not ticker_input.endswith(".TW") else ticker_input
    
    # 2. 買入多少錢
    buy_price = st.sidebar.number_input("買入成本 (每股單價)", value=60.0, step=0.1)
    
    # 3. 買了多少股 (新增功能)
    shares_held = st.sidebar.number_input("持有股數 (例如: 5張請輸入 5000)", value=1000, step=1000)
    
    # 4. 停損設定
    stop_loss_pct = st.sidebar.number_input("最大容忍虧損 (%)", value=10, min_value=1, max_value=50)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("策略選擇")
    
    # 先獲取資料
    df, info = get_stock_data(ticker)
    
    if df is None:
        st.error(f"錯誤：無法獲取代號 {ticker} 之數據，請檢查輸入是否正確。")
        return

    # 自動產業偵測
    detected_industry = detect_industry_type(info)
    default_mode_index = 0
    industry_msg = "未偵測到特定循環產業特徵，建議使用標準趨勢模式。"
    
    if detected_industry:
        default_mode_index = 1 # 切換到循環模式
        industry_msg = f"系統偵測此為 **{detected_industry}** 相關產業，屬於景氣循環股。"
    
    st.sidebar.info(industry_msg)
    
    strategy_mode = st.sidebar.radio(
        "分析模式",
        ("Trend (趨勢操作)", "Cycle (循環/價值操作)"),
        index=default_mode_index
    )
    
    use_trailing = st.sidebar.checkbox("啟用移動停利 (Trailing Stop)", value=False)
    
    # --- 執行分析 ---
    report = analyze_logic(df, info, buy_price, stop_loss_pct, strategy_mode.split()[0], use_trailing)
    
    # --- 顯示主要數據看板 ---
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100
    
    # 計算精確損益
    total_cost = buy_price * shares_held
    current_value = current_price * shares_held
    pl_amount = current_value - total_cost
    pl_pct = (pl_amount / total_cost) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("當前股價", f"{current_price:.2f}", f"{change:.2f} ({change_pct:.2f}%)")
    col2.metric("總報酬率 (%)", f"{pl_pct:.2f}%", delta_color="normal")
    col3.metric("總損益金額 (TWD)", f"{int(pl_amount):,}", delta_color="normal") # 加千分位逗號
    col4.metric("風險評分 (0-100)", f"{report['score']}")

    st.divider()

    # --- 詳細分析報告 ---
    st.subheader("系統分析報告 (System Report)")
    
    if report['score'] >= 80:
        st.markdown(f"<div class='status-danger'>【建議動作：賣出 / 避險】風險分數 {report['score']}</div>", unsafe_allow_html=True)
    elif report['score'] <= 30:
        st.markdown(f"<div class='status-safe'>【建議動作：{report['action']}】風險分數 {report['score']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='status-neutral'>【建議動作：中性觀察】風險分數 {report['score']}</div>", unsafe_allow_html=True)

    st.write("判斷依據 (Evidence)：")
    for detail in report['details']:
        st.text(f"• {detail}")

    # 定存比較 (修正為總金額計算)
    if pl_amount < 0:
        deposit_rate = 0.017
        deposit_loss_years = abs(pl_amount) / (total_cost * deposit_rate)
        st.markdown(f"**機會成本換算**：目前的虧損金額 ({int(abs(pl_amount)):,} 元)，等同於損失了該筆本金 **{deposit_loss_years:.1f} 年** 的定存利息。")

    st.divider()

    # --- 互動圖表 ---
    tab1, tab2 = st.tabs(["K線與均線圖", "基本面數據"])
    
    with tab1:
        fig = go.Figure()
        
        # K線
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'], name='Price'))
        
        # 均線
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name='MA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], line=dict(color='green', width=1), name='MA60'))
        
        # ATR 停損線
        atr_line = df['Close'] - (2.0 * df['ATR'])
        fig.add_trace(go.Scatter(x=df.index, y=atr_line, 
                         line=dict(color='red', width=1, dash='dot'), 
                         name='ATR Support'))
        
        # 使用者買入價
        fig.add_hline(y=buy_price, line_dash="dash", line_color="blue", annotation_text="Cost")

        if use_trailing and report['trailing_stop_price'] > 0:
             fig.add_hline(y=report['trailing_stop_price'], line_color="purple", annotation_text="Trailing Stop")

        fig.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"數據驗證區：MA20={df['MA20'].iloc[-1]:.2f} | MA60={df['MA60'].iloc[-1]:.2f} | RSI={df['RSI'].iloc[-1]:.2f} | ATR={df['ATR'].iloc[-1]:.2f}")

    with tab2:
        col_b1, col_b2, col_b3 = st.columns(3)
        
        pb = info.get('priceToBook', 'N/A')
        pe = info.get('trailingPE', 'N/A')
        roe = info.get('returnOnEquity', 'N/A')
        
        col_b1.metric("股價淨值比 (P/B)", f"{pb}" if isinstance(pb, str) else f"{pb:.2f}")
        col_b2.metric("本益比 (P/E)", f"{pe}" if isinstance(pe, str) else f"{pe:.2f}")
        col_b3.metric("股東權益報酬率 (ROE)", f"{roe*100:.2f}%" if isinstance(roe, float) else "N/A")
        
        st.write("公司簡介 (Business Summary):")
        st.text(info.get('longBusinessSummary', '無資料'))

if __name__ == "__main__":
    main()
