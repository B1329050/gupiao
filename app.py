"""
Stock Guardian Pro (Ver 5.1 - Math Edge & Forward Testing Edition)
Author: Stock Guardian AI
Description: 
    一個專為實戰設計的台股決策儀表板。
    不依賴回測，而是使用數學優勢 (VWAP, Slope, R/R Ratio, Volatility Sizing) 
    來輔助當下的交易決策。
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# ---------------------------------------------------------
# 1. 系統設定與 CSS
# ---------------------------------------------------------
st.set_page_config(page_title="Stock Guardian Pro", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .status-danger { 
        color: #D32F2F; font-weight: bold; font-size: 1.2rem; 
        background-color: #FFEBEE; padding: 15px; border-radius: 8px; 
        border-left: 6px solid #D32F2F; margin-bottom: 10px;
    }
    .status-safe { 
        color: #2E7D32; font-weight: bold; font-size: 1.2rem; 
        background-color: #E8F5E9; padding: 15px; border-radius: 8px; 
        border-left: 6px solid #2E7D32; margin-bottom: 10px;
    }
    .status-neutral { 
        color: #EF6C00; font-weight: bold; font-size: 1.2rem; 
        background-color: #FFF3E0; padding: 15px; border-radius: 8px; 
        border-left: 6px solid #EF6C00; margin-bottom: 10px;
    }
    .explanation-text { font-size: 1rem; color: #444; margin-left: 5px; line-height: 1.5; }
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .tooltip-text {
        color: #0066cc; font-weight: bold; text-decoration: underline dotted; cursor: help;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 資料獲取 (核心數學運算層)
# ---------------------------------------------------------
@st.cache_data(ttl=900) 
def get_stock_data(ticker_input):
    try:
        ticker_clean = str(ticker_input).replace(".TW", "").replace(".TWO", "").strip()
        
        # 1. 抓取資料
        try_ticker = f"{ticker_clean}.TW"
        stock = yf.Ticker(try_ticker)
        df = stock.history(period="5y") 
        
        if df.empty:
            try_ticker = f"{ticker_clean}.TWO"
            stock = yf.Ticker(try_ticker)
            df = stock.history(period="5y")
            
        if df.empty: return None, None, None
        if len(df) < 65: return None, None, None # 資料不足防呆

        try:
            info = stock.info
        except:
            info = {}

        # --- 基礎技術指標 ---
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['Bias'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
        
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=9, smooth_window=3)
        df['K'] = stoch.stoch()
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # --- 數學優勢 1: ATR & 吊燈停損 (防禦演算法) ---
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        # [關鍵修正] 使用 shift(1) 確保是用「昨天以前」的數據來決定今天的停損點 (無 Look-ahead bias)
        df['High_20'] = df['High'].shift(1).rolling(window=20).max() 
        df['Chandelier_Exit'] = df['High_20'] - (2.0 * df['ATR'])
        
        # --- 數學優勢 2: VWAP (法人成本線) ---
        # 計算 20 日滾動 VWAP，代表近期市場的平均持有成本
        v = df['Volume'].values
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        # 使用 numpy 處理避免分母為 0
        df['VWAP'] = (tp * v).rolling(window=20).sum() / v.rolling(window=20).sum().replace(0, np.nan)
        
        # --- 數學優勢 3: Slope (動能斜率) ---
        # 計算過去 10 天收盤價的線性回歸斜率
        # 使用滾動視窗計算斜率 (較慢但精確)
        slope_list = [np.nan] * len(df)
        window_size = 10
        closes = df['Close'].values
        
        # 為了效能，我們只算最後 60 筆 (夠畫圖就好)
        start_idx = max(window_size, len(df) - 60)
        for i in range(start_idx, len(df)):
            y_segment = closes[i-window_size:i]
            x_segment = np.arange(window_size)
            if len(y_segment) == window_size:
                slope, _ = np.polyfit(x_segment, y_segment, 1)
                # 標準化斜率：(斜率 / 股價) * 100 -> 變成百分比斜率
                slope_list[i] = (slope / closes[i]) * 100
            else:
                slope_list[i] = 0
            
        df['Slope_Pct'] = slope_list
        # 填補前面的空值為0，避免報錯
        df['Slope_Pct'] = df['Slope_Pct'].fillna(0)
        
        # 籌碼與位階
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()
        
        lookback = 500
        df['2Y_High'] = df['High'].rolling(window=lookback).max() if len(df) > lookback else df['High'].max()
        df['2Y_Low'] = df['Low'].rolling(window=lookback).min() if len(df) > lookback else df['Low'].min()
        
        denom = df['2Y_High'] - df['2Y_Low']
        df['Price_Pos'] = np.where(denom == 0, 0, (df['Close'] - df['2Y_Low']) / denom)

        return df, info, try_ticker
    except Exception as e:
        # print(f"Error: {e}") # Debug use
        return None, None, None

def calculate_seasonality(df):
    try:
        if len(df) < 250: return None, None
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly.index.month
        df_monthly['Pct_Change'] = df_monthly['Close'].pct_change() * 100
        seasonal_stats = df_monthly.groupby('Month')['Pct_Change'].mean()
        win_rate = (df_monthly[df_monthly['Pct_Change'] > 0].groupby('Month')['Pct_Change'].count() / df_monthly.groupby('Month')['Pct_Change'].count() * 100).fillna(0)
        return seasonal_stats, win_rate
    except:
        return None, None

# 硬編碼的循環股清單 (避免 Info 抓不到)
CYCLE_STOCKS = ["2603", "2609", "2615", "2618", "2408", "2344", "2337", "2002", "1301", "1303", "2409", "3481", "1101"]

def detect_industry_type_optimized(ticker, info):
    clean_ticker = str(ticker).replace(".TW", "").replace(".TWO", "").strip()
    if clean_ticker in CYCLE_STOCKS: return "Cyclical (名單)"
    if not info: return None
    short_name = info.get('shortName', '')
    if 'ETF' in short_name or 'Dividend' in short_name: return 'ETF'
    cycle_keywords = ['semiconductors', 'memory', 'dram', 'marine', 'shipping', 'steel', 'chemical', 'panel']
    check_str = (str(info.get('sector', '')) + " " + str(info.get('longBusinessSummary', ''))).lower()
    for kw in cycle_keywords:
        if kw.lower() in check_str: return kw
    return None

# ---------------------------------------------------------
# 3. AI 分析邏輯 (加入數學濾網)
# ---------------------------------------------------------
def analyze_logic(df, buy_price, stop_loss_pct, strategy_mode, use_trailing):
    current_close = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    bias = df['Bias'].iloc[-1]
    k_val = df['K'].iloc[-1]
    price_pos = df['Price_Pos'].iloc[-1]
    atr_stop_price = df['Chandelier_Exit'].iloc[-1]
    
    # 數學優勢參數
    vwap_val = df['VWAP'].iloc[-1]
    slope_pct = df['Slope_Pct'].iloc[-1]
    
    obv_curr = df['OBV'].iloc[-1]
    obv_ma = df['OBV_MA20'].iloc[-1]
    obv_trend_text = "📈 強勢流入" if obv_curr > obv_ma else "📉 弱勢流出"
    
    report = {
        "score": 50, "action": "觀望 / 持有", "details": [],
        "atr_stop_price": atr_stop_price, "trailing_stop_price": 0.0, 
        "obv_trend": obv_trend_text, "price_pos": price_pos,
        "vwap": vwap_val, "slope": slope_pct, "rr_ratio": 0.0
    }

    # --- 1. 技術面與動能 (數學優勢) ---
    # Slope (斜率) 判斷
    if slope_pct > 0.4:
        report['details'].append(("[動能] 🚀 噴出狀態", f"上漲斜率 {slope_pct:.2f}%，力道極強。"))
    elif slope_pct < -0.2:
        report['score'] += 20
        report['details'].append(("[動能] ⚠️ 下墜狀態", f"下跌斜率 {slope_pct:.2f}%，切勿接刀。"))
    elif strategy_mode == "Trend" and 0 < slope_pct < 0.1:
        report['details'].append(("[預警] 🐢 漲勢鈍化", "雖然在漲但速度變慢 (斜率 < 0.1%)。"))

    # VWAP (成本) 判斷
    if current_close > vwap_val:
        report['score'] -= 5
        report['details'].append(("[籌碼] ✅ 站上法人成本", "股價高於 VWAP，支撐轉強。"))
    else:
        report['score'] += 10
        report['details'].append(("[籌碼] ❌ 跌破法人成本", "股價低於 VWAP，上方有解套賣壓。"))

    if bias > 12:
        report['score'] += 15
        report['details'].append(("[風險] 🔥 乖離率過大", "漲太兇，隨時可能回測。"))

    # --- 2. 策略分支 ---
    if strategy_mode == "Trend":
        if price_pos > 0.8 and rsi > 80:
             report['score'] += 15
             report['details'].append(("[風險] 高檔過熱", "位階高且 RSI 過熱。"))
        
        if current_close < ma20:
            report['score'] += 20
            report['details'].append(("[警告] 跌破月線", "短線轉弱。"))
        if current_close < atr_stop_price:
            report['score'] += 40
            report['details'].append(("[賣出] 🛑 跌破 ATR 防線", "趨勢反轉確認，請離場。"))

    elif strategy_mode == "Cycle":
        if price_pos < 0.2:
            report['score'] = 10
            report['action'] = "佈局 (低基期)"
            report['details'].append(("[機會] 💎 歷史低檔", "位階 < 20%，長線安全。"))
        elif price_pos > 0.8:
            report['score'] = 70
            report['details'].append(("[注意] ⛰️ 歷史高檔", "位階 > 80%，風險較高。"))

    # --- 3. 盈虧比 (Reward/Risk Ratio) 計算 ---
    # 潛在風險 = 現價 - ATR停損
    risk_dist = current_close - atr_stop_price
    if risk_dist <= 0: risk_dist = 0.01 # 避免分母為0或負數 (代表已跌破)
    
    # 潛在獲利 = 前波高點 - 現價 (如果已經創新高，假設獲利空間為 2倍 ATR)
    recent_high = df['High'].tail(60).max()
    if recent_high <= current_close:
        reward_dist = 2.0 * df['ATR'].iloc[-1] # 順勢假設
    else:
        reward_dist = recent_high - current_close
        
    rr_ratio = reward_dist / risk_dist
    report['rr_ratio'] = rr_ratio
    
    if rr_ratio < 1.5 and report['score'] < 50:
        report['details'].append(("[算盤] 📉 盈虧比不佳", f"賺賠比僅 {rr_ratio:.1f}，肉少骨頭多，不建議追。"))

    # --- 4. 停損停利 ---
    if buy_price > 0:
        user_stop_price = buy_price * (1 - stop_loss_pct / 100)
        if current_close <= user_stop_price:
            report['score'] = 100
            report['details'].append(("[停損] 🛑 觸及虧損底線", "紀律執行，保留本金。"))

        if use_trailing and current_close > buy_price:
            recent_high_hold = df['High'].tail(60).max()
            if recent_high_hold < buy_price: recent_high_hold = buy_price 
            report['trailing_stop_price'] = recent_high_hold * 0.90
            
            if current_close < report['trailing_stop_price']:
                report['score'] = 100
                report['details'].append(("[停利] 💰 觸發移動停利", "回檔 10% 獲利了結。"))

    report['score'] = min(100, max(0, report['score']))
    return report

# ---------------------------------------------------------
# 4. 儀表板頁面 (Ver 5.1 訊號速查版)
# ---------------------------------------------------------
def dashboard_page():
    st.title("🛡️ Stock Guardian Pro")
    st.caption("Ver 5.1 (Math Edge & Signal Reader)")
    st.divider()

    # --- 側邊欄 ---
    st.sidebar.header("📊 輸入參數")
    ticker_input = st.sidebar.text_input("股票代號", "2408")
    risk_budget = st.sidebar.number_input("單筆願賠金額 ($)", value=5000, step=1000)
    buy_price = st.sidebar.number_input("買入成本 (未買填0)", value=0.0)
    shares_held = st.sidebar.number_input("持有股數", value=1000, step=1000)
    
    df, info, final_ticker = get_stock_data(ticker_input)
    if df is None:
        st.error(f"無法獲取 {ticker_input} 資料，請檢查代號或確認市場是否開盤。")
        return

    st.sidebar.success(f"✅ 目標：{final_ticker}")
    detected = detect_industry_type_optimized(ticker_input, info)
    mode_index = 1 if detected else 0
    if detected: st.sidebar.success(f"🔍 產業：**{detected}**")
    strategy_mode = st.sidebar.radio("策略模式", ("Trend (趨勢)", "Cycle (循環)"), index=mode_index)
    use_trailing = st.sidebar.checkbox("🚀 移動停利", value=False)
    
    current_price = df['Close'].iloc[-1]
    report = analyze_logic(df, buy_price, 10, strategy_mode.split()[0], use_trailing)

    # --- 💡 訊號速查表 ---
    with st.expander("📖 點我打開：指標讀心術 (怎樣買？怎樣賣？)", expanded=True):
        st.markdown("""
        | 指標名稱 | 意義 | ✅ 什麼時候是 **買訊/安全** | 🛑 什麼時候是 **賣訊/危險** |
        | :--- | :--- | :--- | :--- |
        | **VWAP** | **法人的成本** | 股價 **>** VWAP (站上成本線) | 股價 **<** VWAP (跌破成本線) |
        | **Slope** | **衝刺的速度** | 數值 **>** 0% (且數字越大越好) | 數值 **<** 0% (車子在倒退嚕) |
        | **ATR 防線** | **最後逃生門** | 股價 **>** 防線 (還在門內) | 股價 **<** 防線 (破門而出，逃!) |
        | **位階** | **便宜還是貴** | 數值 **< 20%** (在地板，便宜) | 數值 **> 80%** (在天花板，貴) |
        | **盈虧比** | **划不划算** | 數值 **> 2.0** (贏得比輸的多) | 數值 **< 1.5** (賺太少賠太多) |
        """)

    st.divider()

    # --- 頂部指標區 (加入詳細 Tooltip) ---
    c1, c2, c3 = st.columns(3)
    c1.metric("當前股價", f"{current_price:.2f}")
    
    c2.metric(
        "盈虧比 (R/R)", 
        f"{report['rr_ratio']:.1f}", 
        help="【定義】：預期賺的錢 ÷ 預期賠的錢\n✅ 買訊：大於 2.0 (值得賭)\n🛑 賣訊：小於 1.5 (不值得冒險)"
    )
    
    c3.metric(
        "風險評分", 
        f"{report['score']} / 100", 
        help="【定義】：綜合危險程度\n✅ 安全：低於 30 分\n🛑 危險：高於 80 分"
    )

    # --- 建議倉位 ---
    atr_val = df['ATR'].iloc[-1]
    if atr_val > 0:
        suggested_shares = int(risk_budget / (2 * atr_val))
        st.info(f"🧮 **資金管理建議**：根據您的風險預算，建議最大購買 **{suggested_shares:,} 股** (約 {suggested_shares//1000} 張)。")

    st.markdown("---")
    
    # --- 關鍵數據矩陣 (加入詳細 Tooltip) ---
    k1, k2, k3, k4 = st.columns(4)
    
    k1.metric(
        "VWAP (法人成本)", 
        f"{report['vwap']:.1f}", 
        delta=f"{current_price - report['vwap']:.1f}", 
        delta_color="normal",
        help="【定義】：這個月法人的平均買入成本\n✅ 買訊：股價在數字之上 (正數)\n🛑 賣訊：股價在數字之下 (負數)"
    )
    
    k2.metric(
        "Slope (動能斜率)", 
        f"{report['slope']:.2f}%", 
        help="【定義】：股價上漲的猛烈程度\n✅ 買訊：正數 (+)，且越大越好\n🛑 賣訊：負數 (-)，或從大正數變小 (漲不動了)"
    )
    
    k3.metric(
        "ATR 吊燈防線", 
        f"{report['atr_stop_price']:.1f}", 
        help="【定義】：跌破這個價格代表趨勢反轉\n✅ 持有：股價高於此數字\n🛑 賣出：收盤價低於此數字 (無條件停損)"
    )
    
    pos_val = report['price_pos'] * 100
    k4.metric(
        "位階 (2年)", 
        f"{pos_val:.0f}%",
        help="【定義】：目前價格在過去兩年的位置\n✅ 買訊：低於 20% (低檔佈局)\n🛑 賣訊：高於 80% (高檔風險)"
    )

    # --- 詳細報告 ---
    st.subheader("📋 AI 分析報告")
    if report['score'] >= 80: st.markdown(f"<div class='status-danger'>🛑 危險 (賣出/減碼)</div>", unsafe_allow_html=True)
    elif report['score'] <= 30: st.markdown(f"<div class='status-safe'>✅ 安全 ({report['action']})</div>", unsafe_allow_html=True)
    else: st.markdown(f"<div class='status-neutral'>⚠️ 中性觀察</div>", unsafe_allow_html=True)
    
    for title, explanation in report['details']:
        with st.container():
            st.markdown(f"**{title}**")
            st.markdown(f"<div class='explanation-text'>{explanation}</div>", unsafe_allow_html=True)
            st.divider()

    # --- 圖表區 ---
    st.markdown("### 📈 戰情室")
    tab1, tab2 = st.tabs(["主圖分析 (VWAP + ATR)", "副圖分析 (OBV + Slope)"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='股價'))
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#2962FF', width=2), name='VWAP (法人成本)'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Chandelier_Exit'], line=dict(color='#D50000', width=2, dash='dot'), name='ATR 防線 (跌破賣)'))
        if buy_price > 0:
            fig.add_hline(y=buy_price, line_dash="dash", line_color="blue", annotation_text="您的成本")
        fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(t=30, b=20), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        fig_sub = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig_sub.add_trace(go.Scatter(x=df.index, y=df['OBV'], name="OBV (籌碼)", line=dict(color="orange")), row=1, col=1)
        fig_sub.add_trace(go.Scatter(x=df.index, y=df['OBV_MA20'], name="OBV均線", line=dict(color="gray", dash='dot')), row=1, col=1)
        colors = ['red' if v < 0 else 'green' for v in df['Slope_Pct']]
        fig_sub.add_trace(go.Bar(x=df.index, y=df['Slope_Pct'], name="動能斜率 %", marker_color=colors), row=2, col=1)
        fig_sub.update_layout(height=500, title_text="上圖：OBV線往上代表有人買 / 下圖：綠棒代表衝刺、紅棒代表墜落")
        st.plotly_chart(fig_sub, use_container_width=True)

# ---------------------------------------------------------
# 5. 智慧選股雷達 (加入 R/R 篩選)
# ---------------------------------------------------------
def scanner_page():
    st.title("🎯 智慧選股雷達")
    st.markdown("### AI 自動掃描 50 檔重要股票")
    st.info("💡 篩選標準：加入盈虧比 (R/R) 與 VWAP 過濾")
    
    watchlist_groups = {
        "🤖 科技權值": {
            "台積電": "2330", "鴻海": "2317", "聯發科": "2454", "廣達": "2382", 
            "台達電": "2308", "聯電": "2303", "日月光": "3711", "大立光": "3008",
            "緯創": "3231", "華碩": "2357", "欣興": "3037", "和碩": "4938"
        },
        "💰 金融保險": {
            "富邦金": "2881", "國泰金": "2882", "中信金": "2891", "兆豐金": "2886", 
            "玉山金": "2884", "元大金": "2885", "第一金": "2892", "合庫金": "5880",
            "華南金": "2880", "台新金": "2887"
        },
        "🚢 傳產循環": {
            "長榮": "2603", "陽明": "2609", "萬海": "2615", "長榮航": "2618",
            "中鋼": "2002", "台塑": "1301", "南亞": "1303", "台化": "1326",
            "台泥": "1101", "統一": "1216", "南亞科": "2408", "華邦電": "2344"
        },
        "📦 熱門 ETF": {
            "0050 台灣50": "0050", "0056 高股息": "0056", "00878 永續": "00878",
            "00929 科技優息": "00929", "00919 精選高息": "00919", "006208 富邦台50": "006208",
            "00713 低波高息": "00713", "00940 價值高息": "00940"
        }
    }
    
    if st.button("🚀 開始掃描"):
        full_list = []
        for category, items in watchlist_groups.items():
            for name, ticker in items.items():
                full_list.append((category, name, ticker))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for i, (category, name, ticker) in enumerate(full_list):
            status_text.text(f"正在分析：{name} ({ticker})...")
            try:
                time.sleep(0.3) 
                
                df, info, final_ticker = get_stock_data(ticker)
                if df is not None:
                    detected = detect_industry_type_optimized(ticker, info)
                    mode = "Cycle" if detected or "ETF" in category else "Trend"
                    if "ETF" in category: mode = "Trend" 
                    
                    report = analyze_logic(df, 0, 10, mode, False)
                    
                    status_icon = "⚪"
                    rec_text = "觀察"
                    if report['score'] <= 30 and report['rr_ratio'] > 2: 
                        status_icon = "🟢"
                        rec_text = "強力買進"
                    elif report['score'] >= 80: 
                        status_icon = "🔴"
                        rec_text = "賣出"
                    
                    pos_val = report['price_pos'] * 100
                    
                    results.append({
                        "分類": category,
                        "股票": name,
                        "現價": f"{df['Close'].iloc[-1]:.1f}",
                        "分數": report['score'],
                        "狀態": status_icon,
                        "盈虧比": f"{report['rr_ratio']:.2f}",
                        "VWAP關係": "站上" if df['Close'].iloc[-1] > report['vwap'] else "跌破",
                        "建議": rec_text
                    })
            except:
                pass
            
            progress_bar.progress((i + 1) / len(full_list))
            
        status_text.text("掃描完成！")
        
        if results:
            res_df = pd.DataFrame(results)
            res_df = res_df.sort_values(by="分數")
            
            st.dataframe(
                res_df,
                column_config={
                    "分數": st.column_config.NumberColumn(help="越低越好"),
                    "盈虧比": st.column_config.NumberColumn(help="大於2.0才值得買"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.warning("無法獲取資料，請稍後再試。")

# ---------------------------------------------------------
# 6. 說明書頁面
# ---------------------------------------------------------
def instruction_page():
    st.title("📖 媽媽的股票操作說明書")
    st.info("💡 提示：儀表板頁面現在已經有「速查表」囉，可以直接在那邊看！")
    st.divider()
    
    st.markdown("""
    <h3>1. 系統是做什麼的？</h3>
    <p>這套系統是您的 <b>「實戰過濾器」</b>。它不保證賺大錢，但它用數學幫您：</p>
    <ul>
        <li>算出這筆交易<b>划不划算</b> (盈虧比)。</li>
        <li>算出這筆交易<b>該買多少</b> (資金管理)。</li>
        <li>算出法人<b>真正的成本</b> (VWAP)。</li>
    </ul>
    <hr>
    <h3>2. 核心指標複習</h3>
    <ul>
        <li><b>VWAP (藍線)</b>：法人的成本。股價在上面才安全。</li>
        <li><b>Slope (動能)</b>：車子的油門。正數代表還在衝，變負數代表要煞車了。</li>
        <li><b>ATR 吊燈防線</b>：最後的防守點。收盤跌破這條線，無條件賣出。</li>
    </ul>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 7. 主程式入口
# ---------------------------------------------------------
def main():
    st.sidebar.title("導覽選單")
    page = st.sidebar.radio("請選擇功能", ["📊 股票分析儀表板", "🎯 智慧選股雷達", "📖 媽媽專用說明書"])
    st.sidebar.divider()
    
    if page == "📊 股票分析儀表板":
        dashboard_page()
    elif page == "🎯 智慧選股雷達":
        scanner_page()
    else:
        instruction_page()

if __name__ == "__main__":
    main()
