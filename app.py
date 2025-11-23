import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# ---------------------------------------------------------
# 1. 系統設定與 CSS
# ---------------------------------------------------------
st.set_page_config(page_title="Stock Guardian Pro", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .status-box { 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 15px; 
        border-left: 6px solid #ccc; 
        background-color: #f9f9f9;
    }
    .danger { background-color: #FFEBEE; border-color: #D32F2F; color: #C62828; }
    .safe { background-color: #E8F5E9; border-color: #2E7D32; color: #1B5E20; }
    .neutral { background-color: #FFF3E0; border-color: #EF6C00; color: #E65100; }
    .market-bear { background-color: #212121; border-color: #FF5252; color: #FF5252; } 
    
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .explanation-text { font-size: 1rem; color: #555; margin-left: 5px; line-height: 1.5; }
    .tooltip-text { color: #0066cc; font-weight: bold; text-decoration: underline dotted; cursor: help; }
    </style>
    """, unsafe_allow_html=True)

# --- Tooltip 小工具 ---
def tooltip(text, desc):
    return f'<abbr title="{desc}">{text}</abbr>'

# ---------------------------------------------------------
# 2. 資料獲取層 (最穩定的版本)
# ---------------------------------------------------------

@st.cache_data(ttl=1800)
def get_macro_data():
    """抓取大盤"""
    try:
        twii = yf.Ticker("^TWII")
        hist_tw = twii.history(period="6mo")
        if hist_tw.empty: return "Unknown", 0
        
        tw_close = hist_tw['Close'].iloc[-1]
        tw_ma60 = hist_tw['Close'].rolling(window=60).mean().iloc[-1]
        market_status = "Bear" if tw_close < tw_ma60 else "Bull"
        
        vix = yf.Ticker("^VIX")
        hist_vix = vix.history(period="5d")
        vix_val = hist_vix['Close'].iloc[-1] if not hist_vix.empty else 0
            
        return market_status, vix_val
    except:
        return "Unknown", 0

@st.cache_data(ttl=300)
def get_stock_data(ticker_input):
    """
    抓取個股資料：只要有股價就回傳，不因為 info 失敗而報錯
    """
    try:
        ticker_clean = str(ticker_input).replace(".TW", "").replace(".TWO", "").strip()
        
        # 1. 嘗試上市
        try_ticker = f"{ticker_clean}.TW"
        stock = yf.Ticker(try_ticker)
        df = stock.history(period="2y")
        
        # 2. 嘗試上櫃
        if df.empty:
            try_ticker = f"{ticker_clean}.TWO"
            stock = yf.Ticker(try_ticker)
            df = stock.history(period="2y")
            
        if df.empty: return None, {}, None # 真的抓不到

        # 3. 嘗試抓 info (失敗就算了，給空字典)
        try: info = stock.info
        except: info = {}

        # --- 指標運算 ---
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['Bias'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # MVWAP (半年線級別法人成本)
        anchor_window = 120
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['TPV'] = df['TP'] * df['Volume']
        df['Cum_TPV'] = df['TPV'].rolling(window=anchor_window).sum()
        df['Cum_Vol'] = df['Volume'].rolling(window=anchor_window).sum()
        df['MVWAP'] = df['Cum_TPV'] / df['Cum_Vol'].replace(0, np.nan)
        
        # RVOL
        df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
        df['RVOL'] = df['Volume'] / df['Vol_MA20'].replace(0, np.nan)
        
        # OBV
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()

        # 風控指標
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['High_20'] = df['High'].shift(1).rolling(window=20).max()
        df['Chandelier_Exit'] = df['High_20'] - (2.0 * df['ATR'])
        
        # 位階
        lookback = 500
        if len(df) > lookback:
            h, l = df['High'].rolling(window=lookback).max(), df['Low'].rolling(window=lookback).min()
        else:
            h, l = df['High'].max(), df['Low'].min()
        df['Price_Pos'] = (df['Close'] - l) / (h - l).replace(0, np.nan)

        return df, info, try_ticker
    except:
        return None, {}, None

def calculate_seasonality(df):
    try:
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly.index.month
        df_monthly['Pct_Change'] = df_monthly['Close'].pct_change() * 100
        seasonal_stats = df_monthly.groupby('Month')['Pct_Change'].mean()
        win_rate = df_monthly[df_monthly['Pct_Change'] > 0].groupby('Month')['Pct_Change'].count() / df_monthly.groupby('Month')['Pct_Change'].count() * 100
        return seasonal_stats, win_rate
    except:
        return None, None

def detect_industry_type(info):
    if not info: return None
    sector = info.get('sector', '')
    industry = info.get('industry', '')
    summary = info.get('longBusinessSummary', '')
    short_name = info.get('shortName', '')
    if 'ETF' in short_name or 'Dividend' in short_name: return 'ETF'
    cycle_keywords = ['Semiconductors', 'Memory', 'DRAM', 'Flash', 'Marine', 'Shipping', 'Freight', 'Steel', 'Iron', 'Panel', 'LCD']
    check_str = (str(sector) + " " + str(industry) + " " + str(summary)).lower()
    for kw in cycle_keywords:
        if kw.lower() in check_str: return kw
    return None

# ---------------------------------------------------------
# 3. AI 核心邏輯
# ---------------------------------------------------------
def analyze_logic(df, info, buy_price, stop_loss_pct, strategy_mode, use_trailing, macro_data, manual_inst_score):
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    close = curr['Close']
    ma20 = curr['MA20']
    ma60 = curr['MA60']
    atr_stop = curr['Chandelier_Exit']
    mvwap = curr['MVWAP']
    rsi = curr['RSI']
    price_pos = curr['Price_Pos']
    rvol = curr['RVOL']
    
    eps = info.get('trailingEps', None)
    mkt_status, vix_val = macro_data

    report = {
        "score": 0, "action": "觀望 / 持有", "details": [],
        "atr_stop_price": atr_stop, "trailing_stop_price": 0.0,
        "price_pos": price_pos, "vwap": mvwap, "market_penalty": False
    }

    tech_score = 0
    chip_score = 0
    fund_score = 0

    # 1. 宏觀與基本面
    if mkt_status == "Bear":
        fund_score -= 40
        report['market_penalty'] = True
        report['details'].append(("[宏觀] ☁️ 大盤空頭", "加權指數跌破季線，建議保守。"))
    
    if eps is not None and eps < 0:
        fund_score -= 20
        report['details'].append(("[財報] ⚠️ 基本面虧損", "公司賠錢中。"))

    if strategy_mode == "Cycle":
        if price_pos < 0.2:
            if close > ma20:
                fund_score += 50
                report['details'].append(("[價值] 💎 底部轉強", "位階低且站上月線。"))
            else:
                fund_score += 10
                report['details'].append(("[價值] 📉 低檔弱勢", "股價便宜但趨勢仍弱。"))
        elif price_pos > 0.8:
            fund_score -= 50
            report['details'].append(("[價值] ⛰️ 歷史高檔", "位階 > 80%，風險高。"))

    fund_score = max(-100, min(100, fund_score))

    # 2. 技術面
    if close > ma60: tech_score += 20
    else: tech_score -= 30
    
    if close > ma20: tech_score += 10
    else: tech_score -= 10

    break_today = close < atr_stop
    break_yesterday = prev['Close'] < prev['Chandelier_Exit']
    
    if break_today and break_yesterday:
        tech_score -= 60
        report['details'].append(("[防守] 🛑 趨勢確認反轉", "連續兩日跌破吊燈防線，建議賣出。"))
    elif break_today:
        tech_score -= 20
        report['details'].append(("[防守] ⚠️ 跌破 ATR 防線", "首日跌破，密切觀察。"))
    else:
        tech_score += 10

    if rsi > 80: tech_score -= 10
    
    tech_score = max(-100, min(100, tech_score))

    # 3. 籌碼面
    mvwap_slope_up = mvwap > prev['MVWAP']
    
    if close > mvwap:
        if mvwap_slope_up:
            chip_score += 40
            report['details'].append(("[籌碼] ✅ 站上上揚成本線", "股價強於法人成本，多頭強勢。"))
        else:
            chip_score += 10
            report['details'].append(("[籌碼] ⚠️ 站上下彎成本線", "雖然站上 MVWAP 但趨勢向下，僅視為反彈。"))
    else:
        chip_score -= 40
        report['details'].append(("[籌碼] ❌ 跌破法人成本", "股價弱於平均成本。"))

    if manual_inst_score != 0:
        chip_score += (manual_inst_score * 3)
        status = "買超" if manual_inst_score > 0 else "賣超"
        report['details'].append(("[籌碼] 🖐️ 參考新聞資訊", f"外資近期 {status}。"))
    else:
        if rvol > 1.5:
            if close > prev['Close']:
                chip_score += 10
                report['details'].append(("[籌碼] 📈 出量上漲", f"量能放大 (RVOL {rvol:.1f})。"))
            else:
                chip_score -= 20
                report['details'].append(("[籌碼] 📉 出量下跌", f"量能放大 (RVOL {rvol:.1f})，疑出貨。"))

    chip_score = max(-100, min(100, chip_score))

    # 4. 總結
    final_score = (tech_score * 0.4) + (chip_score * 0.4) + (fund_score * 0.2)
    
    if buy_price > 0:
        user_stop_price = buy_price * (1 - stop_loss_pct / 100)
        if current_close <= user_stop_price:
            final_score = -100
            report['details'].append(("[紀律] 🛑 觸及硬性停損", f"虧損已達 {stop_loss_pct}%。"))

        if use_trailing and current_close > buy_price:
            recent_high = df['High'].tail(60).max()
            if recent_high < buy_price: recent_high = buy_price
            report['trailing_stop_price'] = recent_high * 0.90
            
            if current_close < report['trailing_stop_price']:
                final_score = -100
                report['details'].append(("[紀律] 💰 觸發移動停利", "回檔 10% 獲利了結。"))

    report['score'] = final_score
    if final_score >= 40: report['action'] = "做多/持有"
    elif final_score <= -40: report['action'] = "賣出/空手"
    else: report['action'] = "觀望"
    
    return report, tech_score, chip_score, fund_score

# ---------------------------------------------------------
# 4. 儀表板頁面 (保證輸入框不消失)
# ---------------------------------------------------------
def dashboard_page():
    st.title("🛡️ Stock Guardian Pro")
    st.caption("Ver 15.1 (Full Control)")
    
    mkt_status, vix_val = get_macro_data()
    if mkt_status == "Bear":
        st.markdown("""<div class='status-box market-bear'>⚠️ 市場警報：大盤走空 (跌破季線)</div>""", unsafe_allow_html=True)
    else:
        st.success(f"✅ 大盤多頭，VIX：{vix_val:.1f}")

    st.divider()

    # --- 側邊欄 (所有輸入框都放在這裡，保證不消失) ---
    st.sidebar.header("📊 1. 查詢股票")
    ticker_input = st.sidebar.text_input("股票代號", "2408")
    
    # 為了讓側邊欄的選項在抓不到資料時也能顯示，我們先把 UI 畫出來
    # 然後再嘗試獲取資料
    
    # 嘗試獲取資料
    df, info, final_ticker = get_stock_data(ticker_input)
    
    # 自動偵測 (如果有 info)
    detected = detect_industry_type(info)
    # 預設索引：如果有偵測到循環股，預設選 Cycle (index 1)，否則 Trend (index 0)
    default_idx = 1 if detected else 0
    
    # ★★★ 關鍵修復：模式切換按鈕回歸 ★★★
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 2. 策略設定")
    
    if detected:
        st.sidebar.success(f"🔍 AI 偵測：**{detected}** (循環股)")
    else:
        st.sidebar.info("🔍 AI 偵測：一般趨勢股")
        
    strategy_mode = st.sidebar.radio(
        "分析模式 (可手動切換)", 
        ("Trend (趨勢)", "Cycle (循環)"), 
        index=default_idx,
        help="Trend：適合台積電、金融股 (破線就跑)\nCycle：適合航運、記憶體 (越跌越買)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("💰 3. 持倉設定 (選填)")
    buy_price = st.sidebar.number_input("買入成本 (未買填0)", value=0.0)
    shares_held = st.sidebar.number_input("持有股數", value=1000, step=1000)
    stop_loss_pct = st.sidebar.number_input("容忍虧損 %", value=10)
    use_trailing = st.sidebar.checkbox("🚀 啟用移動停利", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📰 4. 外資動向 (選填)")
    inst_option = st.sidebar.selectbox(
        "外資買賣超？",
        ("🤷‍♂️ 不知道 / 沒看", "🔴 新聞說外資大賣", "🟢 新聞說外資大買")
    )
    manual_score = 0
    if "大賣" in inst_option: manual_score = -10
    elif "大買" in inst_option: manual_score = 10
    
    st.sidebar.markdown("---")
    debug_mode = st.sidebar.checkbox("🔧 開發者驗證模式", value=False)

    # --- 主畫面邏輯 ---
    if df is None:
        st.error(f"❌ 找不到代號 {ticker_input} 的資料。請確認輸入正確 (例如 2330 或 0050)。")
        return # 只有主畫面停止，側邊欄依然存在

    # 若成功獲取，顯示內容
    st.sidebar.success(f"✅ 已載入：{final_ticker}")

    report, t_s, c_s, f_s = analyze_logic(
        df, info, buy_price, stop_loss_pct, strategy_mode.split()[0], use_trailing, 
        (mkt_status, vix_val), manual_score
    )
    
    current_price = df['Close'].iloc[-1]
    pl_amount = (current_price - buy_price) * shares_held if buy_price > 0 else 0
    pl_pct = (pl_amount / (buy_price * shares_held)) * 100 if buy_price > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("當前股價", f"{current_price:.2f}")
    c2.metric("預估損益", f"${int(pl_amount):,}", f"{pl_pct:.2f}%")
    
    final_score = report['score']
    if final_score >= 40:
        score_text, box_class = "🟢 強力買進", "safe"
    elif final_score <= -40:
        score_text, box_class = "🔴 強力賣出", "danger"
    else:
        score_text, box_class = "🟠 中性觀望", "neutral"
        
    c3.metric("AI 綜合建議", score_text, f"{final_score:.1f} 分")

    st.markdown("---")
    
    k1, k2, k3, k4 = st.columns(4)
    vwap_val = report['vwap']
    k1.metric("MVWAP 法人成本", f"{vwap_val:.1f}", delta=f"{current_price-vwap_val:.1f}")
    pos_val = report['price_pos'] * 100
    k2.metric("股價位階", f"{pos_val:.0f}%")
    k3.metric("OBV 籌碼", report['obv_trend'])
    k4.metric("ATR 吊燈防線", f"{report['atr_stop_price']:.1f}")

    with st.container():
        clean_ticker = final_ticker.replace(".TW", "").replace(".TWO", "")
        yahoo_link = f"https://tw.stock.yahoo.com/quote/{clean_ticker}/institutional-trading"
        st.write("🔎 **進階查詢**")
        st.link_button("前往 Yahoo 查看外資買賣超", yahoo_link)

    st.markdown("---")

    st.subheader("📋 AI 診斷報告")
    
    s1, s2, s3 = st.columns(3)
    s1.metric("技術面 (40%)", f"{t_s:.0f}")
    s2.metric("籌碼面 (40%)", f"{c_s:.0f}")
    s3.metric("基本/宏觀 (20%)", f"{f_s:.0f}")
    
    st.markdown(f"""<div class='status-box {box_class}'><b>綜合評價：{score_text}</b></div>""", unsafe_allow_html=True)

    if report['details']:
        for title, text in report['details']:
            st.info(f"**{title}**\n\n{text}")
    else:
        st.success("各項指標走勢正常。")

    if debug_mode:
        st.divider()
        st.write("🔧 Debug Data (含預估量):")
        st.dataframe(df[['Close', 'Volume', 'MVWAP', 'RVOL']].tail())

    st.divider()
    st.markdown("### 📈 趨勢戰情室")
    tab1, tab2, tab3 = st.tabs(["主圖 (價格+防線)", "副圖 (籌碼 OBV)", "季節性"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='股價'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MVWAP'], line=dict(color='#2962FF', width=2), name='MVWAP'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Chandelier_Exit'], line=dict(color='#D50000', width=2, dash='dot'), name='ATR 防線'))
        if buy_price > 0:
            fig.add_hline(y=buy_price, line_dash="dash", line_color="gray", annotation_text="成本")
        if use_trailing and report['trailing_stop_price'] > 0:
            fig.add_hline(y=report['trailing_stop_price'], line_color="purple", line_width=3, annotation_text="移動停利")
        fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(t=30, b=20), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        fig_obv = make_subplots(specs=[[{"secondary_y": True}]])
        fig_obv.add_trace(go.Scatter(x=df.index, y=df['Close'], name="股價", line=dict(color="gray", width=1)), secondary_y=True)
        fig_obv.add_trace(go.Scatter(x=df.index, y=df['OBV'], name="OBV", line=dict(color="orange", width=2)), secondary_y=False)
        fig_obv.add_trace(go.Scatter(x=df.index, y=df['OBV_MA20'], name="OBV均線", line=dict(color="blue", width=1, dash='dot')), secondary_y=False)
        fig_obv.update_layout(height=500, legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_obv, use_container_width=True)

    with tab3:
        season_stats, win_rate = calculate_seasonality(df)
        if season_stats is not None:
            fig_season = go.Figure()
            colors = ['#EF5350' if x > 0 else '#26A69A' for x in season_stats.values]
            fig_season.add_trace(go.Bar(x=season_stats.index, y=season_stats.values, marker_color=colors, name='漲跌幅'))
            fig_season.add_trace(go.Scatter(x=win_rate.index, y=win_rate.values, name='勝率', yaxis='y2', line=dict(color='blue', width=2, dash='dot')))
            fig_season.update_layout(xaxis=dict(title="月份"), yaxis2=dict(title="勝率 %", overlaying='y', side='right', range=[0, 100]), height=500)
            st.plotly_chart(fig_season, use_container_width=True)

# ---------------------------------------------------------
# 5. 智慧選股雷達
# ---------------------------------------------------------
def scanner_page():
    st.title("🎯 智慧選股雷達")
    mkt_status, _ = get_macro_data()
    
    if mkt_status == "Bear":
        st.error("⚠️ 警告：大盤空頭，選股評分已自動加嚴。")
    else:
        st.success("✅ 大盤多頭，選股環境良好。")

    st.info("💡 掃描約需 60 秒。")
    
    watchlist_groups = {
        "🤖 科技權值": {"台積電": "2330", "鴻海": "2317", "聯發科": "2454", "廣達": "2382", "台達電": "2308"},
        "💰 金融保險": {"富邦金": "2881", "國泰金": "2882", "中信金": "2891", "兆豐金": "2886"},
        "🚢 傳產循環": {"長榮": "2603", "陽明": "2609", "中鋼": "2002", "南亞科": "2408", "台塑": "1301"},
        "📦 熱門 ETF": {"0050": "0050", "0056": "0056", "00878": "00878", "00929": "00929"}
    }
    
    if st.button("🚀 開始掃描"):
        full_list = []
        for category, items in watchlist_groups.items():
            for name, ticker in items.items():
                full_list.append((category, name, ticker))
        
        progress_bar = st.progress(0)
        results = []
        
        for i, (category, name, ticker) in enumerate(full_list):
            try:
                time.sleep(0.1) 
                df, info, final_ticker = get_stock_data(ticker)
                if df is not None:
                    detected = detect_industry_type(info)
                    mode = "Cycle" if detected or "ETF" in category else "Trend"
                    if "ETF" in category: mode = "Trend"
                    
                    current_price = df['Close'].iloc[-1]
                    # 掃描時手動籌碼設為 0
                    report, _, _, _ = analyze_logic(
                        df, info, current_price, 10, mode, False, (mkt_status, 0), 0
                    )
                    
                    final_score = report['score']
                    status_icon = "⚪"
                    if final_score >= 40: status_icon = "🟢" 
                    elif final_score <= -40: status_icon = "🔴" 
                    else: status_icon = "🟠" 
                    
                    pos_val = report['price_pos'] * 100
                    
                    results.append({
                        "分類": category,
                        "代號": final_ticker.replace(".TW", "").replace(".TWO", ""),
                        "名稱": name,
                        "現價": f"{current_price:.1f}",
                        "分數": f"{final_score:.1f}",
                        "狀態": status_icon,
                        "位階": f"{pos_val:.0f}%",
                        "建議": report['action']
                    })
            except:
                pass
            progress_bar.progress((i + 1) / len(full_list))
            
        st.success("掃描完成！")
        if results:
            res_df = pd.DataFrame(results).sort_values(by="分數", ascending=False)
            st.dataframe(res_df, hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# 6. 說明書
# ---------------------------------------------------------
def instruction_page():
    st.title("📖 股票操作說明書")
    st.markdown("""
    ### 1. 核心功能
    * **MVWAP (法人成本)**：這條藍色線模擬法人半年的平均成本。股價在上面代表法人賺錢，趨勢偏多。
    * **ATR 吊燈防線**：這是紅色的虛線，跌破賣出。
    
    ### 2. 關於分數 (-100 ~ +100)
    * **🟢 正分 (> +40)**：看多！
    * **🔴 負分 (< -40)**：看空！
    * **🟠 零分附近**：觀望。

    ### 3. 盤中量能推算 (Ver 13.0 新功能)
    系統會根據現在幾點，自動推算今天的預估成交量。
    這解決了早上看盤時，因為累積量太少而誤判「量縮」的問題。
    """)

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
