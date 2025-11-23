import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ---------------------------------------------------------
# 1. 系統設定
# ---------------------------------------------------------
st.set_page_config(page_title="Stock Guardian Pro", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .status-box { padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 6px solid #ccc; background-color: #f9f9f9; }
    .danger { background-color: #FFEBEE; border-color: #D32F2F; color: #C62828; }
    .safe { background-color: #E8F5E9; border-color: #2E7D32; color: #1B5E20; }
    .neutral { background-color: #FFF3E0; border-color: #EF6C00; color: #E65100; }
    .market-bear { background-color: #212121; border-color: #FF5252; color: #FF5252; } 
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .explanation-text { font-size: 1rem; color: #555; margin-left: 5px; line-height: 1.5; }
    .tooltip-text { color: #0066cc; font-weight: bold; text-decoration: underline dotted; cursor: help; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 資料獲取層 (回歸最原始、最穩定的邏輯)
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def get_stock_data_robust(ticker_input):
    """
    最穩定的抓取邏輯：
    1. 不做任何預判，直接暴力測試 .TW 和 .TWO
    2. Info 抓不到就直接忽略，絕不因為 Info 讓程式崩潰
    """
    # 清理輸入，只留代號本體 (例如 "2408")
    ticker_clean = str(ticker_input).strip().upper().replace(".TW", "").replace(".TWO", "")
    
    df = pd.DataFrame()
    final_ticker = ""
    info = {}

    # 定義嘗試清單
    try_list = [f"{ticker_clean}.TW", f"{ticker_clean}.TWO", ticker_clean]

    # 迴圈嘗試
    for t in try_list:
        try:
            stock = yf.Ticker(t)
            temp_df = stock.history(period="2y") # 抓 2 年
            
            if not temp_df.empty:
                df = temp_df
                final_ticker = t
                
                # 嘗試抓 Info (這是最常報錯的地方，所以獨立包起來)
                try:
                    info = stock.info
                except:
                    info = {} # 抓不到就算了，給空字典
                
                break # 成功就跳出迴圈
        except:
            continue # 失敗就試下一個

    # 如果試完全部還是空的，回傳 None
    if df.empty:
        return None, {}, None

    # --- 計算指標 (只要有 df 就能算) ---
    try:
        # 均線
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['Bias'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # KD
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=9, smooth_window=3)
        df['K'] = stoch.stoch()

        # OBV
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['OBV_MA20'] = df['OBV'].rolling(window=20).mean()

        # ATR & 吊燈停損 (Chandelier Exit)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['High_20'] = df['High'].shift(1).rolling(window=20).max()
        df['Chandelier_Exit'] = df['High_20'] - (2.0 * df['ATR'])
        
        # MVWAP (法人成本)
        v = df['Volume'].values
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['MVWAP'] = (tp * v).rolling(window=120).sum() / v.rolling(window=120).sum().replace(0, np.nan)

        # 位階 (Price Position)
        lookback = 500
        if len(df) > lookback:
            h = df['High'].rolling(window=lookback).max()
            l = df['Low'].rolling(window=lookback).min()
        else:
            h = df['High'].max()
            l = df['Low'].min()
        df['Price_Pos'] = (df['Close'] - l) / (h - l).replace(0, np.nan)

        return df, info, final_ticker

    except Exception as e:
        st.error(f"指標計算錯誤: {e}")
        return None, {}, None

@st.cache_data(ttl=1800)
def get_macro_data():
    try:
        twii = yf.Ticker("^TWII")
        hist = twii.history(period="3mo")
        if hist.empty: return "Unknown", 0
        close = hist['Close'].iloc[-1]
        ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        status = "Bear" if close < ma20 else "Bull"
        
        vix = yf.Ticker("^VIX")
        v_hist = vix.history(period="5d")
        vix_val = v_hist['Close'].iloc[-1] if not v_hist.empty else 0
        return status, vix_val
    except:
        return "Unknown", 0

def detect_industry_type(info):
    if not info: return None
    check_str = (str(info.get('sector','')) + str(info.get('industry','')) + str(info.get('longBusinessSummary',''))).lower()
    if 'etf' in info.get('shortName', '').lower(): return 'ETF'
    keywords = ['semiconductor', 'memory', 'dram', 'marine', 'shipping', 'steel', 'iron', 'panel', 'lcd']
    for k in keywords:
        if k in check_str: return k
    return None

# ---------------------------------------------------------
# 3. AI 邏輯
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
    
    obv_curr = curr['OBV']
    obv_ma = curr['OBV_MA20']
    obv_trend = "📈 資金流入" if obv_curr > obv_ma else "📉 資金流出"
    
    eps = info.get('trailingEps', None)
    mkt_status, vix_val = macro_data

    report = {
        "score": 0, "action": "觀望", "details": [],
        "atr_stop_price": atr_stop, "trailing_stop_price": 0.0,
        "price_pos": price_pos, "vwap": mvwap, "obv_trend": obv_trend
    }

    tech_score, chip_score, fund_score = 0, 0, 0

    # 1. 宏觀/基本
    if mkt_status == "Bear":
        fund_score -= 40
        report['details'].append(("[宏觀] ☁️ 大盤走空", "加權指數跌破月線，建議保守。"))
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
                report['details'].append(("[價值] 📉 低檔弱勢", "便宜但趨勢向下。"))
        elif price_pos > 0.8:
            fund_score -= 50
            report['details'].append(("[價值] ⛰️ 歷史高檔", "位階過高。"))

    fund_score = max(-100, min(100, fund_score))

    # 2. 技術
    if close > ma60: tech_score += 20
    else: tech_score -= 30
    if close > ma20: tech_score += 10
    else: tech_score -= 10
    
    if close < atr_stop:
        tech_score -= 60
        report['details'].append(("[防守] 🛑 跌破 ATR 防線", "趨勢反轉，建議賣出。"))
    
    if rsi > 80: tech_score -= 10
    tech_score = max(-100, min(100, tech_score))

    # 3. 籌碼
    if close > mvwap: chip_score += 40
    else: chip_score -= 40
    
    if manual_inst_score != 0:
        chip_score += (manual_inst_score * 3)
        st_text = "買超" if manual_inst_score > 0 else "賣超"
        report['details'].append(("[籌碼] 🖐️ 手動輸入", f"外資 {st_text}。"))
    else:
        if obv_curr > obv_ma: chip_score += 20
        else: chip_score -= 20

    chip_score = max(-100, min(100, chip_score))

    # 4. 總結
    final_score = (tech_score * 0.4) + (chip_score * 0.4) + (fund_score * 0.2)
    
    # 停損檢查
    if buy_price > 0:
        user_stop_price = buy_price * (1 - stop_loss_pct / 100)
        if close <= user_stop_price:
            final_score = -100
            report['details'].append(("[紀律] 🛑 觸及停損", f"虧損已達 {stop_loss_pct}%。"))
        
        if use_trailing and close > buy_price:
            recent_high = df['High'].tail(60).max()
            if recent_high < buy_price: recent_high = buy_price
            report['trailing_stop_price'] = recent_high * 0.90
            if close < report['trailing_stop_price']:
                final_score = -100
                report['details'].append(("[紀律] 💰 觸發移動停利", "回檔 10% 獲利了結。"))

    report['score'] = final_score
    if final_score >= 40: report['action'] = "做多/持有"
    elif final_score <= -40: report['action'] = "賣出/空手"
    else: report['action'] = "觀望"
    
    return report, tech_score, chip_score, fund_score

# ---------------------------------------------------------
# 4. 儀表板頁面
# ---------------------------------------------------------
def dashboard_page():
    st.title("🛡️ Stock Guardian Pro")
    
    # 1. 顯示大盤
    mkt_status, vix_val = get_macro_data()
    if mkt_status == "Bear":
        st.markdown("""<div class='status-box market-bear'>⚠️ 市場警報：大盤走空 (跌破月線)</div>""", unsafe_allow_html=True)
    else:
        st.success(f"✅ 大盤多頭，VIX：{vix_val:.1f}")

    st.divider()

    # --- 側邊欄 (保證顯示) ---
    st.sidebar.header("📊 1. 查詢股票")
    ticker_input = st.sidebar.text_input("股票代號", "2408")
    
    st.sidebar.markdown("---")
    st.sidebar.header("💰 2. 持倉設定 (選填)")
    buy_price = st.sidebar.number_input("買入成本 (未買填0)", value=0.0)
    shares_held = st.sidebar.number_input("持有股數", value=1000, step=1000)
    stop_loss_pct = st.sidebar.number_input("容忍虧損 %", value=10)
    use_trailing = st.sidebar.checkbox("🚀 啟用移動停利", value=False)
    
    st.sidebar.markdown("---")
    st.sidebar.header("📰 3. 外資動向 (選填)")
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
    df, info, final_ticker = get_stock_data_robust(ticker_input)
    
    if df is None:
        st.error(f"❌ 找不到資料 ({ticker_input})。請確認代號是否正確。")
        return

    st.sidebar.success(f"✅ 已載入：{final_ticker}")
    
    detected = detect_industry_type(info)
    if detected: st.sidebar.info(f"🔍 循環股：{detected}")
    else: st.sidebar.info("🔍 一般趨勢股")
    strategy_mode = "Cycle" if detected else "Trend"

    report, t_s, c_s, f_s = analyze_logic(
        df, info, buy_price, stop_loss_pct, strategy_mode, use_trailing, 
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
        st.write("🔧 Debug Data:")
        st.dataframe(df.tail())

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
# 5. 智慧選股雷達 (修復：跳過 info 避免報錯)
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
        "🤖 科技權值": {"台積電": "2330", "鴻海": "2317", "聯發科": "2454", "廣達": "2382"},
        "💰 金融保險": {"富邦金": "2881", "國泰金": "2882", "中信金": "2891", "兆豐金": "2886"},
        "🚢 傳產循環": {"長榮": "2603", "陽明": "2609", "中鋼": "2002", "南亞科": "2408"},
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
                df, info, final_ticker = get_stock_data_robust(ticker)
                if df is not None:
                    # 簡易分類
                    mode = "Trend"
                    if "循環" in category or "南亞科" in name or "長榮" in name: mode = "Cycle"
                    
                    current_price = df['Close'].iloc[-1]
                    report, _, _, _ = analyze_logic(
                        df, {}, current_price, 10, mode, False, (mkt_status, 0), 0
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
    * **MVWAP (法人成本)**：這條藍色線模擬法人半年的平均成本。
    * **ATR 吊燈防線**：這是紅色的虛線，跌破賣出。
    
    ### 2. 關於分數 (-100 ~ +100)
    * **🟢 正分 (> +40)**：看多！
    * **🔴 負分 (< -40)**：看空！
    * **🟠 零分附近**：觀望。
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
