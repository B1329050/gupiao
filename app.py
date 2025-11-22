import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ---------------------------------------------------------
# 1. 系統設定與 CSS
# ---------------------------------------------------------
st.set_page_config(page_title="Stock Guardian Ultimate", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    /* 風險訊號樣式 */
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
    .explanation-text { font-size: 1rem; color: #444; margin-left: 5px; line-height: 1.5; }
    
    /* 懸停提示字詞樣式 (Tooltip) */
    abbr {
        text-decoration: underline dotted #0066cc; 
        cursor: help;
        color: #0066cc;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 懸停提示小工具 ---
def tooltip(text, desc):
    """產生帶有懸停解釋的 HTML 標籤"""
    return f'<abbr title="{desc}">{text}</abbr>'

# ---------------------------------------------------------
# 2. 資料獲取與運算
# ---------------------------------------------------------
@st.cache_data(ttl=900)
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="5y")
        if df.empty: return None, None
        info = stock.info

        # 基礎指標
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['Bias'] = ((df['Close'] - df['MA20']) / df['MA20']) * 100
        
        # 進階指標
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=9, smooth_window=3)
        df['K'] = stoch.stoch() # KD指標中的K值
        
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
        
        return df, info
    except:
        return None, None

def calculate_seasonality(df):
    df_monthly = df.copy()
    df_monthly['Month'] = df_monthly.index.month
    df_monthly['Pct_Change'] = df_monthly['Close'].pct_change() * 100
    seasonal_stats = df_monthly.groupby('Month')['Pct_Change'].mean()
    win_rate = df_monthly[df_monthly['Pct_Change'] > 0].groupby('Month')['Pct_Change'].count() / df_monthly.groupby('Month')['Pct_Change'].count() * 100
    return seasonal_stats, win_rate

def detect_industry_type(info):
    sector = info.get('sector', '')
    industry = info.get('industry', '')
    summary = info.get('longBusinessSummary', '')
    cycle_keywords = ['Semiconductors', 'Memory', 'DRAM', 'Flash', 'Marine', 'Shipping', 'Freight', 'Transport', 'Steel', 'Iron', 'Metal', 'Chemical', 'Oil', 'Panel', 'Display', 'LCD']
    
    primary_check = (str(sector) + " " + str(industry)).lower()
    for kw in cycle_keywords:
        if kw.lower() in primary_check: return kw
    summary_check = str(summary).lower()
    for kw in cycle_keywords:
        if kw.lower() in summary_check: return kw
    return None

def analyze_logic(df, info, buy_price, stop_loss_pct, strategy_mode, use_trailing):
    # 取得最新數據
    current_close = df['Close'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma60 = df['MA60'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    mfi = df['MFI'].iloc[-1]
    bias = df['Bias'].iloc[-1]
    k_val = df['K'].iloc[-1] 
    
    pb_ratio = info.get('priceToBook', None)
    
    price_change_5d = current_close - df['Close'].iloc[-5]
    obv_change_5d = df['OBV'].iloc[-1] - df['OBV'].iloc[-5]
    
    report = {
        "score": 50, "action": "觀望 / 持有", "details": [],
        "atr_stop_price": current_close - (2.0 * atr), "trailing_stop_price": 0.0, "obv_trend": "持平"
    }
    
    if obv_change_5d > 0: report['obv_trend'] = "上升 (資金流入)"
    elif obv_change_5d < 0: report['obv_trend'] = "下降 (資金流出)"

    # 邏輯判斷
    if bias > 10:
        report['score'] += 15
        report['details'].append(("[風險] 乖離率過大", "股價衝太快，像橡皮筋拉太緊，容易回檔。"))
    elif bias < -10 and strategy_mode == "Cycle":
        report['score'] -= 10
        report['details'].append(("[機會] 負乖離過大", "股價跌太深，容易出現反彈。"))

    if price_change_5d >= 0 and obv_change_5d < 0:
        report['score'] += 20
        report['details'].append(("[籌碼背離] 主力正在偷賣", "股價沒跌但大戶在跑，危險訊號。"))
    if price_change_5d <= 0 and obv_change_5d > 0:
        report['score'] -= 15
        report['details'].append(("[籌碼背離] 主力正在偷買", "股價在跌但大戶在撿，底部訊號。"))

    if strategy_mode == "Trend":
        if current_close < ma20:
            report['score'] += 20
            report['details'].append(("[警告] 跌破月線", "短期支撐破裂。"))
        if current_close < ma60:
            report['score'] += 30
            report['details'].append(("[危險] 跌破季線", "中期趨勢轉空。"))
        if current_close < report['atr_stop_price']:
            report['score'] += 40
            report['details'].append(("[賣出訊號] 跌破 ATR 安全線", "跌破主力防守價，請離場。"))
        if rsi > 80 or mfi > 80:
            report['score'] += 10
            report['details'].append(("[風險] 指標過熱", "市場太嗨，容易回檔。"))

    elif strategy_mode == "Cycle":
        report['action'] = "觀察循環位階"
        if pb_ratio:
            if pb_ratio < 1.0:
                report['score'] = 10
                report['action'] = "建議分批佈局 (價值區)"
                report['details'].append(("[機會] P/B < 1.0", "股價低於淨值，歷史底部。"))
            elif pb_ratio < 1.5:
                report['score'] = 40
                report['action'] = "續抱 / 觀望"
                report['details'].append(("[中性] P/B 正常", "價格合理。"))
            else:
                report['score'] = 70
                report['details'].append(("[注意] P/B 過高", "循環股價格偏貴。"))
        if k_val < 20:
            report['score'] -= 10
            report['details'].append(("[訊號] KD低檔鈍化", "嚴重超賣，隨時可能反彈。"))

    user_stop_price = buy_price * (1 - stop_loss_pct / 100)
    if current_close <= user_stop_price:
        report['score'] = 100
        report['details'].append(("[強制停損] 觸及虧損極限", "請執行紀律。"))

    if use_trailing:
        recent_high = df['High'].tail(60).max()
        if buy_price > recent_high: recent_high = buy_price
        report['trailing_stop_price'] = recent_high * 0.90
        if current_close < report['trailing_stop_price']:
            report['score'] = 100
            report['details'].append(("[停利訊號] 觸發移動停利", "回檔 10%，鎖住獲利。"))

    report['score'] = min(100, max(0, report['score']))
    return report

# ---------------------------------------------------------
# 3. 頁面 A: 股票分析儀表板 (Main Dashboard)
# ---------------------------------------------------------
def dashboard_page():
    st.title("🛡️ 股票決策輔助系統 (Ultimate)")
    st.caption("請在左側輸入資料，系統將自動運算風險與建議。")
    st.divider()

    # 側邊欄
    st.sidebar.header("📊 輸入參數")
    ticker_input = st.sidebar.text_input("股票代號", "2408")
    ticker = f"{ticker_input}.TW" if not ticker_input.endswith(".TW") else ticker_input
    buy_price = st.sidebar.number_input("買入成本", value=60.0)
    shares_held = st.sidebar.number_input("持有股數", value=1000, step=1000)
    stop_loss_pct = st.sidebar.number_input("容忍虧損 %", value=10)
    
    df, info = get_stock_data(ticker)
    if df is None:
        st.error("查無資料，請檢查代號或網路連線。")
        return

    detected = detect_industry_type(info)
    mode_index = 1 if detected else 0
    
    st.sidebar.markdown("---")
    if detected: st.sidebar.success(f"🔍 偵測為：**{detected}** (循環股)")
    else: st.sidebar.info("🔍 偵測為：**一般趨勢股**")

    strategy_mode = st.sidebar.radio("模式", ("Trend (趨勢)", "Cycle (循環)"), index=mode_index)
    st.sidebar.markdown("---")
    use_trailing = st.sidebar.checkbox("🚀 啟用移動停利", value=False)

    report = analyze_logic(df, info, buy_price, stop_loss_pct, strategy_mode.split()[0], use_trailing)
    
    current_price = df['Close'].iloc[-1]
    pl_amount = (current_price - buy_price) * shares_held
    pl_pct = (pl_amount / (buy_price * shares_held)) * 100 if buy_price > 0 else 0
    
    # 看板
    c1, c2, c3 = st.columns(3)
    c1.metric("當前股價", f"{current_price:.2f}")
    c2.metric("總損益", f"{int(pl_amount):,} 元", f"{pl_pct:.2f}%")
    c3.metric("風險評分", f"{report['score']} / 100")

    st.markdown("---")
    
    # 指標
    st.subheader("📊 關鍵指標體檢")
    k1, k2, k3, k4 = st.columns(4)
    
    bias_val = df['Bias'].iloc[-1]
    k1.metric("乖離率", f"{bias_val:.1f}%")
    
    div_yield = info.get('dividendYield', 0)
    div_display = f"{div_yield*100:.2f}%" if div_yield else "無"
    k2.metric("現金殖利率", div_display)
    
    k3.metric("OBV 動向", report['obv_trend'])
    k4.metric("ATR 安全線", f"{report['atr_stop_price']:.2f}")

    with st.container():
        st.write("🔎 **進階查詢**")
        yahoo_link = f"https://tw.stock.yahoo.com/quote/{ticker_input}/institutional-trading"
        st.link_button("查看外資買賣超 (Yahoo)", yahoo_link)

    st.markdown("---")

    # 報告
    st.subheader("📋 AI 分析報告")
    if report['score'] >= 80: st.markdown(f"<div class='status-danger'>🛑 危險 (賣出/減碼)</div>", unsafe_allow_html=True)
    elif report['score'] <= 30: st.markdown(f"<div class='status-safe'>✅ 安全 ({report['action']})</div>", unsafe_allow_html=True)
    else: st.markdown(f"<div class='status-neutral'>⚠️ 中性觀察</div>", unsafe_allow_html=True)
    
    st.write("")
    if not report['details']: st.info("走勢正常。")
    for title, explanation in report['details']:
        with st.container():
            st.markdown(f"**{title}**")
            st.markdown(f"<div class='explanation-text'>💡 {explanation}</div>", unsafe_allow_html=True)
            st.divider()

    if pl_amount < 0:
        deposit_rate = 0.017
        total_cost = buy_price * shares_held
        loss_years = abs(pl_amount) / (total_cost * deposit_rate) if total_cost > 0 else 0
        st.error(f"💸 **現實換算**：賠掉了 **{loss_years:.1f} 年** 的定存利息。")

    # 圖表
    st.markdown("### 📈 全方位分析圖")
    tab1, tab2, tab3 = st.tabs(["價量走勢", "OBV 能量", "📅 月份慣性"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='股價', increasing_line_color='#EF5350', decreasing_line_color='#26A69A'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#FFA726', width=2), name='月線'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close']-(2*df['ATR']), line=dict(color='red', width=2, dash='dot'), name='安全底線'))
        fig.add_hline(y=buy_price, line_dash="dash", line_color="blue", annotation_text="成本")
        if use_trailing and report['trailing_stop_price'] > 0:
             fig.add_hline(y=report['trailing_stop_price'], line_color="purple", line_width=3, annotation_text="移動停利")
        fig.update_layout(xaxis_rangeslider_visible=False, height=600, margin=dict(t=30, b=20), legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        fig_obv = make_subplots(specs=[[{"secondary_y": True}]])
        fig_obv.add_trace(go.Scatter(x=df.index, y=df['Close'], name="股價", line=dict(color="gray", width=1)), secondary_y=True)
        fig_obv.add_trace(go.Scatter(x=df.index, y=df['OBV'], name="OBV", line=dict(color="blue", width=2)), secondary_y=False)
        fig_obv.update_layout(height=500, legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"))
        st.plotly_chart(fig_obv, use_container_width=True)
        
    with tab3:
        season_stats, win_rate = calculate_seasonality(df)
        fig_season = go.Figure()
        colors = ['#EF5350' if x > 0 else '#26A69A' for x in season_stats.values]
        fig_season.add_trace(go.Bar(x=season_stats.index, y=season_stats.values, marker_color=colors, name='平均漲跌幅'))
        fig_season.add_trace(go.Scatter(x=win_rate.index, y=win_rate.values, name='上漲機率', yaxis='y2', line=dict(color='blue', width=2, dash='dot')))
        fig_season.update_layout(xaxis=dict(title="月份", tickmode='linear', tick0=1, dtick=1), yaxis2=dict(title="勝率 %", overlaying='y', side='right', range=[0, 100]), height=500, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_season, use_container_width=True)

# ---------------------------------------------------------
# 4. 頁面 B: 媽媽專用說明書 (Instruction Manual)
# ---------------------------------------------------------
def instruction_page():
    st.title("📖 媽媽的股票操作說明書")
    st.markdown("### 歡迎使用！請把這裡當作您的「投資字典」。")
    st.info("💡 提示：下方有底線的藍色文字，滑鼠移上去(不要點) 稍微停一下，就會出現解釋喔！")
    
    st.divider()
    
    st.header("1. 系統是做什麼的？")
    st.markdown(f"""
    這套系統就像是您開車時的 **{tooltip('安全氣囊', '當發生意外時，保護您不要受重傷')}** 與 **{tooltip('倒車雷達', '偵測後方有無障礙物，預防撞擊')}**。
    
    * 它**不能**保證您買在最低點。
    * 它**可以**保證當危險發生時，第一時間叫您跑，保護您的退休金。
    """)
    
    st.header("2. 名詞解釋 (滑鼠移到藍字上看解釋)")
    
    st.subheader("💰 判斷貴不貴")
    st.markdown(f"""
    * **{tooltip('P/B (股價淨值比)', '就像去百貨公司買衣服。數值 0.8 代表衣服打 8 折，比成本還便宜；數值 2.0 代表賣兩倍價錢，很貴。')}**：
        * 如果是 **{tooltip('景氣循環股', '例如航運、記憶體、鋼鐵。賺錢時大賺，賠錢時大賠的股票。')}** (如南亞科、長榮)，看到 P/B < 1.0 代表很便宜，可以買。
    * **{tooltip('現金殖利率', '假設股價都不漲，光靠公司發的利息，每年可以拿多少 %。')}**：
        * 就像銀行定存利息。如果有 5% 以上，就算被套牢也比較安心。
    """)
    
    st.subheader("🚀 判斷會不會漲")
    st.markdown(f"""
    * **{tooltip('OBV (能量潮)', '主力的測謊機。如果股價沒漲，但這條線一直往上爬，代表主力大戶正在偷偷買進。')}**：
        * 這是最好的進場訊號，代表有人在吃貨。
    * **{tooltip('乖離率', '像溜狗的繩子。如果股價衝太快(乖離太大)，繩子會把狗拉回來，代表漲太多了，不要追高。')}**：
        * 如果數值超過 10%，千萬不要買，很容易買在最高點。
    """)
    
    st.subheader("🛡️ 判斷什麼時候跑")
    st.markdown(f"""
    * **{tooltip('ATR 安全線', '電腦算出的「最後防線」。如果收盤價跌破這個價格，代表趨勢壞了，一定要跑。')}**：
        * 不要心存僥倖，跌破就是賣。
    * **{tooltip('移動停利', '一種鎖住獲利的策略。當股價從最高點掉下來 10%，就強制獲利了結。')}**：
        * 這是為了防止「賺 20 萬變賠錢」的慘劇。開啟後，系統會幫您顧好錢包。
    """)
    
    st.divider()
    
    st.header("3. 紅綠燈號怎麼看？")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.error("🛑 紅色：危險")
        st.markdown("主力在賣、跌破支撐。**請賣出或減碼**，不要加碼。")
    with c2:
        st.success("✅ 綠色：安全")
        st.markdown("價值浮現、主力在買。**可以分批買進**。")
    with c3:
        st.warning("⚠️ 橘色：觀望")
        st.markdown("方向不明確。**多看少做**，不要亂動。")

# ---------------------------------------------------------
# 5. 主程式 (導航控制)
# ---------------------------------------------------------
def main():
    st.sidebar.title("導覽選單")
    page = st.sidebar.radio("請選擇頁面", ["📊 股票分析儀表板", "📖 媽媽專用說明書"])
    st.sidebar.divider()
    
    if page == "📊 股票分析儀表板":
        dashboard_page()
    else:
        instruction_page()

if __name__ == "__main__":
    main()
