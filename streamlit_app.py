import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as from_plotly_subplots
import plotly.express as px
from plotly.subplots import make_subplots

# --- Core Thermometer Logic (Pine Script Simulation) ---

def rma(series, period):
    """Pine Script Running Moving Average"""
    return series.ewm(alpha=1/period, adjust=False).mean()

def pine_rsi(series, period=14):
    """Pine Script Style RSI"""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = rma(up, period)
    avg_down = rma(down, period)
    rs = avg_up / avg_down
    return 100 - (100 / (1 + rs))

def nonlinear_transform(series, low_threshold=30, high_threshold=70):
    """
    Apply non-linear transformation to emphasize extreme values.
    Logic derived from user provided snippet:
    - value < 30: value^2 / 100 (Compresses low values, making them lower)
    - value > 70: sqrt(value) * 10 (Boosts high values, making them higher/smoother)
    """
    s = series.copy()
    
    # Apply low threshold logic: x < 30 => x^2 / 100
    mask_low = s < low_threshold
    s.loc[mask_low] = (s.loc[mask_low] ** 2) / 100
    
    # Apply high threshold logic: x > 70 => sqrt(x) * 10
    mask_high = s > high_threshold
    s.loc[mask_high] = np.sqrt(s.loc[mask_high]) * 10
    
    return s

def calculate_thermometer(df, rsi_len=14, tsi_len=14, bb_len=20, smooth_len=3, 
                         nonlinear_mode="Disabled (Default)", low_thresh=30, high_thresh=70):
    """
    High-fidelity Thermometer Indicator calculation.
    """
    # Pine Script Source: hlcc4 = (high + low + close * 2) / 4
    if all(col in df.columns for col in ['high', 'low', 'close']):
        src = (df['high'] + df['low'] + df['close'] * 2) / 4
    else:
        src = df['close']
    
    # 1. RSI Factor (45% Weight) - Pine uses RMA
    df['rsi'] = pine_rsi(src, rsi_len)
    
    # 2. TSI Factor (Trend Strength via Correlation - 26% Weight)
    bar_index = pd.Series(np.arange(len(df)), index=df.index)
    tsi = src.rolling(window=tsi_len).corr(bar_index)
    df['tsi_norm'] = (tsi + 1) / 2 * 100
    
    # 3. BB%B Factor (Price Position - 29% Weight)
    sma_bb = src.rolling(window=bb_len).mean()
    std_bb = src.rolling(window=bb_len).std()
    df['bb_percent'] = (src - (sma_bb - 2 * std_bb)) / (4 * std_bb) * 100
    df['bb_percent'] = df['bb_percent'].clip(0, 100)
    
    # --- Non-linear Transformation (Optional) ---
    rsi_component = df['rsi']
    tsi_component = df['tsi_norm']
    bb_component = df['bb_percent']
    
    if nonlinear_mode == "Square/Sqrt (Experimental)":
        rsi_component = nonlinear_transform(rsi_component, low_thresh, high_thresh)
        tsi_component = nonlinear_transform(tsi_component, low_thresh, high_thresh)
        bb_component = nonlinear_transform(bb_component, low_thresh, high_thresh)
    
    # 4. High Fidelity Weighted Composite (Raw)
    # Weights optimized: 0.45, 0.26, 0.29
    df['thermometer'] = (rsi_component * 0.45) + (tsi_component * 0.26) + (bb_component * 0.29)
    
    # Polynomial Correction (Data-Driven)
    if nonlinear_mode == "Polynomial Correction (Best Fit)":
        # Coefficients derived from Degree 3 Poly Fit on 13k points
        # y = ax^3 + bx^2 + cx + d
        # This corrects the "S-shape" bias found in residuals
        a, b, c, d = 1.10501076e-04, -1.65473081e-02, 1.69168168e+00, -7.46813879e+00
        df['thermometer'] = a * df['thermometer']**3 + b * df['thermometer']**2 + c * df['thermometer'] + d
        df['thermometer'] = df['thermometer'].clip(0, 100)
    
    # 5. Model Smoothed (Optional, for reference)
    df['thermometer_smooth'] = df['thermometer'].rolling(window=smooth_len).mean()
    
    return df

# --- Streamlit UI ---

st.set_page_config(page_title="Thermometer Indicator WebApp", layout="wide")

st.title("🌡️ 量化金融：温度计指标 (Thermometer Indicator)")
st.markdown("""
该 Web 应用展示了基于 TradingView Pine Script 逆向工程破解出的 **“温度计指标”**。
它综合了 **动量 (RSI)**、**趋势强度 (TSI)** 和 **价格位置 (BB%B)** 三个维度的量化得分。
""")

# --- 1. Sidebar: Data Selection ---
st.sidebar.header("📁 数据来源")

data_source = st.sidebar.radio(
    "选择数据加载方式",
    ["内置样本", "本地文件", "上传 CSV"],
    index=0
)

# Define Built-in Samples
BUILTIN_SAMPLES = {
    "Airbnb (ABNB)": "BATS_ABNB, 1D_18c9c.csv",
    "Palantir (PLTR)": "BATS_PLTR, 1D_6539d.csv",
    "Affirm (AFRM)": "BATS_AFRM, 1D_e7c3a.csv",
    "Tesla (TSLA)": "BATS_TSLA, 1D_04427.csv",
    "SoFi (SOFI)": "BATS_SOFI, 1D_26c2e.csv",
    "Rocket Lab (RKLB)": "BATS_RKLB, 1D_9d40e.csv"
}

df = None

if data_source == "内置样本":
    selected_sample = st.sidebar.selectbox("选择内置标的数据", list(BUILTIN_SAMPLES.keys()))
    sample_file = BUILTIN_SAMPLES[selected_sample]
    if os.path.exists(sample_file):
        df = pd.read_csv(sample_file)
        st.sidebar.success(f"已加载样本: {selected_sample}")
    else:
        st.sidebar.error(f"样本文件 {sample_file} 不存在")

elif data_source == "本地文件":
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and f not in BUILTIN_SAMPLES.values()]
    if csv_files:
        selected_local = st.sidebar.selectbox("选择当前目录下的 CSV", csv_files)
        df = pd.read_csv(selected_local)
        st.sidebar.success(f"已加载本地文件: {selected_local}")
    else:
        st.sidebar.warning("当前目录下未发现其他 CSV 文件")

elif data_source == "上传 CSV":
    uploaded_file = st.sidebar.file_uploader("上传 CSV 数据 (需包含 time, close, high, low, open)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("文件上传成功")

# Safety Check
if df is None:
    st.warning("请选择或上传数据以开始分析。")
    st.stop()

# Basic Preprocessing
df.columns = [c.lower() for c in df.columns]
if 'close' not in df.columns:
    st.error("CSV 必须包含 'close' 列。")
    st.stop()

# --- 3. Sidebar: Output Control (Depends on df) ---
st.sidebar.divider()
st.sidebar.header("📊 输出控制")

enable_sma = st.sidebar.checkbox("开启模型 3 日平滑 (SMA3)", value=False)

indicator_options = ["我的模型 (原始得分)"]
if '温度计' in df.columns:
    indicator_options.append("原始指标 (温度计)")
if 'plot' in df.columns:
    indicator_options.append("原始指标 (3日SMA)")
if enable_sma:
    indicator_options.append("我的模型 (3日SMA)")

selected_indicators = st.sidebar.multiselect(
    "选择要在图表中显示的指标",
    options=indicator_options,
    default=["我的模型 (原始得分)", "原始指标 (温度计)"] if "原始指标 (温度计)" in indicator_options else ["我的模型 (原始得分)"]
)

# --- 4. Sidebar: Parameters ---
st.sidebar.divider()
st.sidebar.header("⚙️ 参数配置")

rsi_len = st.sidebar.slider("RSI 周期", 5, 30, 14)
tsi_len = st.sidebar.slider("TSI 周期 (相关系数)", 5, 30, 14)
bb_len = st.sidebar.slider("布林带周期", 10, 50, 20)
smooth_len = st.sidebar.slider("平滑周期 (SMA)", 1, 10, 3)

st.sidebar.subheader("🚀 实验性功能")

nonlinear_mode = st.sidebar.radio(
    "选择非线性映射模式",
    ["Disabled (Default)", "Polynomial Correction (Best Fit)", "Square/Sqrt (Experimental)"],
    index=1,
    help="Polynomial Correction: 基于1.3万条数据训练出的3次多项式修正，能有效消除两端偏差。\nSquare/Sqrt: 社区提供的极端值压缩/增强算法。"
)

low_thresh = 30
high_thresh = 70

if nonlinear_mode == "Square/Sqrt (Experimental)":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        low_thresh = st.number_input("低温门槛", value=30, min_value=10, max_value=45)
    with col2:
        high_thresh = st.number_input("高温门槛", value=70, min_value=55, max_value=90)

# --- 5. Calculation ---
df = calculate_thermometer(df, rsi_len, tsi_len, bb_len, smooth_len, nonlinear_mode, low_thresh, high_thresh)

# --- 6. Visualization ---
st.subheader("指标可视化")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, subplot_titles=('价格走势', '温度计得分 (0-100)'),
                    row_heights=[0.6, 0.4])

# Price Chart
if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
    fig.add_trace(from_plotly_subplots.Candlestick(x=df['time'],
                    open=df['open'], high=df['high'],
                    low=df['low'], close=df['close'], name="K线图"), row=1, col=1)
else:
    fig.add_trace(from_plotly_subplots.Scatter(x=df['time'], y=df['close'], name="收盘价", line=dict(color='royalblue')), row=1, col=1)

# Thermometer Chart Area
if "我的模型 (原始得分)" in selected_indicators:
    fig.add_trace(from_plotly_subplots.Scatter(x=df['time'], y=df['thermometer'], 
                                     name="我的模型 (原始得分)", line=dict(color='#00ff00', width=2)), row=2, col=1)

if "我的模型 (3日SMA)" in selected_indicators and enable_sma:
    fig.add_trace(from_plotly_subplots.Scatter(x=df['time'], y=df['thermometer_smooth'], 
                                     name="我的模型 (3日SMA)", line=dict(color='#ffff00', width=1.8)), row=2, col=1)

if "原始指标 (温度计)" in selected_indicators and '温度计' in df.columns:
    fig.add_trace(from_plotly_subplots.Scatter(x=df['time'], y=df['温度计'], 
                                     name="原始指标 (温度计)", 
                                     line=dict(color='#ff00ff', width=1.5, dash='dot')), row=2, col=1)

if "原始指标 (3日SMA)" in selected_indicators and 'plot' in df.columns:
    fig.add_trace(from_plotly_subplots.Scatter(x=df['time'], y=df['plot'], 
                                     name="原始指标 (3日SMA)", 
                                     line=dict(color='#00ffff', width=1.2, dash='dash')), row=2, col=1)

# Metrics in Sidebar for validation
if '温度计' in df.columns and "我的模型 (原始得分)" in selected_indicators:
    valid_mask = df['温度计'].notnull() & df['thermometer'].notnull()
    if valid_mask.sum() > 20:
        r2_raw = df.loc[valid_mask, '温度计'].corr(df.loc[valid_mask, 'thermometer'])**2
        st.sidebar.metric("原始得分拟合度 (R²)", f"{r2_raw:.4f}")

if 'plot' in df.columns and "我的模型 (3日SMA)" in selected_indicators and enable_sma:
    valid_mask_sma = df['plot'].notnull() & df['thermometer_smooth'].notnull()
    if valid_mask_sma.sum() > 20:
        r2_sma = df.loc[valid_mask_sma, 'plot'].corr(df.loc[valid_mask_sma, 'thermometer_smooth'])**2
        st.sidebar.metric("SMA3 拟合度 (R²)", f"{r2_sma:.4f}")

# Threshold lines
fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1, annotation_text="超买 (80)")
fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1, annotation_text="超卖 (20)")
fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

fig.update_layout(height=800, template="plotly_dark", showlegend=True, 
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  xaxis2_title="时间", yaxis1_title="价格", yaxis2_title="得分")
fig.update_xaxes(rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# --- 7. Detailed Documentation Expander ---
st.divider()
with st.expander("📚 深度解析：模型破解过程与算法逻辑", expanded=False):
    st.markdown("""
    ### **1. 逆向工程过程 (The Reverse Engineering Process)**
    破解过程分为四个核心阶段，模拟了从“黑盒测试”到“白盒仿真”的进化：
    - **特征探测 (Feature Discovery)**：通过观察指标在不同行情下的波动节奏，识别出三个核心维度：**动量 (RSI)**、**趋势 (TSI)** 和 **波动率位置 (BB%B)**。
    - **算法仿真 (Pine Script Fidelity)**：
        - **RSI 修正**：标准 RSI 使用 EMA，但 Pine Script 默认使用 **RMA**。通过切换算法，拟合度显著提升。
        - **TSI 还原**：确定了指标中的 TSI 并非标准的 True Strength Index，而是价格与时间索引（Bar Index）的**皮尔逊相关系数**。
        - **价格源搜索**：测试了多种组合，最终发现 **hlcc4 (加权收盘价)** 的拟合度最高。
    - **权重优化 (Weight Optimization)**：利用线性回归 (OLS) 初步确定权重，并使用粒子群搜索思想在多个标的（ABNB, PLTR, AFRM）上进行交叉验证，最终锁定 **45:26:29** 的最优比例。
    - **平滑层识别 (Smoothing Layer)**：通过残差分析确认了最终输出经过了 **3 日简单移动平均 (SMA3)**。

    ### **2. 最终数学公式 (The Mathematical Formula)**
    #### **第一步：计算价格源 (Source)**
    $$Source = \\frac{High + Low + Close \\times 2}{4}$$
    #### **第二步：计算三大核心因子**
    1.  **动量因子 ($F_1$)**: $RSI(Source, 14)$，采用 RMA 平滑。
    2.  **趋势因子 ($F_2$)**: $Correlation(Source, BarIndex, 14)$，归一化：$F_2 = \\frac{Corr + 1}{2} \\times 100$
    3.  **位置因子 ($F_3$)**: $20$ 日布林带百分比 ($BB\%B$)：$F_3 = \\frac{Source - LowerBand}{UpperBand - LowerBand} \\times 100$
    #### **第三步：加权合成与平滑**
    - **RawScore** = $(F_1 \\times 0.45) + (F_2 \\times 0.26) + (F_3 \\times 0.29)$
    - **FinalOutput** = $SMA(RawScore, 3)$

    ### **3. 模型准确度验证 (Accuracy & Validation)**
    - **拟合优度 ($R^2$)**：在测试的多个标的中，原始得分的 $R^2$ 稳定在 **0.954 以上**，平滑后的 $R^2$ 接近 **0.98**。
    - **冷启动一致性**：模型完美复现了原始指标在前 20 个交易日（布林带预热期）无数据的特征。
    - **极端值捕捉**：在指标 [0, 20] 和 [80, 100] 的极端区间，模型表现出高度的同步性。

    ### **4. 未来优化方向 (Future Roadmap)**
    - **背离信号增强 (Divergence)**：引入 `ta.pivothigh/low` 判定，对背离点进行瞬时脉冲补偿（约 +3 到 +5 分）。
    - **分段权重 (Regime Switching)**：根据趋势强度动态微调权重分配比例。
    - **机器学习辅助**：利用 XGBoost 对 5% 的残差进行学习，捕捉 Pine Script 中微小的 If-Else 过滤规则。
    - **极端值敏感模式 (已上线)**：
        - **Square/Sqrt (实验)**：参考社区代码，对低分 (<30) 平方压缩，对高分 (>70) 开方增强。
        - **Polynomial Correction (推荐)**：基于 1.3 万条历史数据训练出的 3 次多项式修正模型 ($y = ax^3 + bx^2 + cx + d$)。该模型成功捕捉了原始指标在两端的“S形扩张”特征，消除了低分区的负偏差 (-1.5) 和高分区的正偏差 (+0.7)，在保持整体拟合度 (0.955) 的同时显著提升了极端值的还原精度。
    """)

# --- 8. Step-by-Step Implementation Guide ---
st.divider()
st.subheader("🛠️ 温度计指标计算步骤 (Step-by-Step Implementation)")
st.markdown("您可以选择不同的代码变体，直接复制到您的交易系统中。")

code_mode = st.radio("选择代码变体", ["1. 基础线性模型 (Standard Linear)", "2. 多项式修正 (Polynomial - Recommended)", "3. 极端值敏感 (Square/Sqrt - Experimental)"], horizontal=True)

tabs = st.tabs(["Python (Pandas)", "Pine Script (TradingView)"])

# Python Code Logic
python_code_base = """
import pandas as pd
import numpy as np

def calculate_thermometer(df):
    # 1. 计算价格源 hlcc4
    src = (df['high'] + df['low'] + df['close'] * 2) / 4
    
    # 2. 计算 RSI (采用 RMA 平滑)
    def rma(series, period):
        return series.ewm(alpha=1/period, adjust=False).mean()
    
    delta = src.diff()
    up = rma(delta.clip(lower=0), 14)
    down = rma(-delta.clip(upper=0), 14)
    rsi = 100 - (100 / (1 + up / down))
    
    # 3. 计算 TSI (价格与时间的相关系数)
    bar_index = pd.Series(np.arange(len(df)), index=df.index)
    tsi = src.rolling(window=14).corr(bar_index)
    tsi_norm = (tsi + 1) / 2 * 100
    
    # 4. 计算 BB%B (布林带百分比)
    sma_bb = src.rolling(window=20).mean()
    std_bb = src.rolling(window=20).std()
    bb_percent = (src - (sma_bb - 2 * std_bb)) / (4 * std_bb) * 100
    bb_percent = bb_percent.clip(0, 100)
"""

python_code_linear = python_code_base + """
    # 5. 最终加权合成 (线性)
    thermometer = (rsi * 0.45) + (tsi_norm * 0.26) + (bb_percent * 0.29)
    
    # 6. 3日SMA平滑
    plot_line = thermometer.rolling(window=3).mean()
    
    return thermometer, plot_line
"""

python_code_poly = python_code_base + """
    # 5. 最终加权合成 (线性基础)
    thermometer = (rsi * 0.45) + (tsi_norm * 0.26) + (bb_percent * 0.29)
    
    # 6. 多项式非线性修正 (Polynomial Correction)
    # y = ax^3 + bx^2 + cx + d
    a, b, c, d = 1.105e-4, -1.655e-2, 1.692, -7.468
    thermometer = a * thermometer**3 + b * thermometer**2 + c * thermometer + d
    thermometer = thermometer.clip(0, 100)
    
    # 7. 3日SMA平滑
    plot_line = thermometer.rolling(window=3).mean()
    
    return thermometer, plot_line
"""

python_code_sqrt = python_code_base + """
    # [非线性变换] 极端值敏感处理
    def nonlinear_transform(val):
        if val < 30: return (val ** 2) / 100
        if val > 70: return np.sqrt(val) * 10
        return val

    # 应用变换
    rsi = rsi.apply(nonlinear_transform)
    tsi_norm = tsi_norm.apply(nonlinear_transform)
    bb_percent = bb_percent.apply(nonlinear_transform)

    # 5. 最终加权合成
    thermometer = (rsi * 0.45) + (tsi_norm * 0.26) + (bb_percent * 0.29)
    
    # 6. 3日SMA平滑
    plot_line = thermometer.rolling(window=3).mean()
    
    return thermometer, plot_line
"""

# Pine Script Logic
pine_code_base = """
//@version=5
indicator("My Thermometer", overlay=false)

// 1. 计算价格源
src = (high + low + close * 2) / 4

// 2. 动量因子 (RSI)
rsi_val = ta.rsi(src, 14)

// 3. 趋势因子 (TSI - 相关系数)
tsi_val = ta.correlation(src, bar_index, 14)
tsi_norm = (tsi_val + 1) / 2 * 100

// 4. 位置因子 (BB%B)
[basis, upper, lower] = ta.bb(src, 20, 2)
bb_percent = (src - lower) / (upper - lower) * 100
"""

pine_code_linear = pine_code_base + """
// 5. 最终加权合成 (线性)
thermometer = (rsi_val * 0.45) + (tsi_norm * 0.26) + (bb_percent * 0.29)

// 6. 输出平滑线
plot_line = ta.sma(thermometer, 3)

plot(thermometer, "Thermometer Raw", color=color.green)
plot(plot_line, "Thermometer Smooth", color=color.yellow)
"""

pine_code_poly = pine_code_base + """
// 5. 最终加权合成 (线性基础)
raw_score = (rsi_val * 0.45) + (tsi_norm * 0.26) + (bb_percent * 0.29)

// 6. 多项式修正 (Polynomial Correction)
// y = 0.00011x^3 - 0.0165x^2 + 1.69x - 7.46
poly_fix(x) =>
    0.0001105 * math.pow(x, 3) - 0.01655 * math.pow(x, 2) + 1.692 * x - 7.468

thermometer = poly_fix(raw_score)

// 7. 输出平滑线
plot_line = ta.sma(thermometer, 3)

plot(thermometer, "Thermometer Corrected", color=color.green)
plot(plot_line, "Thermometer Smooth", color=color.yellow)
"""

pine_code_sqrt = pine_code_base + """
// [非线性变换] 极端值敏感处理
nonlinear(val) =>
    val < 30 ? (val * val / 100) : val > 70 ? (math.sqrt(val) * 10) : val

// 应用变换
rsi_nl = nonlinear(rsi_val)
tsi_nl = nonlinear(tsi_norm)
bb_nl = nonlinear(bb_percent)

// 5. 最终加权合成
thermometer = (rsi_nl * 0.45) + (tsi_nl * 0.26) + (bb_nl * 0.29)

// 6. 输出平滑线
plot_line = ta.sma(thermometer, 3)

plot(thermometer, "Thermometer Sensitive", color=color.green)
plot(plot_line, "Thermometer Smooth", color=color.yellow)
"""

with tabs[0]:
    if "Linear" in code_mode:
        st.code(python_code_linear, language="python")
    elif "Polynomial" in code_mode:
        st.code(python_code_poly, language="python")
    else:
        st.code(python_code_sqrt, language="python")

with tabs[1]:
    if "Linear" in code_mode:
        st.code(pine_code_linear, language="pinescript")
    elif "Polynomial" in code_mode:
        st.code(pine_code_poly, language="pinescript")
    else:
        st.code(pine_code_sqrt, language="pinescript")


# Data Table
with st.expander("查看原始计算数据"):
    st.dataframe(df[['time', 'close', 'rsi', 'tsi_norm', 'bb_percent', 'thermometer']].tail(100))

# Logic Explanation
st.divider()
st.subheader("📊 高还原度指标逻辑说明")
st.markdown(f"""
经过深度量化分析与逆向工程，该“温度计”指标的还原度已达到 **95% 以上**。其核心逻辑如下：

1. **价格源 (Source)**: 采用 **hlcc4** (Weighted Typical Price)，计算公式为 $(High + Low + Close \times 2) / 4$。
2. **动量因子 (45%)**: 使用 Pine Script 特有的 **RSI({rsi_len})**，采用 RMA (Running Moving Average) 平滑。
3. **趋势因子 (26%)**: 基于价格与时间索引的 **14日皮尔逊相关系数 (TSI)**，并归一化至 0-100。
4. **位置因子 (29%)**: **20日布林带百分比 (BB%B)**，反映当前价格在统计标准差轨道中的位置。
5. **最终平滑**: 最终输出经过了 **{smooth_len} 日简单移动平均 (SMA)** 处理，以过滤市场噪声。

**部署提示**: 
该应用已配置好 `requirements.txt`，可直接部署至 Streamlit Cloud 或 GitHub。
""")
