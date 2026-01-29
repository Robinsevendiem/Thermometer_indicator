# Thermometer Indicator WebApp

This is a Streamlit-based web application for visualizing the "Thermometer Indicator", reverse-engineered from TradingView Pine Script.

## Features
- Upload your own financial CSV data.
- Interactive charts using Plotly.
- Adjustable parameters for RSI, TSI, and Bollinger Bands.
- Detailed logic breakdown of the indicator.

## Deployment
You can deploy this directly to [Streamlit Cloud](https://share.streamlit.io/) by connecting your GitHub repository.

## 如何运行
### 1. 一键启动 (推荐)
直接运行项目根目录下的 `run_app.py`，它会自动启动服务并打开浏览器：
```bash
python run_app.py
```

### 2. 手动启动
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行应用：
   ```bash
   streamlit run streamlit_app.py
   ```

## 主要功能
- **本地文件浏览**：自动识别并列出项目目录下的所有 `.csv` 文件，无需手动上传。
- **高还原度算法**：采用 hlcc4 价格源和优化后的 45/26/29 权重模型。
- **交互式图表**：使用 Plotly 展示价格与指标的联动。
