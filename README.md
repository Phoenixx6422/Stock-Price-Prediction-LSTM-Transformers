# Stock Price Prediction with LSTMs and Transformers

An experimental time-series forecasting project exploring how a recurrent neural network compares with a Transformer encoder when predicting Apple's (AAPL) next-step closing price from recent market data.

I started with an LSTM baseline, then studied *Attention Is All You Need* and built an encoder-only Transformer using positional encoding and multi-head self-attention. The first Transformer attempt performed very poorly; the second version changed the input processing, pooling, and training callbacks and produced much stronger **reported** AAPL metrics.

> **Important:** This is an educational ML experiment, not financial advice or a trading system. Stock prices are noisy and non-stationary, and a high R² on one ticker does not guarantee generalization to other stocks or future periods.

## Results

Metrics below are taken from the saved notebook outputs. MAE and RMSE are in USD.

| Experiment | Model | R² | MAE | RMSE |
|---|---|---:|---:|---:|
| V1 | LSTM | 0.7103 | $11.26 | $14.52 |
| V1 | Transformer encoder | -142.6828 | $322.06 | $323.40 |
| V2 | LSTM | 0.8058 | $9.59 | $11.89 |
| V2 | Transformer encoder | **0.9707** | **$3.42** | **$4.62** |

The V2 Transformer experiment adds a log transform for trading volume, gradient clipping, `GlobalAveragePooling1D`, `EarlyStopping`, and `ReduceLROnPlateau`. See the notebook cells for implementation details.

## Approach

- **Data:** AAPL OHLCV data downloaded with `yfinance` for 2015–2024.
- **Features:** Open, High, Low, Close, and Volume. V2 applies `log1p` to Volume.
- **Sequence length:** 60 trading sessions to predict the next session's closing price.
- **Split:** Chronological 80/20 train/test split.
- **Models:** Two-layer LSTM baseline and custom encoder-only Transformer with positional encoding, multi-head self-attention, residual connections, layer normalization, and a feed-forward network.
- **Evaluation:** R², mean absolute error (MAE), and root mean squared error (RMSE), computed after inverse scaling.

## Project structure

```text
stock-price-transformer/
├── notebooks/
│   ├── 01_lstm_transformer_v1.ipynb
│   └── 02_lstm_transformer_v2.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Run locally

Python 3.10–3.12 is recommended. The notebooks can also be run in Google Colab.

```bash
git clone https://github.com/<YOUR_USERNAME>/stock-price-transformer.git
cd stock-price-transformer
python -m venv .venv
```

Activate the environment, then install dependencies and launch Jupyter:

**Windows (Command Prompt):**
```bat
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

**macOS/Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open either notebook and run the cells in order. Data is fetched from Yahoo Finance when the download cell runs, so network access is required. TensorFlow installation can vary by OS and Python version; Google Colab is a convenient alternative if local setup fails.

## Evaluation caveat / next steps

The reported scores should be treated as **preliminary**. In the current notebooks, the feature scaler is fitted before the chronological train/test split, so information from the test period can influence preprocessing. V1 also uses the test set as validation data for early stopping. These choices can make evaluation optimistic or otherwise unreliable. Before comparing models rigorously:

1. Split the time series first and fit preprocessing only on the training period.
2. Use a separate chronological validation period for early stopping and reserve the test period for final evaluation.
3. Compare against simple baselines (for example, predicting the previous close) and evaluate across multiple tickers and market regimes.
4. Track directional accuracy and risk-aware metrics in addition to regression errors.

## Roadmap

- Make scaling and model selection leakage-safe.
- Test robustly across more tickers and time periods.
- Explore BERT-based financial sentiment features from news or filings.
- Explore retrieval-augmented generation (RAG) to provide relevant textual context, then evaluate whether it adds measurable predictive value.

## Reference

Vaswani et al., *Attention Is All You Need* (2017). This paper motivated the Transformer exploration in this project.
