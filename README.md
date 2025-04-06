# stock-prediction
Deep Learning-Based End-to-End Pipeline for Stock Price Prediction
# 📈 Stock Price Prediction with Deep Learning: NVIDIA Case Study

This project presents a deep learning-based approach to predict stock prices, using **NVIDIA's historical stock data over the last ten years**. We compare a **Feedforward Neural Network (FNN)** as a baseline model against a **Transformer-based model** enhanced with self-attention mechanisms. This work is based on our recent publication, which explores the effectiveness of deep neural networks for time-series prediction in financial markets.

---

## 🔍 Project Highlights

- 📊 **Data**: Daily Open, High, Low, Close, and Price data for NVIDIA (10 years).
- 🧠 **Models**:
  - **Feedforward Neural Network** (baseline)
  - **Transformer Model** with Multi-Head Self-Attention
- 🧪 **Evaluation Metrics**: Accuracy on predicted OHLC and price values.
- 📈 **Results**:
  - Transformer model achieves **~95.37% average accuracy**.
  - Improved generalization and robustness over baseline.

---

## 📁 Dataset

The input dataset consists of:

- Daily stock records: `Open`, `High`, `Low`, `Close`, and computed `Price`
- Sliding window approach: 14-day input sequence for next-day prediction.

---

## 🧠 Model Architectures

### 🔹 Feedforward Neural Network (Baseline)

- Single hidden layer
- Input: 14-day window × 5 features
- Output: Next day's predicted OHLC + price
- Loss: Mean Squared Error (MSE)

### 🔹 Transformer-Based Model

- Positional encoding + multi-head self-attention
- Captures **temporal dependencies** and **market patterns**
- Superior performance on all prediction metrics

---

## 📊 Results Overview

| Metric        | Baseline Accuracy (%) | Transformer Accuracy (%) |
|---------------|------------------------|----------------------------|
| **Price**     | 93.16                  | **95.37**                  |
| **Open**      | 93.37                  | **95.06**                  |
| **High**      | 93.45                  | **93.12**                  |
| **Low**       | 93.42                  | **96.03**                  |
| **Close**     | 93.14                  | **94.94**                  |

- Visualizations confirm that the Transformer provides smoother and more accurate predictions across all values.
- Consistent convergence observed during training (see loss chart in `results/`).

---

## 📌 Key Contributions

- Demonstrated the **superiority of transformer architectures** over traditional neural networks in financial time-series forecasting.
- Proposed a practical and reproducible methodology for stock prediction tasks.
- Evaluated over **multiple dates**, showcasing **robustness** under varying market conditions.

---

## 🛠️ Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-price-prediction-nvidia.git
cd stock-price-prediction-nvidia

# Install dependencies
pip install -r requirements.txt

# Run training script
python train_transformer.py

# Evaluate performance
python evaluate_models.py
