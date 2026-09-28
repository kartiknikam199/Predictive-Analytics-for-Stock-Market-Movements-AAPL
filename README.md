# AAPL Stock Direction Prediction

A machine learning project to predict whether **Apple (AAPL)** stock is likely to move **up or down on the following trading day**, using historical price, volume, and technical indicators.

The main goal of this project was not just to build a machine learning model, but to understand **how useful historical market data and technical indicators actually are for short-term prediction**.

> **Key result:** The Random Forest model achieved **49.8% accuracy** on the test set, showing that the selected technical indicators did not provide a strong predictive signal for next-day stock direction.

---
## Project Overview

Predicting stock prices is a challenging machine learning problem because financial markets are affected by many factors, including company performance, news, investor sentiment, economic conditions, and market events.

For this project, I focused on a simpler problem:

**Can historical AAPL price and volume data, combined with common technical indicators, predict the direction of the next trading day's closing price?**

The project follows a complete data analytics and machine learning workflow:

**Data → Cleaning → EDA → Feature Engineering → Modeling → Evaluation → Insights**

---

## Dataset

The dataset contains historical stock market data for Apple Inc. (AAPL).

| Detail            | Information                                     |
| ----------------- | ----------------------------------------------- |
| Stock             | Apple Inc. (AAPL)                               |
| Data Source       | Yahoo Finance via Kaggle                        |
| Original Period   | Dec 1980 – Apr 2020                             |
| Analysis Period   | Jan 2010 – Apr 2020                             |
| Trading Days Used | 2,579                                           |
| Features          | Date, Open, High, Low, Close, Adj Close, Volume |

The original dataset was filtered to **2010 onwards** to focus the analysis on a more recent market period.

**Data quality checks:**

* No missing values
* No duplicate records
* No invalid or non-positive price values
* Dates converted to datetime and sorted chronologically

Dataset source: [Kaggle — Price-Volume Data for All US Stocks & ETFs](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)

---

## Technologies Used

* **Python**
* **Pandas** — Data manipulation and analysis
* **NumPy** — Numerical operations
* **Scikit-learn** — Machine learning and evaluation
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualizations
* **Jupyter Notebook** — Analysis and experimentation

---

## Project Workflow

### 1. Data Preparation

The AAPL dataset was loaded into Python and prepared for analysis.

Key steps included:

* Loading the CSV dataset
* Converting the `Date` column to datetime
* Sorting records chronologically
* Filtering the data from 2010 onwards
* Checking missing values and duplicates
* Validating price and volume values

### 2. Exploratory Data Analysis

I explored the historical behavior of AAPL using:

* Closing price trend
* Daily return distribution
* 30-day rolling volatility
* OHLCV correlation analysis

These visualizations helped understand the underlying characteristics of the stock before building the model.

### 3. Feature Engineering

Several technical indicators and historical features were created:

| Feature            | Description                    |
| ------------------ | ------------------------------ |
| MA 7               | 7-day moving average           |
| MA 21              | 21-day moving average          |
| MA 50              | 50-day moving average          |
| RSI                | 14-day Relative Strength Index |
| Rolling Volatility | 7-day rolling volatility       |
| Lagged Close       | Previous closing prices        |
| Lagged Volume      | Previous trading volume        |

These features were used as inputs for the classification model.

### 4. Target Creation

The target variable represents the **next trading day's price direction**.

```text
Target = 1 → Next day's Close > Today's Close
Target = 0 → Next day's Close ≤ Today's Close
```

This converts the problem into a binary classification task.

### 5. Train/Test Split

Because stock data is time-dependent, the dataset was split chronologically rather than randomly.

* **80%** → Training data
* **20%** → Test data
* **No shuffling**

This helps prevent future information from being used during model training.

### 6. Model

I used a **Random Forest Classifier** with:

```text
n_estimators = 100
max_depth = 10
```

Random Forest was selected because it can capture non-linear relationships between multiple technical indicators without requiring strong assumptions about the data.

---

# Results

The model was evaluated on **506 trading days**.

| Metric           |    Result |
| ---------------- | --------: |
| Accuracy         | **48.8%** |
| Precision — Down |      0.47 |
| Recall — Down    |      0.90 |
| Precision — Up   |      0.63 |
| Recall — Up      |      0.14 |

### Confusion Matrix

| Actual / Predicted | Down | Up |
| ------------------ | ---: | -: |
| **Down**           |  209 | 22 |
| **Up**             |  237 | 38 |

The results show that the model predicted the **Down** class much more frequently than the **Up** class.

Although the model achieved relatively high recall for downward movements, it identified only a small proportion of actual upward movements.

---

## What I Learned From the Results

One of the most useful outcomes of this project was that the model **did not produce a strong predictive signal**.

An accuracy of **48.8%** is close to random guessing for a balanced binary direction problem. More importantly, the confusion matrix shows that overall accuracy alone does not tell the full story.

The model had:

* High recall for the Down class
* Very low recall for the Up class
* A noticeable prediction bias toward Down
* Limited ability to distinguish the two directions consistently

This was an important learning point: **building a machine learning model does not necessarily mean finding a useful prediction strategy.**

The experiment suggests that price, volume, and the selected technical indicators were not sufficient to reliably predict AAPL's next-day direction during this period.

---

## Feature Importance

The Random Forest feature importance analysis indicated that features such as **RSI and short-term moving averages** were among the more influential variables used by the model.

However, feature importance should not be interpreted as proof that a feature can independently predict future prices.

The overall model performance remained weak despite using these indicators.

---

## Key Takeaways

### 1. Next-day stock direction is difficult to predict

Short-term market movements contain substantial noise and can be influenced by information that is not present in historical OHLCV data.

### 2. Accuracy is not enough

Looking only at the 48.8% accuracy would hide the model's strong class bias. Precision, recall, and the confusion matrix provided a much clearer picture of model behavior.

### 3. Technical indicators have limitations

Moving averages, RSI, volatility, and lagged features can describe historical market behavior, but they did not provide enough information to reliably predict the next day's direction in this experiment.

### 4. A failed prediction model can still be a successful analytics project

The project helped demonstrate the complete machine learning workflow while also showing the importance of **testing assumptions rather than assuming a model will work**.

---

# Limitations

This project has several limitations:

* Only price and volume-based information was used.
* News and market sentiment were not included.
* Company fundamentals were not considered.
* Macroeconomic variables were not included.
* The analysis covers a specific historical period.
* Next-day direction is a particularly noisy prediction target.
* No live trading or transaction-cost analysis was performed.

Most importantly, historical model performance does not guarantee future performance.

---

# Future Improvements

There are several directions that could make the analysis more comprehensive.

### Market & External Data

Add features such as:

* News sentiment
* Earnings announcements
* Market indices such as S&P 500
* Interest rates
* Economic indicators
* Sector performance

### Alternative Targets

Instead of predicting the next day's direction, experiment with:

* 5-day direction
* 20-day direction
* Future return
* Volatility prediction

### Model Improvements

Compare multiple approaches:

* Logistic Regression
* XGBoost
* Gradient Boosting
* Support Vector Machines
* LSTM / other time-series models

Hyperparameter tuning and proper time-series cross-validation could also be explored.

### Better Baselines

The model should also be compared with simple strategies such as:

* Always predicting the majority class
* Predicting tomorrow's direction based on today's direction
* Buy-and-hold performance

This would provide better context for evaluating whether the machine learning model adds value.

---

# Project Structure

```text
AAPL-Stock-Direction-Prediction/
│
├── README.md
│
├── dataset/
│   └── AAPL.csv
│
├── notebook/
│   ├── stock_prediction.ipynb
│   └── stock_prediction.html
│
└── charts/
    ├── price_trend.png
    ├── returns_distribution.png
    ├── volatility_chart.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── confusion_matrix.png
    └── prediction_vs_actual.png
```

---

# Visualizations

The project includes visualizations covering:

* AAPL historical price trend
* Daily return distribution
* Rolling volatility
* OHLCV correlation
* Feature importance
* Confusion matrix
* Actual vs predicted direction

---

# Conclusion

This project explored whether historical AAPL price and volume information could be used to predict next-day stock direction with a Random Forest classifier.

The model achieved **49.8% accuracy**, but the detailed evaluation showed a strong bias toward predicting downward movements and poor recall for upward movements.

Rather than treating this as simply a low-accuracy model, the project highlights an important part of practical data science:

> **A good machine learning workflow is not only about building models — it is also about understanding when the data does not contain enough information to support the prediction task.**

This project strengthened my understanding of:

* Data cleaning
* Exploratory data analysis
* Feature engineering
* Time-series data handling
* Classification modeling
* Model evaluation
* Data visualization
* Interpreting machine learning results

---

## Disclaimer

This project is intended for **educational and portfolio purposes only**.

It is not financial or investment advice. The model should not be used to make real-world trading or investment decisions.

---

## Author

**Nemuri Sathwik Goud**
Data Analytics Intern — Techn Global
