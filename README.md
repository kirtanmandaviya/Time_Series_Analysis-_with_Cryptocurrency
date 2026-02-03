# 📈 Cryptocurrency Time Series Prediction Project

A complete **end-to-end time series forecasting system for cryptocurrency prices** using multiple models — **ARIMA, Facebook Prophet, and LSTM** — built with a clean modular architecture, configuration-driven pipelines, and a Streamlit UI.

This project follows **production-style ML project structure** with separation of concerns (ingestion, preprocessing, training, prediction, evaluation) and supports saved models for fast inference.

---

## Features

*  **Multi-model forecasting**

  * ARIMA (statistical)
  * Facebook Prophet (trend + seasonality)
  * LSTM (deep learning)
*  **Model comparison & evaluation**
*  **Reusable prediction pipeline**
*  **Highly modular & extensible codebase**
*  **Config-driven workflow (YAML)**
*  **Interactive Streamlit app**
*  **Saved models & scalers for inference**
*  **Clean logging & custom exception handling**

---

##  Project Structure

```
Cryptocurrency_Time_Series_Project
│
├── final_model/                 # Trained & saved models
│   ├── arima.pkl
│   ├── crypto_prophet_model.pkl
│   ├── lstm_model.h5
│   └── lstm_scaler.pkl
│
├── src/
│   ├── components/              # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── lstm_modular.py
│   │   ├── model_evaluate.py
│   │   ├── model_prediction.py
│   │   ├── prophet_model.py
│   │   └── train_arima.py
│   │
│   ├── constants/               # Project constants
│   ├── entity/                  # Config & artifact entities
│   ├── exception/               # Custom exception handling
│   ├── logging/                 # Centralized logging
│   ├── pipeline/                # Prediction pipeline
│   ├── utils/                   # Helper utilities
│   └── __init__.py
│
├── app.py                       # Streamlit application
├── config.yaml                  # Project configuration
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

##  Models Used

### 1️⃣ ARIMA

* Classical statistical time-series model
* Best for short-term linear patterns
* Implemented using **statsmodels / pmdarima**

### 2️⃣ Facebook Prophet

* Handles trend, seasonality & holidays well
* Robust to missing data & outliers

### 3️⃣ LSTM

* Deep learning model for sequential data
* Captures long-term dependencies
* Scaled input + saved scaler for inference

---

##  Configuration

All major parameters (paths, model settings, data configs) are controlled via:

```yaml
config.yaml
```

This makes the project:

* Easy to tune
* Easy to deploy
* Easy to extend

---

##  Environment Setup (Conda)

### 1️⃣ Create Conda Environment

```bash
conda create -p crypto_ts python=3.10 -y
conda activate crypto_ts
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

>  Make sure **TensorFlow ≥ 2.16** is supported on your system.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

This will launch the interactive UI where you can:

* Load data
* Select model
* Generate predictions
* Visualize forecasts

---

## 🔁 Prediction Pipeline Flow

```
Data → Preprocessing → Feature Engineering
     → Model Loading → Prediction → Visualization
```

The pipeline is reusable and can be easily integrated into:

* APIs
* Scheduled jobs
* Batch prediction systems

---

##  Logging & Error Handling

* Centralized logging system
* Custom exception class
* Clean error tracebacks for debugging

---

##  Extensibility

You can easily add:

* New models (XGBoost, Transformer, etc.)
* More indicators
* Live data ingestion
* REST API layer

---

## Future Improvements

* Live crypto price ingestion (API)
* Model ensemble strategy
* Dockerization
* CI/CD pipeline
* Cloud deployment

---

## Contributing

Pull requests are welcome.
For major changes, please open an issue first.

---

## License

This project is for **educational & research purposes**.

---

###  If you found this project helpful, consider giving it a star!
