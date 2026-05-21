# NeuralTrade

AI-powered stock prediction system for NSE large-cap equities using an ensemble of XGBoost and LSTM models.

---

## Overview

NeuralTrade is a full-stack machine learning platform that generates stock market prediction signals for major Indian stocks listed on the National Stock Exchange (NSE).

The system combines:

* XGBoost for structured feature-based prediction
* LSTM neural networks for sequential time-series forecasting
* Ensemble logic for generating final trading signals

The platform includes:

* Machine learning pipeline in Python
* FastAPI inference server
* Express.js backend with TypeScript
* Next.js frontend dashboard
* PostgreSQL database
* JWT authentication system

This project is built for educational and research purposes only.

---

## Features

* Ensemble prediction system using XGBoost and LSTM
* Predictions for 12 NSE large-cap stocks
* Authentication and protected dashboard
* Watchlist management
* Prediction confidence visualization
* PostgreSQL caching layer
* REST API architecture
* Full-stack TypeScript implementation
* FastAPI integration for ML inference

---

## Prediction Signals

| Signal      | Meaning                               |
| ----------- | ------------------------------------- |
| Strong Buy  | Both models predict upward movement   |
| Buy         | XGBoost predicts upward movement      |
| Strong Sell | Both models predict downward movement |
| Sell        | LSTM predicts downward movement       |
| Hold        | Models disagree                       |

---

## Supported Stocks

| Ticker        | Company                   | Sector  |
| ------------- | ------------------------- | ------- |
| RELIANCE.NS   | Reliance Industries       | Energy  |
| TCS.NS        | Tata Consultancy Services | IT      |
| INFY.NS       | Infosys                   | IT      |
| HDFCBANK.NS   | HDFC Bank                 | Banking |
| ICICIBANK.NS  | ICICI Bank                | Banking |
| SBIN.NS       | State Bank of India       | Banking |
| AXISBANK.NS   | Axis Bank                 | Banking |
| WIPRO.NS      | Wipro                     | IT      |
| HCLTECH.NS    | HCL Technologies          | IT      |
| ITC.NS        | ITC Limited               | FMCG    |
| MARUTI.NS     | Maruti Suzuki             | Auto    |
| BHARTIARTL.NS | Bharti Airtel             | Telecom |

---

## System Architecture

```text
Frontend (Next.js)
        |
        v
Backend API (Node.js + Express)
        |
        v
FastAPI ML Server (Python)
        |
        v
XGBoost + LSTM Models
```

---

## Machine Learning Pipeline

### Features Used

* Trading Volume
* Daily Returns
* Moving Averages (10, 20, 50)
* Volatility
* RSI
* MACD
* Signal Line
* Distance from MA50
* Lag Features

### Target Definition

```python
Return > +1%  -> BUY
Return < -1%  -> SELL
Otherwise      -> Ignore (noise filtering)
```

### Ensemble Logic

```python
if xgb == BUY and lstm == BUY:
    signal = "STRONG BUY"

elif xgb == SELL and lstm == SELL:
    signal = "STRONG SELL"

else:
    signal = "HOLD"
```

---

## Project Structure

```text
stock_prediction_system/

├── ml-model/
│   ├── fetch_stock_data.py
│   ├── preprocess_data.py
│   ├── train_models.py
│   ├── train_lstm.py
│   ├── predict_ensemble.py
│   ├── backtest.py
│   ├── check_models.py
│   ├── main.py
│   └── requirements.txt
│
├── server/
│   └── src/
│       ├── config/
│       ├── controllers/
│       ├── middleware/
│       ├── prisma/
│       ├── routes/
│       ├── services/
│       ├── types/
│       └── utils/
│
└── client/
    ├── app/
    ├── components/
    ├── hooks/
    ├── lib/
    └── types/
```

---

## Tech Stack

### Machine Learning

* Python
* TensorFlow / Keras
* XGBoost
* scikit-learn
* pandas
* numpy
* yfinance

### Backend

* Node.js
* Express.js
* TypeScript
* Prisma ORM
* PostgreSQL
* JWT Authentication

### Frontend

* Next.js 14
* TypeScript
* Tailwind CSS
* Zustand
* Axios

---

## Getting Started

### Prerequisites

* Node.js 18+
* Python 3.10+
* PostgreSQL 14+

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/shivanshsuryan04/stock_prediction_system.git

cd stock_prediction_system
```

---

### 2. Setup Database

```bash
createdb stock_prediction_db
```

---

### 3. Setup ML Pipeline

```bash
cd ml-model

python -m venv venv

source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run ML scripts:

```bash
python fetch_stock_data.py

python preprocess_data.py

python train_models.py

python train_lstm.py

python check_models.py
```

---

### 4. Setup Backend

```bash
cd ../server

npm install

cp .env.example .env
```

Generate Prisma client:

```bash
npx prisma generate
```

Run migrations:

```bash
npx prisma migrate dev --name init
```

Start backend server:

```bash
npm run dev
```

---

### Backend Environment Variables

```env
NODE_ENV=development

PORT=5000

DATABASE_URL="postgresql://USER:PASSWORD@localhost:5432/stock_prediction_db"

JWT_ACCESS_SECRET=your_access_secret

JWT_REFRESH_SECRET=your_refresh_secret

FRONTEND_URL=http://localhost:3000

PYTHON_ML_API_URL=http://localhost:8000

ML_DIR=../ml-model
```

---

### 5. Setup Frontend

```bash
cd ../client

npm install

cp .env.example .env.local
```

Frontend environment variable:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000/api
```

Start frontend:

```bash
npm run dev
```

---

### 6. Start FastAPI ML Server

```bash
cd ../ml-model

source venv/bin/activate

uvicorn main:app --port 8000 --reload
```

---

## Running Services

| Service           | Port |
| ----------------- | ---- |
| Next.js Frontend  | 3000 |
| Express Backend   | 5000 |
| FastAPI ML Server | 8000 |

Open:

```text
http://localhost:3000
```

---

## API Endpoints

### Authentication

| Method | Endpoint           | Description          |
| ------ | ------------------ | -------------------- |
| POST   | /api/auth/register | Register user        |
| POST   | /api/auth/login    | Login user           |
| POST   | /api/auth/refresh  | Refresh access token |
| POST   | /api/auth/logout   | Logout user          |
| GET    | /api/auth/me       | Get current user     |

---

### Predictions

| Method | Endpoint                 | Description           |
| ------ | ------------------------ | --------------------- |
| GET    | /api/predictions         | Get all predictions   |
| GET    | /api/predictions/:ticker | Get single prediction |
| GET    | /api/predictions/tickers | Get supported tickers |

---

### Watchlist

| Method | Endpoint               | Description   |
| ------ | ---------------------- | ------------- |
| GET    | /api/watchlist         | Get watchlist |
| POST   | /api/watchlist         | Add ticker    |
| DELETE | /api/watchlist/:ticker | Remove ticker |

---

## Security Features

* JWT Authentication
* HTTP-only refresh cookies
* Password hashing using bcrypt
* Helmet.js security headers
* Rate limiting
* Input validation
* CORS protection
* Prisma parameterized queries

---

## Backtesting

Run:

```bash
cd ml-model

python backtest.py
```

Example output:

```text
RELIANCE.NS   | Market: 142.35% | XGB: 198.12% | LSTM: 221.44%

TCS.NS        | Market: 198.71% | XGB: 245.33% | LSTM: 268.90%

HDFCBANK.NS   | Market: 89.42%  | XGB: 134.67% | LSTM: 156.23%
```

---

## Disclaimer

This project is intended for educational and research purposes only.

The generated predictions are not financial advice. Machine learning models do not guarantee future market performance.

Always conduct your own research before making investment decisions.

---

## Author

Shivansh Suryan

LinkedIn: www.linkedin.com/in/shivanshsuryan04

Source content adapted from uploaded README draft. 
