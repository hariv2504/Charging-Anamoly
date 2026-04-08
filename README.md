# EV Charging Anomaly Detector

This project detects weird things happening at EV charging stations using machine learning and simple physics rules.

## The Data

We have about 200,000 charging events from 20 different stations over the whole year 2024.

Examples of weird things we're looking for:

- Voltage too low or too high (should be around 220V, but sometimes it's 88V or 390V)
- Power going backwards (negative values)
- Current is zero but power is not zero (doesn't make sense physically)
- Station getting too hot (over 93°C)
- Energy spikes that are way too big
- Error codes showing up in the logs

## How We Find These Problems

We use two methods working together:

1. **Machine Learning (Isolation Forest)** - finds patterns that look different from normal
2. **Physics Rules** - simple checks like "power should never be negative" or "voltage should be between 190-260V"

If either method says something is wrong, we flag it as an anomaly.

## Setup

First, install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

## Usage

### Step 1: Train the model

Before we can find anomalies, we need to train the model on your data:

```bash
python3 src/train.py charging_logs.csv
```

This creates two files in the `models/` folder:
- `isolation_forest.pkl` - the trained model
- `scaler.pkl` - settings for preprocessing the data

### Step 2: Find anomalies

Now you can use the trained model to check new data:

```bash
python3 predict.py --input charging_logs.csv --output predictions.csv
```

This takes your input CSV and adds a new column called `is_anomaly`:
- `0` = normal
- `1` = anomaly detected

## Project Files

```
src/
  features.py    - creates features from raw data
  train.py       - trains the model
models/          - saves the trained model here
predict.py       - checks new data for anomalies
requirements.txt - list of packages to install
README.md        - this file
```

## Notes

- The input CSV must have the same columns as the training data
- About 2-3% of events will be flagged as anomalies
- If you train again, it will overwrite the old model
