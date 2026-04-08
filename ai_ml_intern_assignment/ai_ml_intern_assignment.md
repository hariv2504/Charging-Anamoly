# AI/ML Intern 
## Take‑Home Technical Exercise - Candidate Instructions

## 📌 Overview
This take‑home assignment simulates a realistic problem our Network Operations Center (NOC) teams face:
**detecting anomalies in EV charging station behavior using machine learning.**

You will work with a synthetic dataset representing charging session events from a fleet of EV chargers. Your task is to explore the data, engineer meaningful features, build an anomaly‑detection model, and produce a concise report of your findings.

This assignment is designed to be completed in **4-8 hours**.

---

## 🧩 Requirements
You will submit your work as a **public GitHub repository** containing:

### **1. Source Code**
A clean, well‑structured Python project that includes:
- Data loading and preprocessing
- Exploratory data analysis (EDA)
- Feature engineering 
- Model development and evaluation
- An inference pipeline for detecting anomalies
- Clear comments and docstrings

Use any ML libraries you prefer (scikit‑learn, PyTorch, TensorFlow, etc.).

---

### **2. A Short Technical Report (`REPORT.md`)**
3–5 pages summarizing:
- Your understanding of the problem
- Key insights from EDA
- Modeling approach and rationale
- Evaluation methodology
- Results and interpretation
- What you would improve with more time

---

### **3. AI Tool Usage & Documentation (`AI_USAGE.md`)**
We encourage the use of AI tools (GitHub Copilot, ChatGPT, Claude, etc.) as part of your workflow.

Please document:
- Which AI tools you used and how
- Tasks where AI was helpful
- Areas where AI struggled
- How you validated AI‑generated code

This helps us understand your engineering judgment and how you work with modern tools.

---

### **4. A Lightweight Inference Script**
A single Python file (e.g., `predict.py`) that:
- Accepts a CSV file of new logs
- Applies your preprocessing + feature engineering
- Runs your anomaly‑detection model
- Outputs a CSV with an additional column: `is_anomaly` (0/1)

Example:

```
python predict.py --input new_logs.csv --output predictions.csv
```

---

### **5. (Optional) A Notebook**
If you prefer to prototype in a Jupyter notebook, include it as well.

---

## 📊 Dataset
You are provided with a synthetic dataset:

```
charging_logs.csv
```

Each row represents an **event** during a charging session (not a full session). Columns include:

| Column | Description |
|--------|-------------|
| `station_id` | Unique charger identifier |
| `timestamp` | Event timestamp (UTC) |
| `session_id` | Charging session identifier |
| `voltage` | Voltage reading (V) |
| `current` | Current reading (A) |
| `power_kw` | Power delivered (kW) |
| `temperature_c` | Internal charger temperature (°C) |
| `error_code` | Error code emitted by charger (0 = none) |
| `message` | Text log message emitted by the charger |
| `duration_sec` | Duration of the event |
| `energy_kwh` | Energy delivered (kWh) |

The dataset includes:
- Normal operating behavior
- Known fault patterns
- Subtle anomalies
- Noise and missing values

Your job is to **identify anomalous events** using an unsupervised or semi‑supervised approach.

---

## 🧪 Expectations

### **1. Explore the Data**
Perform EDA to understand:
- Distributions
- Correlations
- Outliers
- Missing data patterns
- Station‑level differences
- Temporal patterns

Include visualizations where helpful.

---

### **2. Engineer Features**
Examples (not exhaustive):
- Rolling averages / deltas
- Power‑temperature relationships
- Session‑level aggregates
- Station‑level baselines
- Time‑based features (hour, weekday, etc.)

Explain your choices and how they help detect anomalies.

---

### **3. Build an Anomaly Detection Model**
You may choose any approach, such as:
- Isolation Forest
- One‑Class SVM
- Autoencoder
- Clustering‑based anomaly detection
- Statistical thresholds

Justify your selection and discuss tradeoffs.

---

### **4. Evaluate Your Model**
You should:
- Explain how you validated your approach
- Describe how you selected thresholds
- Discuss examples of events your model flags as anomalies
- Interpret false positives and false negatives based on your reasoning
- Explain how your model might behave in production

We are evaluating your thinking, not just your metrics.

---

### **5. Build an Inference Pipeline**
Your `predict.py` script should:
- Load a CSV file
- Apply your preprocessing + feature engineering
- Run your model
- Output a CSV with a new column: `is_anomaly`

---

## 📝 Evaluation Criteria

### **Technical Depth**
- Correctness of ML approach
- Quality of feature engineering
- Soundness of evaluation

### **Code Quality**
- Readability
- Structure
- Documentation
- Reproducibility

### **Reasoning & Communication**
- Clarity of explanations
- Ability to justify decisions
- Awareness of tradeoffs

### **Practicality**
- Would this approach scale to real NOC data
- Robustness to noise and missing values
- Interpretability of results

---

## 📦 Submission Instructions
1. Create a **public GitHub repository**.
2. **Do not** put ChargePoint's name in the repo title or any filenames as our SecOps team will flag them.
3. Include:
   - `README.md` (instructions to run your code)
   - `REPORT.md`
   - `AI_USAGE.md`
   - Source code
   - Notebook (optional)
4. Share the GitHub link with us.

---

## 🤖 Our Philosophy on AI
We view AI coding assistants as powerful productivity tools. We’re interested in how you integrate these tools into your workflow while maintaining code quality, understanding, and ownership. There is no penalty for using AI tools.

---

## 💼 About This Exercise
This assignment is designed as a realistic but self‑contained problem. It is not part of ChargePoint’s product roadmap. We will never use your submission for commercial purposes.

We recognize that take‑home assignments require time and effort. We’ve intentionally scoped this to be reasonable, and we will provide feedback whenever possible.

---

## 🧡 Thank You
We appreciate the time and effort you invest in this exercise. Your submission will be reviewed by senior engineering staff, and we will follow up with feedback and next steps.

Good luck!  We’re excited to see your work!
