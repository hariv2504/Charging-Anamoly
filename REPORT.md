# Report — EV Charging Anomaly Detection

## What We're Trying to Do

We want to find weird things happening at EV charging stations. Each row in our data is one charging event, not a whole session. So we need to catch problems that happen in individual events.

We don't have anyone telling us which events are actually bad (no labels). So we have to figure out what "normal" looks like on our own, and then look for things that don't match that pattern.

---

## Key Insights from the Data

The dataset has 199,566 events across 20 stations and 4,000 sessions spanning a full year (Jan–Dec 2024). No missing values, which made preprocessing straightforward.

A few things stood out immediately from basic exploration:

**Physics violations** — some events are clearly wrong regardless of any model:
- ~1,100 events with negative `power_kw` (energy flowing backwards during a charging event makes no sense)
- ~1,150 events where `current = 0` but `power_kw > 0` (inconsistent metering)
- ~950 events with voltage below 150V (normal Level 2 charging is 208–240V)
- ~350 events with temperature above 75°C (thermal threshold for concern)

**Error codes** — about 1,100 events have non-zero error codes (101, 202, 303, 404) with messages like "Severe voltage instability detected" and "Unexpected reboot during active session". These are labeled faults.

**Station differences** — stations don't all operate at the same baseline. Some run consistently warmer (STATION_3, STATION_4 average ~50°C) while others run cooler (~35°C). This matters for anomaly detection — a temperature of 55°C is normal for one station and suspicious for another.

**Subtle anomalies** — beyond the obvious violations, there are events where the power factor (P / V·I) deviates significantly from the expected ~0.95–1.0 range, and events where `energy_kwh` is inconsistent with `power_kw * duration_sec / 3600`. These are harder to catch with rules alone.

---

## How We Built Features

We created different types of features to help the model learn:

**Simple yes/no flags:**
- Is power negative? Is current zero when power isn't? Is voltage too low or too high? Is it too hot? Is there an error code?

**Power factor check:**
- For a working charger, power / (voltage × current) should be close to 1.0. If it's way off, something is wrong.

**Energy check:**
- Does the energy match what we'd expect from the power and time? If not, something is odd.

**Looking at recent history:**
- For each station, we look at the last 20 events and compute the average
- Then we compare each new event to that average and see how different it is
- A 55°C temperature is weird if the station usually runs at 35°C, even if 55°C isn't globally weird

**Session stats:**
- For each session, we track average power, power swings, max temperature, and error count
- Then we add these stats to each event in that session
- One bad event looks different from an event in a session that's been having problems all along

**Time of day:**
- Hour of day and day of week
- Charging patterns are different at night versus during work hours

---

## Modeling Approach

I used **Isolation Forest** as the main model, combined with a **hard rule layer** on top.

**How Isolation Forest works:**
- It builds random decision trees that split the data
- Rare events (anomalies) get split off quickly because they're unusual
- Normal events take longer to separate
- So we can tell what's normal and what's weird by counting how many splits it takes

**Why this approach:**
- It's good at finding things that are weird in multiple ways at once
- It's fast and doesn't need us to guess what a "normal" distribution looks like
- It's easy to understand compared to other ML models

**The hard rule layer:**
- This catches obvious physics violations like negative power or temperature that's too high
- These are flagged no matter what the model says

**The contamination parameter:**
- I set this to 2.5%, which means the model looks for about that percentage of events to be anomalies
- This was based on the obvious physics violations we saw in the data

---

## Evaluation

Without ground truth labels, direct precision/recall isn't possible. Instead I validated the approach in a few ways:

**Sanity check on hard rules** — all events with non-zero error codes, negative power, zero current, voltage drops, and overtemperature are flagged. These are unambiguous faults, so 100% recall on them is a baseline requirement.

**Qualitative inspection** — I sampled flagged events from the Isolation Forest (not caught by hard rules) and checked whether they made physical sense. Most fell into a few patterns: unusual power factor values, energy readings inconsistent with duration, or readings that were normal in isolation but anomalous relative to the station's recent rolling baseline.

**Anomaly rate by station** — the distribution of flagged events across stations is uneven, which is expected. Some stations have more fault events in the data. If the rate were uniform across all stations it would suggest the model is just randomly sampling rather than detecting real patterns.

**False positives** — the main risk is flagging normal events that happen to be at the edge of the distribution (e.g., a legitimate high-power event at a station that usually runs low). The rolling z-score features help here because they're relative to recent behavior rather than global thresholds. Still, some false positives are inevitable in an unsupervised setting.

**False negatives** — subtle anomalies that look normal across all features individually but are anomalous in combination are the hardest to catch. The multivariate nature of Isolation Forest helps, but there's no way to know the true false negative rate without labels.

---

## Results

On the training data:
- About 3% of events flagged as anomalies
- Hard rules alone catch about 1,100 obvious physics violations
- The Isolation Forest finds several thousand more that are statistically unusual but don't trigger hard rules
- Different stations have different anomaly rates, which makes sense given the data

---

## What I'd Improve With More Time

**Labels** — even a small set of manually labeled anomalies would let me tune the contamination parameter properly and measure precision/recall. Right now the 2.5% threshold is a judgment call.

**Autoencoder comparison** — an LSTM autoencoder trained on session sequences could capture temporal dependencies within a session better than the rolling window approach. Worth benchmarking.

**Per-station models** — since stations have different baselines, training a separate Isolation Forest per station might reduce false positives from the global model treating inter-station variation as anomalous.

**Message text** — the `message` column has structured fault descriptions. A simple keyword classifier on top of the existing model would improve recall on known fault types.

**Drift detection** — the current model is static. In production, charger behavior drifts over time (aging hardware, seasonal temperature changes). An online or periodically retrained model would handle this better.
