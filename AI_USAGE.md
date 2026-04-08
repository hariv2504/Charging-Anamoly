# AI Tool Usage

## Tools Used

I used **GitHub Copilot** to help write the code. It's really useful for getting boilerplate code right quickly.

---

## What AI Helped With

**Basic code structure** — things like how to set up argparse, save/load models with joblib, and pandas groupby operations. I could write these myself but AI got them right the first time.

**Pandas syntax** — the rolling window and z-score calculations in pandas are tricky. AI helped with the `.groupby().transform().rolling()` syntax which I verified was correct.

**Exploration notebook** — AI suggested a good flow for the notebook cells. I rearranged some sections based on what I actually found in the data.

**Documentation and reports** — I used ChatGPT to help structure the README, REPORT, and other documentation. It helped me organize my thoughts and write clearly. I then edited everything to match my actual findings and make it simpler.

---

## What AI Got Wrong

**Domain knowledge** — AI doesn't know what voltage ranges are safe for EV chargers or what temperature is too hot. I had to research those myself. The 190-260V range and 75°C threshold came from my own research.

**Tuning parameters** — AI suggested contamination=0.1 (10%) which was too high. I lowered it to 0.025 based on how many obvious violations I saw in the data.

**Report writing** — AI's drafts were generic and boring. I rewrote the report to actually describe what I found, not just generic anomaly detection stuff.

---

## How I Checked AI's Code

- Ran the feature engineering on a small sample and checked the values looked right
- Tested that voltage drop events got high z-scores (they did)
- Compared hard rule flags to my manual counts (they matched)
- Ran the full training and looked at the flagged events to make sure they made sense
