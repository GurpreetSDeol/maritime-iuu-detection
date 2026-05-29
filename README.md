# Maritime IUU Fishing Detection

A two-component system for detecting and scoring illegal, unreported, and unregulated (IUU) fishing activity from AIS vessel tracks. Built as a data science portfolio project targeting the UK geospatial and maritime analytics sector.

**Live app:** [share.streamlit.io](https://maritime-iuu-detection-2c8wahpeysoxkb8z9bmgkd.streamlit.app/)

---

## Overview

The project approaches IUU fishing detection from two angles that reflect how the problem is handled in practice: behavioural classification and spatial risk scoring.

**Fishing Classifier** trains a LightGBM model on 160,000 windowed AIS segments from 258 labelled fishing vessels. Rather than relying on pre-existing fishing labels as model inputs, features are engineered from raw ping kinematics: speed variance, trajectory entropy, and net displacement ratio. The model achieves 85% accuracy and an AUC-ROC of 0.927 on held-out vessels it has never seen.

**IUU Risk Scoring** applies a composite risk score to confirmed fishing events from the Global Fishing Watch API. Each event is scored based on whether it occurred inside a Marine Protected Area, the vessel's authorisation status against relevant RFMOs, whether it took place on the high seas outside any EEZ, and whether GFW flagged it as a potential risk. The output is a ranked alert feed of the highest-risk fishing activity.

---

## Data Sources

**GFW AIS Training Data** — the classifier is trained on Global Fishing Watch's published AIS training dataset, which contains per-ping fishing labels across seven gear types: trawlers, drifting longlines, fixed gear, purse seines, trollers, pole and line, and unknown. Labels were assigned by GFW analysts and a crowdsourcing campaign, with fractional values indicating annotator agreement.

**GFW Events API** — the risk scoring component uses confirmed fishing events retrieved from GFW's public fishing events API, which provides event-level metadata including vessel authorisation status, MPA overlap, and RFMO membership.

---

## Methodology

### Feature Engineering

Each vessel's ping history is processed using a sliding 2-hour window with a 30-minute stride. For each window, the following kinematic features are computed from raw AIS pings:

| Feature | Description |
|---|---|
| `speed_mean`, `speed_std` | Mean and standard deviation of speed across the window |
| `pct_slow` | Fraction of pings below 4 knots |
| `turn_rate` | Mean absolute heading change per minute |
| `traj_entropy` | Shannon entropy of heading distribution across 8 compass bins |
| `bbox_area_km2` | Area of the bounding box enclosing all pings in the window |
| `net_displacement_ratio` | Bounding box diagonal divided by total path length |
| `dist_shore_km`, `dist_port_km` | Mean distance from shore and port |
| `dark_gap_flag` | Whether any inter-ping gap exceeded 60 minutes |

The net displacement ratio is the most discriminating single feature. A vessel looping repeatedly over a fishing ground covers substantial distance but returns close to its starting point, producing a ratio near zero. A transiting vessel moving in a straight line produces a ratio near one.

### Train / Test Split

The dataset is split at vessel level, not at window level. Because consecutive windows overlap by 90 minutes, adjacent windows from the same vessel share most of their pings. A row-level split would put near-identical windows in both train and test, inflating accuracy estimates. By splitting on unique vessel identifiers, every test window comes from a vessel the model has never seen — the same condition it faces in production.

### Model

LightGBM was chosen over the original Random Forest for three reasons: it consistently outperforms Random Forest on tabular data with mixed feature types, it builds smaller model files without accuracy compromise, and its sequential tree-building approach focuses successive trees on the hardest-to-classify windows rather than treating all samples equally.

Removing the gear type feature reduces accuracy by only 1% and AUC by 0.003, confirming that the kinematic features are carrying the model rather than vessel identity.

### Risk Scoring

The composite risk score is:

```
risk_score = mpa_tier_weight × auth_multiplier × high_seas_multiplier × risk_flag_multiplier
```

MPA tier weights follow IUCN protection categories: 3.0 for strict no-take zones (Category I–II), 2.0 for partial no-take, 1.5 for general MPAs, and 1.0 for events outside any protected area. The authorisation multiplier penalises vessels with no matching RFMO authorisation (2.0) or partial matches (1.5). Fishing on the high seas outside any EEZ applies a 1.3 multiplier. GFW's own potential risk flag applies a 1.5 multiplier.

---

## Repository Structure

```
maritime-iuu-detection/
├── Python Files/
│   ├── Illegal_Fishing_Detection.ipynb   # Full pipeline: cleaning, feature engineering, model training, inference
│   └── Datasets/
│       ├── windows_features_slim.csv     # 50k labelled training windows
│       ├── unlabelled_predictions_slim.csv  # 50k inference predictions on unlabelled vessels
│       ├── ping_map_sample.csv           # 50k sampled pings for the classifier map
│       └── gfw_risk_scored.csv          # Risk-scored GFW fishing events
└── Streamlit Files/
    ├── app.py                            # Streamlit application
    └── requirements.txt
```

---

## Results

| Metric | Value |
|---|---|
| Accuracy | 85% |
| AUC-ROC | 0.927 |
| Training windows | 160,464 |
| Training vessels | 258 |
| Test vessels | 52 |
| Accuracy without gear type feature | 84% |

The 1% drop when removing gear type confirms the model has learned fishing kinematics rather than vessel identity. The most informative features are distance from port, mean speed, net displacement ratio, and trajectory entropy — all of which have clear physical interpretations grounded in how fishing vessels actually move.

---

## Limitations

The training data covers 2012–2016 and consists entirely of vessels already classified as fishing gear types by GFW. This means the model is trained on fishing vessels behaving in both fishing and non-fishing ways, rather than distinguishing fishing vessels from cargo ships or tankers. In a production system, non-fishing vessel types would be included as negative examples to make the classifier more general.

The risk scoring component uses a small sample of GFW events (400) which limits the number of high-risk cases visible in the dashboard. A larger date range or regional filter targeting known IUU hotspots would produce a more varied risk distribution.

AIS dark gap detection (gaps over 60 minutes) has near-zero importance in the training data because the GFW dataset has dense ping coverage. In a real-time satellite AIS system with global coverage, transponder disabling would be a stronger signal.

---

## Stack

Python, LightGBM, pandas, GeoPandas, Streamlit, Pydeck, Plotly, Global Fishing Watch API
