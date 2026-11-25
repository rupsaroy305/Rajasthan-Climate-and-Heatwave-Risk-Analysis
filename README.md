# Rajasthan-Climate-and-Heatwave-Risk-Analysis

### Historical Climate Analysis | Machine Learning | Heatwave Prediction | Climate Scenario Simulation

An end-to-end climate data science and machine learning project that investigates historical climate patterns, heatwave trends, and district-level heatwave risks across Rajasthan, India, using daily climate data from **2006 to 2025**.

The project combines exploratory data analysis, temporal feature engineering, machine learning-based next-day heatwave prediction, unsupervised climate regime discovery, and hypothetical 2035 climate scenario simulations through an interactive Streamlit dashboard.

---

## Project Overview

Rajasthan experiences extreme summer temperatures and recurring heatwave conditions. Understanding how these patterns evolve over time and how heatwave risks differ across districts is important for climate resilience and preparedness.

This project aims to analyze historical climate conditions, identify patterns associated with heatwaves, and develop a machine learning framework to estimate next-day heatwave risk.

Rather than focusing only on historical temperature trends, the project also explores how hypothetical changes in temperature, rainfall, and soil moisture could influence future heatwave risk.

## Project Objectives

- Analyze historical climate patterns across Rajasthan from 2006–2025.
- Examine annual heatwave frequency and maximum temperature trends.
- Compare heatwave occurrence and climate characteristics across districts.
- Identify important meteorological factors associated with heatwave conditions.
- Develop a leakage-aware machine learning model for next-day heatwave prediction.
- Discover distinct climate regimes using K-Means clustering.
- Simulate hypothetical climate conditions for 2035.
- Assess how different climate scenarios affect district-level heatwave risk.
- Build an interactive dashboard for exploring the analysis and model outputs.

## Dataset

**Dataset:** `Rajasthan_Heatwave_2006_2025.csv`

The analysis uses daily climate observations covering the period 2006–2025 across nine Rajasthan districts.

The dataset contains temporal, geographical, and meteorological variables, including:

| Category | Variables |
|---|---|
| Temporal | YEAR, MONTH, DAY |
| Geographical | DISTRICT, LAT, LON |
| Temperature | TEMP2M, TMAX, TMIN |
| Atmospheric | DEW2M, MSLP, GEOP, BLH |
| Wind | WIND_U10, WIND_V10 |
| Precipitation | RAIN |
| Solar radiation | SRAD |
| Evaporation | EVAP |
| Soil conditions | SOILT1, SOILM1 |
| Vegetation | LAI |
| Target variable | HEATWAVE |

*The dataset is expected to be supplied separately as a CSV file.*

## Methodology

### 1. Data Preprocessing

- Loaded and examined historical daily climate observations.
- Handled missing and invalid records.
- Constructed date information from year, month, and day.
- Converted temperature measurements from Kelvin to Celsius where applicable.
- Organized observations chronologically by district.
- Prepared the heatwave indicator as the prediction target.

### 2. Exploratory Data Analysis (EDA)

Explored historical climate characteristics through statistical summaries and visualizations.

Key areas of analysis:

- Annual heatwave frequency.
- Annual mean maximum temperature.
- District-wise heatwave occurrence.
- Temporal variations in climate conditions.
- Relationships between meteorological variables and heatwave patterns.

### 3. Feature Engineering

Created temporal features to capture historical climate behaviour and improve next-day predictions.

Engineered features include:

- One-day lag features.
- Three-day rolling averages.
- Seven-day rolling averages.
- Previous-day heatwave indicators.
- Seven-day historical heatwave frequency.
- Seasonal cyclical features using sine and cosine transformations.

Temperature, atmospheric, rainfall, wind, soil, and other climate variables were incorporated into the feature engineering process.

### 4. Next-Day Heatwave Prediction

Developed a binary classification framework to estimate whether a heatwave will occur on the following day.

**Target variable:** `HEATWAVE_NEXT_DAY`

The prediction pipeline uses chronological feature construction and a time-based training strategy to reduce information leakage.

**Modeling approach:**

- XGBoost Classifier as the primary machine learning model.
- HistGradientBoosting Classifier as an alternative when XGBoost is unavailable.

**Temporal validation strategy:**

| Dataset | Period |
|---|---|
| Training | Before 2022 |
| Testing | 2022 onwards |

Model performance is evaluated using:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

### 5. Climate Regime Discovery

Applied K-Means clustering to identify different climate regimes based on meteorological characteristics.

The clustering analysis considers variables such as:

- Maximum temperature
- Rainfall
- Soil moisture
- Solar radiation
- Vegetation index

The resulting clusters are compared using their historical heatwave rates and climate characteristics.

This helps explore whether different combinations of environmental conditions are associated with varying heatwave risks.

### 6. 2035 Climate Scenario Simulation

Developed a hypothetical scenario simulator to investigate how changes in environmental conditions could affect heatwave probabilities.

Six exploratory scenarios are considered:

| Scenario | Assumption |
|---|---|
| Baseline | Existing climate conditions |
| Temperature Increase | +1°C |
| Higher Temperature Increase | +2°C |
| Reduced Rainfall | -20% rainfall |
| Reduced Soil Moisture | -10% soil moisture |
| Combined Hot & Dry | Combined heat and moisture stress |
| Adaptation Scenario | +10% LAI/soil moisture |

The scenario analysis compares predicted heatwave probabilities against baseline conditions.

*These scenarios are controlled what-if simulations, not official climate projections or validated weather forecasts.*

### 7. District-Level Risk Analysis

Investigated how different climate scenarios affect heatwave sensitivity across individual districts.

The analysis includes:

- Historical district-wise heatwave rates.
- District-level predicted heatwave probabilities.
- Changes in predicted risk relative to baseline conditions.
- Identification of districts with greater sensitivity to hypothetical climate changes.

---

## Interactive Streamlit Dashboard

The project includes an interactive dashboard built using Streamlit.

### Dashboard Modules

**1. Climate Overview**

- Historical climate statistics.
- Annual heatwave trends.
- Annual mean maximum temperature.
- District-wise heatwave comparisons.

**2. Next-Day Heatwave Prediction**

- Machine learning model evaluation.
- Predicted heatwave probabilities.
- Confusion matrix.
- ROC and Precision-Recall analysis.

**3. Climate Regimes**

- K-Means clustering results.
- Climate regime comparisons.
- Heatwave frequency across identified clusters.

**4. 2035 Scenario Simulator**

- Compare hypothetical climate scenarios.
- Examine changes in predicted heatwave probabilities.
- Download scenario comparison results.

**5. District Risk**

- Compare district-level scenario sensitivity.
- Identify changes in predicted heatwave probabilities.
- Download district-wise risk analysis.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Clustering | K-Means |
| Model Evaluation | Scikit-learn Metrics |
| Interactive Dashboard | Streamlit |
| Development Environment | Jupyter Notebook |

##  Repository Structure

```text
Rajasthan-Climate-and-Heatwave-Risk-Analysis/
│
├── Rajasthan_Climate_and_Heatwave_Risk_Analysis.ipynb
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

**File descriptions:**

- `Rajasthan_Climate_and_Heatwave_Risk_Analysis.ipynb` – Main notebook containing the climate analysis, preprocessing, feature engineering, modeling, and experiments.
- `app.py` – Interactive Streamlit dashboard for climate exploration, heatwave prediction, clustering, and scenario simulation.
- `requirements.txt` – Required Python dependencies.
- `README.md` – Project documentation.
- `LICENSE` – MIT License.



##  Requirements

The project dependencies include:

```text
streamlit>=1.32
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.3
xgboost>=2.0
```

## Future Scope

Potential improvements and extensions include:

- Incorporating additional meteorological and historical climate datasets.
- Exploring advanced time-series forecasting techniques.
- Comparing additional machine learning and ensemble models.
- Improving model interpretability through SHAP-based feature analysis.
- Integrating geospatial visualization for district-level heatwave mapping.
- Exploring more realistic future climate projections.
- Developing an early-warning framework for heatwave preparedness.

##  Limitations

- Predictions depend on the availability and quality of historical climate observations.
- Model performance on historical test data does not guarantee future forecasting accuracy.
- Climate regime clusters represent statistical groupings rather than official meteorological classifications.
- The 2035 scenarios are hypothetical sensitivity experiments, not scientifically validated future climate forecasts.
- The dashboard is a research prototype and does not provide official weather warnings.



## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Exploring historical climate patterns, understanding heatwave risks, and investigating possible future climate scenarios through data science and machine learning.* 🌍
