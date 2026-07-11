# Superstore Corporate Analytics Platform

An interactive, multi-page data science and predictive analytics dashboard built with **Streamlit** to optimize corporate sales performance, forecast product demand, isolate structural operational anomalies, and execute strategic market portfolio segmentation. 

This repository contains the deployment architecture and live assets for the production application hosted on the Streamlit Community Cloud.

---

## Live Application URL
**[Insert Your Live Streamlit Link Here]**

---

## Platform Core Architecture

The application is structured into four functional pages, designed to bridge complex data science models with actionable business intelligence:

### Page 1: Corporate Sales Performance Overview
* **Functionality:** Real-time data visualization engine equipped with dynamic sidebar filters for geographical regions and product categories.
* **Key Features:** Evaluates interactive annual sales distributions using bar charts and tracks time-series monthly trend aggregations via line charts to identify growth vectors.

### Page 2: Predictive Demand Forecast Explorer
* **Machine Learning Engine:** Driven by an optimized **XGBoost Regressor Baseline**.
* **Functionality:** Implements recursive multi-step monthly forecasting up to a 3-month horizon across specific granular segments (Categories and Regions).
* **Technical Details:** Builds predictive feature spaces using sliding rolling windows, temporal indicators (Month/Quarter), and lag-lag features ($t-1, t-2, t-3$). Features an inline model validation benchmark matrix showing established operational baselines (**MAE: $11,708.98**, **RMSE: $12,894.04**, **MAPE: 16.81%**).

### Page 3: Statistical Outlier & Anomaly Report
* **Machine Learning Engine:** Unsupervised **Isolation Forest Classifier** (configured at a strict 5% contamination threshold).
* **Functionality:** Continuously audits the supply chain ledger to detect weekly structural transaction outliers.
* **Key Features:** Renders a scatter plot marking extreme variances in red, accompanied by a secure corporate table log tracking high-impact fulfillment collapses and bulk purchasing spikes.

### Page 4: Strategic Market Portfolio Segmentation
* **Machine Learning Background:** Derived from multi-dimensional **K-Means Clustering** analyzing transactional velocity, year-over-year operational growth scales, and baseline volume variances.
* **Functionality:** Maps the 17 corporate sub-categories into four highly actionable, interactive stocking groups:
  * *Group 0: Core Commodities* (High Volume, Stable Demand)
  * *Group 1: Enterprise Capital Assets* (High Value, High Volatility)
  * *Group 2: Emerging Channels* (Accelerating Demand Metrics)
  * *Group 3: Low Velocity Segments* (The Long Tail)

---

## Repository Infrastructure

This repository is optimized for isolated cloud containerization and holds the production deployment ecosystem:

* `app.py` - The complete, multi-page application source code including embedded background data processing pipelines, interface scaffolding, and native plotting routines.
* `train.csv` - The underlying corporate transaction ledger containing structural ordering parameters, dates, regions, and financial sales variables.
* `requirements.txt` - The environment configuration file detailing explicit library tracking to allow the cloud server to spin up identical production containers automatically.

---

## Environment Configuration & Deployment Assets

The production environment is sustained by the following dependencies contained within `requirements.txt`:
```text
streamlit
pandas
numpy
matplotlib
xgboost
scikit-learn
