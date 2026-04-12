
# NYC Taxi Fare Prediction 🚕

## Objective

Predict the taxi fare (`fare_amount`) using NYC yellow taxi trip data (2019).

## Project Structure

```
data/
    raw/        # Original monthly data (never modified)
    processed/  # Cleaned datasets and final master table

notebooks/
    01_master_table_creation.ipynb
    02_cleaninig.ipynb
    03_model_comparison.ipynb
```

**Why this structure?**
- Separates raw vs processed data → reproducibility
- Notebooks document the full pipeline step by step
- Processed data is ready for modeling


## CRISP-DM Approach

### 1. Business Understanding
Goal: estimate taxi fares based on trip characteristics.  
Use case: pricing analysis and understanding cost drivers.

---

### 2. Data Understanding
- Source: NYC Yellow Taxi dataset (2019)
- Monthly trip-level data (distance, time, location, etc.)
- Initial exploration to detect missing values and anomalies

---

### 3. Data Preparation
Main steps:
- Sample ~1M trips per month
- Merge all months into one dataset
- Remove invalid trips:
  - non-positive fares
  - negative/zero duration
  - zero distance
  - unrealistic speeds
- Handle missing values and impossible passenger counts
- Enrich pickup/dropoff locations with TLC zone metadata


## Final Dataset

Stored in:


data/processed/master_2019_1M_per_month.parquet


