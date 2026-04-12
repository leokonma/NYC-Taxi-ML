# NYC Taxi Fare Prediction

## Project Summary

This repository is a compact end-to-end machine learning project built on the 2019 NYC Yellow Taxi trip data. The objective is to predict `fare_amount` using information that would reasonably be known at or near pickup time, while avoiding obvious leakage from post-trip variables.

The project is organized as a notebook-based report. Each notebook represents one stage of the analytical pipeline:

1. Build a unified master table from the monthly raw files.
2. Clean the data and create a modeling-ready dataset.
3. Compare several baseline and candidate regression models.
4. Tune the strongest tree-based models and evaluate them in more detail.

Someone new to the project should be able to follow the story from raw data to final evaluation by reading the notebooks in order.

## What This Project Answers

The project is designed to answer four practical questions:

1. Can taxi fare be predicted reasonably well from trip context, distance, calendar features, and pickup/dropoff geography?
2. Which data-cleaning decisions matter most before modeling fare?
3. Do more flexible nonlinear models outperform a simple baseline and a regularized linear model?
4. Between tuned gradient boosting and tuned random forest, which model performs better on later, unseen months?

## Repository Structure

```text
data/
  raw/                     Original source files
  procesed/                Intermediate and modeling-ready datasets

notebooks/
  01_master_table_creation.ipynb
  02_cleaninig.ipynb
  03_model_comparison.ipynb
  04_hyperparameter_tuning_and_evaluation.ipynb

reports/
  models/                  Optional local model exports, ignored by git
```

Important note:
The project folder uses `procesed/` and `02_cleaninig.ipynb` as existing names. They are kept as-is for consistency with the current repository structure.

## Data and Scope

- Source: NYC Yellow Taxi trip data for 2019
- Granularity: trip-level records
- Target: `fare_amount`
- Primary use case: supervised regression for fare estimation
- Modeling philosophy: use leakage-safe predictors and evaluate chronologically

The raw monthly files are large, so the pipeline first creates a balanced master sample and then works from parquet outputs for speed and reproducibility.

## Analytical Workflow

### 1. Master Table Creation

Notebook: [01_master_table_creation.ipynb](notebooks/01_master_table_creation.ipynb)

This stage loads the monthly 2019 CSV files, aligns their schemas, samples each month consistently, and combines them into one master parquet table. The main purpose is to create a stable and traceable analytical base table before any modeling decisions are made.

Key decisions:
- use all 2019 monthly files
- sample each month with a fixed random seed
- align monthly schemas through the union of columns
- preserve metadata such as `month` and `source_file`
- save the consolidated output in parquet format

### 2. Cleaning and Feature Engineering

Notebook: [02_cleaninig.ipynb](notebooks/02_cleaninig.ipynb)

This stage converts the master table into a modeling-ready dataset. It removes invalid or implausible trips, handles impossible passenger counts, creates pickup-time and distance-derived features, and enriches trips with TLC zone metadata.

This is also where the project makes its most important leakage-control decisions.

Important preparation rules:
- remove trips with invalid fare, distance, duration, or unrealistic speed
- avoid post-trip monetary variables such as total charges and tips
- avoid realized trip-duration fields as model inputs
- keep geography through borough and service-zone features
- retain `pickup_month_num` only as an evaluation helper, not as a raw predictor

### 3. Model Comparison

Notebook: [03_model_comparison.ipynb](notebooks/03_model_comparison.ipynb)

This stage compares several regression approaches under the same chronological train-test split:

- dummy median baseline
- ridge regression
- random forest
- gradient boosting

The purpose is to identify whether more flexible nonlinear models justify their extra complexity before moving into hyperparameter tuning.

Evaluation metrics:
- MAE
- RMSE
- R²

### 4. Hyperparameter Tuning and Final Evaluation

Notebook: [04_hyperparameter_tuning_and_evaluation.ipynb](notebooks/04_hyperparameter_tuning_and_evaluation.ipynb)

This stage tunes both `GradientBoostingRegressor` and `RandomForestRegressor` with grouped cross-validation by month, retrains the tuned models on the full training window, and evaluates both on the same held-out months.

The final notebook includes:
- side-by-side tuned model comparison
- holdout MAE, RMSE, and R²
- median and high-percentile absolute error checks
- shares of predictions within small dollar-error thresholds
- performance by month
- performance by fare band
- predicted-vs-actual plots
- residual diagnostics
- worst-case error review

## Important Project Decisions

These are the most important things for a new reader to understand:

1. The project uses a chronological split, not a random split.
Training uses earlier months and testing uses later months so the evaluation is closer to a real forecasting setup.

2. Leakage prevention is a first-class design choice.
The final models avoid using variables that are only known after the trip ends or after payment is processed.

3. Data cleaning is not cosmetic here.
Removing impossible trips and corrupted targets is essential because fare prediction is highly sensitive to bad records.

4. The notebooks are intentionally staged.
Each notebook has one job, which makes the project easier to audit, explain, and rerun.

5. Saved model binaries are not versioned in Git.
Large `.joblib` artifacts are ignored so the repository stays lightweight and GitHub-friendly.

## How To Read This Project

If you are new to the repository, the fastest path is:

1. Read this README for the high-level story.
2. Open [01_master_table_creation.ipynb](notebooks/01_master_table_creation.ipynb) to understand how the base dataset is formed.
3. Open [02_cleaninig.ipynb](notebooks/02_cleaninig.ipynb) to understand the modeling rules and leakage controls.
4. Open [03_model_comparison.ipynb](notebooks/03_model_comparison.ipynb) to see the first benchmark across model families.
5. Open [04_hyperparameter_tuning_and_evaluation.ipynb](notebooks/04_hyperparameter_tuning_and_evaluation.ipynb) for the final tuning and diagnostic evaluation.

## How To Run

Run the notebooks in order because each one depends on outputs from the previous stage:

1. [01_master_table_creation.ipynb](notebooks/01_master_table_creation.ipynb)
2. [02_cleaninig.ipynb](notebooks/02_cleaninig.ipynb)
3. [03_model_comparison.ipynb](notebooks/03_model_comparison.ipynb)
4. [04_hyperparameter_tuning_and_evaluation.ipynb](notebooks/04_hyperparameter_tuning_and_evaluation.ipynb)

The intermediate parquet outputs are intentionally reused so later notebooks do not need to repeat the full raw-data ingestion step.

## Deliverables

Main analytical deliverables:
- a unified 2019 master dataset
- a cleaned, modeling-ready fare dataset
- a baseline model comparison notebook
- a tuned two-model evaluation notebook

Main communication deliverables:
- notebook text rewritten as a guided report
- README rewritten as a project overview for new readers

## Next Useful Extensions

Natural next steps for the project would be:
- add feature importance or permutation importance analysis
- compare performance by borough or airport-trip segment
- package the final scoring pipeline into a script or small app
- add environment and dependency instructions if the repo will be shared widely
