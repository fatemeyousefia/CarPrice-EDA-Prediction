# Car Price | EDA & Prediction

Exploratory data analysis and a Linear Regression model that predicts the selling price of used cars.

## Overview

The project analyzes 301 used car records (299 after duplicate removal, 295 after outlier filtering). It covers the full workflow: data quality checks, feature engineering, exploratory analysis, outlier investigation, leakage safe preprocessing, model selection with cross validation, and final evaluation on a held-out test set.

## Context

This project was completed as part of the **Data Science & ML Course (IMT)** taught by **Mohamadreza Momeni**. The course provided the dataset; the analysis, modeling decisions, and write-up in this repository are my own work.

## Dataset

The dataset was provided by the course instructor and is **not included in this repository**. It contains used car listings with these features:

| Feature | Description |
|---|---|
| `Car_Name` | Name of the car model (not used in modeling) |
| `Year` | Year the car was purchased (converted to `Age`) |
| `Selling_Price` | Price at which the car was sold (target) |
| `Present_Price` | Current market price of the car model |
| `Kms_Driven` | Kilometers driven before the car was sold |
| `Fuel_Type` | Petrol, Diesel, or CNG |
| `Seller_Type` | Dealer or Individual |
| `Transmission` | Manual or Automatic |
| `Owner` | Number of previous owners |


## Method

1. Checked missing values, placeholder values, duplicates (2 removed), and data types
2. Engineered a vehicle `Age` feature (2019 − Year); dropped `Car_Name` and `Year`
3. Explored distributions and relationships between price, age, kilometers driven, fuel type, seller type, transmission, and number of owners
4. Removed 4 observations after inspecting them: 1 car with 500,000 km driven, 2 cars priced above 25, and 1 car with Owner = 3 (a single observation category)
5. Split the data 80/20 into training and test sets
6. One-hot encoded categorical features and standardized numeric features inside a scikit-learn `Pipeline`, so the test set never influences preprocessing
7. Compared three Linear Regression specifications (baseline, + Present_Price², + Present_Price³) with 5-fold cross-validation on the training data only
8. Evaluated the selected model once on the held-out test set

## Results

| Model | Mean CV R² | Std |
|---|---|---|
| Baseline Linear Regression (selected) | 0.882 | 0.012 |
| + Present_Price² | 0.880 | 0.012 |
| + Present_Price³ | 0.880 | 0.012 |

Held-out test set (59 cars): **R² = 0.865, MAE = 1.01, RMSE = 1.49** .

The polynomial terms did not improve on the baseline, so the simplest model was kept.

## Key Findings

- Present Price has by far the strongest linear relationship with Selling Price (correlation ≈ 0.89).
- Older cars tend to sell for less; kilometers driven has a much weaker linear relationship with price than Present Price does.
- Seller type, fuel type, and transmission show differences in price, but these are descriptive and should not be read as causal.

## Limitations

- Small dataset (301 rows); the test set has only 59 cars, so test metrics are noisy.
- Highly imbalanced categories (only 2 CNG cars, about 88% manual transmission).
- Cars priced above 25 were excluded, so the model is not reliable for very high-priced vehicles.
- Intended as an educational baseline, not a production pricing system.

## Project Structure

```
CarPrice-EDA-ML/
├── data/                  # place cardata.csv here (not included)
├── notebooks/
│   └── CarPrice-EDA-Prediction.ipynb
├── requirements.txt
└── README.md
```

## How to Run

1. Install the dependencies:

```bash
   pip install -r requirements.txt
```

2. Add the dataset as `data/cardata.csv` (the CSV is not included in this repository because of licensing).
3. Open and run the notebook from the `notebooks/` folder:

```bash
   jupyter notebook notebooks/CarPrice-EDA-Prediction.ipynb
```

The notebook reads `../data/cardata.csv`. Plotly charts are also saved as static images (requires `kaleido`) so they display on GitHub.

## Tools

Python 3.9, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn. Developed in Visual Studio Code.


**Author:** Fatemeh Yousefi Amiri 
