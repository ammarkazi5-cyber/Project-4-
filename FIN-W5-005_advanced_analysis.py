
# FIN-W5-005 Advanced Financial Intelligence Analysis
# Demonstrates: vectorized Pandas transformations, robust statistics,
# Isolation Forest anomaly detection, Holt-Winters forecasting,
# scenario modeling, and rule-based liquidity scoring.
import os
os.system("cls")

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

print("Financial Analysis & Forecasting")
print("=" * 60)

def add_ar_features(df, as_of):
    df = df.copy()
    df["Outstanding_Amount"] = df["Invoice_Amount"] - df["Amount_Collected"]
    df["Age_Days"] = np.maximum(0, (as_of - df["Due_Date"]).dt.days)
    df["Payment_Delay_Days"] = np.where(
        df["Payment_Date"].notna(),
        (df["Payment_Date"] - df["Due_Date"]).dt.days,
        (as_of - df["Due_Date"]).dt.days
    )
    df["Age_Bucket"] = pd.cut(
        df["Age_Days"], [-1, 30, 60, 90, 120, np.inf],
        labels=["0-30", "31-60", "61-90", "91-120", "120+"]
    )
    return df

def detect_expense_anomalies(expenses):
    x = expenses.copy()
    g = x.groupby(["Department", "Category"])["Actual_Expense"]
    x["Median"] = g.transform("median")
    x["MAD"] = g.transform(lambda s: np.median(np.abs(s - np.median(s))) + 1e-6)
    x["Robust_Z"] = 0.6745 * (x["Actual_Expense"] - x["Median"]) / x["MAD"]
    model = IsolationForest(
        n_estimators=300, contamination=0.02, random_state=42
    )
    x["IF_Flag"] = model.fit_predict(
        x[["Actual_Expense", "Median", "MAD", "Robust_Z"]]
    )
    x["Anomaly"] = (
        (x["IF_Flag"] == -1) | (x["Robust_Z"].abs() > 3.5)
    )
    return x

def forecast_cash(cash_series, horizon=6):
    model = ExponentialSmoothing(
        cash_series, trend="add", damped_trend=True
    ).fit(optimized=True)
    return model.forecast(horizon)

def scenario_model(opening_cash, collections, expenses):
    base = opening_cash + (collections - expenses).cumsum()
    stress = opening_cash + (collections * 0.85 - expenses * 1.10).cumsum()
    optimistic = opening_cash + (collections * 1.12 - expenses * 0.95).cumsum()
    return pd.DataFrame({
        "Base": base,
        "Stress": stress,
        "Optimistic": optimistic
    })
print("Ammar project".center(260))
if __name__ == "__main__":

    FILE = "FIN-W5-005_Intelligence analysis.xlsx"

    print("=" * 261)
    print("FINANCIAL INTELLIGENCE ANALYSIS".center(260))
    print("=" * 261)

    # --------------------------------
    # 1. Load Excel workbook
    # --------------------------------

    excel = pd.ExcelFile(FILE)
    F = excel.sheet_names

    print("\nSheets found:")
    
for  i in range(0, len(F), 3):
    for j in range(i, min(i + 3, len(F))):print(f"{j+1}. {F[j]:<25}",end="")
print()


    # --------------------------------
    # 2. Load datasets
    # --------------------------------

ar = pd.read_excel(FILE, sheet_name="Clean_AR_Data")
expenses = pd.read_excel(FILE, sheet_name="Clean_Expenses")
budget = pd.read_excel(FILE, sheet_name="Budgets")
cash = pd.read_excel(FILE, sheet_name="Cash_Flow")
print("\nData loaded successfully.")

    # --------------------------------
    # 3. AR Analysis
    # --------------------------------

ar_analysis = add_ar_features(
        ar,
        pd.Timestamp.today()
    )
print(("\n"+"===" * 43 +" AR ANALYSIS"+"===" * 42).center(260))
print(ar_analysis.head())

    # --------------------------------
    # 4. Expense Anomaly Detection
    # --------------------------------
 
expense_analysis = detect_expense_anomalies(
        expenses
    )
print(("\n"+"===" * 42 +"EXPENSE ANALYSIS"+"===" * 40).center(260))
print(expense_analysis.head())

    # --------------------------------
    # 5. Cash Forecast
    # --------------------------------

cash_series = (
         pd.to_numeric(
            cash["Closing_Cash"],
            errors="coerce"
        )
        .dropna()
    )

forecast = forecast_cash(
        cash_series,
        horizon=5
    )

print(("\n"+"===" * 41 +"CASH FORECAST"+"===" * 42).center(260))
print(forecast)

print("\n" + "=" * 261)
print("ANALYSIS COMPLETED".center(260))
print("=" * 261)

   
# Use the functions above with the workbook's Clean_AR_Data,
# Clean_Expenses, Budgets and Cash_Flow sheets.
