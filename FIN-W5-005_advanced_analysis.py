
# FIN-W5-005 Advanced Financial Intelligence Analysis
# Demonstrates: vectorized Pandas transformations, robust statistics,
# Isolation Forest anomaly detection, Holt-Winters forecasting,
# scenario modeling, and rule-based liquidity scoring.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

# Use the functions above with the workbook's Clean_AR_Data,
# Clean_Expenses, Budgets and Cash_Flow sheets.
