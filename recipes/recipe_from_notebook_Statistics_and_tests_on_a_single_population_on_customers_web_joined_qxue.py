import dataiku
import numpy as np
import pandas as pd
from scipy import stats

input_ds = dataiku.Dataset("customers_web_joined")
df = input_ds.get_dataframe(limit=10000)

numerical_columns = list(df.select_dtypes(include=[np.number]).columns)
if not numerical_columns:
    raise ValueError("customers_web_joined has no numerical columns to analyze")

value_col = numerical_columns[0]
series = pd.to_numeric(df[value_col], errors="coerce")
missing_values = int(series.isna().sum())
fill_value = series.mean()
if pd.isna(fill_value):
    fill_value = 0.0
series = series.fillna(fill_value)

mode_values = series.mode()
mode_value = np.nan if mode_values.empty else mode_values.iloc[0]
confidence = 0.05
tested_mean = 0.0

if len(series) >= 3:
    shapiro_pvalue = float(stats.shapiro(series.iloc[:5000]).pvalue)
else:
    shapiro_pvalue = np.nan

ttest_pvalue = float(stats.ttest_1samp(series, tested_mean).pvalue) if len(series) >= 2 else np.nan

result = pd.DataFrame([{
    "value_column": value_col,
    "record_count": int(series.count()),
    "missing_values_imputed": missing_values,
    "imputation_value": float(fill_value),
    "mean": float(series.mean()),
    "stddev": float(series.std()),
    "min": float(series.min()),
    "q25": float(series.quantile(0.25)),
    "median": float(series.median()),
    "q75": float(series.quantile(0.75)),
    "max": float(series.max()),
    "mode": float(mode_value) if not pd.isna(mode_value) else np.nan,
    "variance": float(series.var()),
    "skewness": float(series.skew()),
    "kurtosis": float(series.kurtosis()),
    "shapiro_pvalue": shapiro_pvalue,
    "is_normal_at_0_05": bool(shapiro_pvalue >= confidence) if not pd.isna(shapiro_pvalue) else None,
    "tested_mean": tested_mean,
    "ttest_pvalue": ttest_pvalue,
    "mean_test_rejected_at_0_05": bool(ttest_pvalue < confidence) if not pd.isna(ttest_pvalue) else None
}])

output_ds = dataiku.Dataset("output")
output_ds.write_with_schema(result)