import dataiku

from python_library.myfunctions import bin_values

input_ds = dataiku.Dataset("customers_web_joined")
df = input_ds.get_dataframe(limit=100000)

variables = dataiku.get_custom_variables()
if variables.get("mock_drift") == "yes":
    from python_library.myfunctions import mock_drift
    drift_threshold = float(variables["drift_threshold"])
    df["age"] = df["age"].apply(mock_drift, drift_threshold=drift_threshold)

revenue_cutoff = int(variables["revenue_cutoff"])
df["high_value"] = df["revenue"].apply(bin_values, v=revenue_cutoff)
df.drop(columns=["revenue"], inplace=True)

output_ds = dataiku.Dataset("customers_binned")
output_ds.write_with_schema(df)