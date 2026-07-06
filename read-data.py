# %% Import the libraries we need
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------------
# 1. Load the dataset from parquet
# -------------------------------
# Make sure this path matches your local file location
dataset = pd.read_parquet("dataset/renewables-dataset.parquet")

# %% Ensure the time column is parsed as datetime
dataset["Time"] = pd.to_datetime(dataset["Time"])

# Optional: inspect the first rows
print(dataset.head())


# ----------------------------------------------
# 2. Choose penetration/scaling for solar + wind
# ----------------------------------------------
# These scale the already-computed solar_MWh and wind_MWh columns
a_s = 0.01  # solar matches (a_s x 100)% of the average yearly demand across EU
a_w = 0.05  # wind matches (a_w x 100)% of the average yearly demand across EU


# Compute scaled supply and residual load
dataset_residual = dataset.assign(
    solar_scaled_MWh=lambda df: a_s * df["solar_MWh"],
    wind_scaled_MWh=lambda df: a_w * df["wind_MWh"],
    supply_scaled_MWh=lambda df: df["solar_scaled_MWh"] + df["wind_scaled_MWh"],
    residual_MWh=lambda df: df["demand_MWh"] - df["supply_scaled_MWh"],
)[
    [
        "Time",
        "ID",
        "solar_scaled_MWh",
        "wind_scaled_MWh",
        "demand_MWh",
        "supply_scaled_MWh",
        "residual_MWh",
    ]
]

# Optional: inspect the result
print(dataset_residual.head())

# %% ---------------------------------------------------------
# 3. Filter to one node and to the dates 7-10 August 2013
# ---------------------------------------------------------
# Important: ID is stored as a string
subset_station = dataset_residual.loc[dataset_residual["ID"] == "1"].drop(columns="ID")

# %%
mask = (
    (subset_station["Time"].dt.year == 2013)
    & (subset_station["Time"].dt.month == 8)
    & (subset_station["Time"].dt.day.isin([7, 8, 9, 10]))
)

subset_data = subset_station.loc[mask].melt(
    id_vars="Time",
    var_name="source",
    value_name="MWh",
)

# Optional: inspect filtered long-format data
print(subset_data.head())


# -------------------------------
# 4. Plot the selected time series
# -------------------------------
plt.figure(figsize=(12, 5))

for source, group in subset_data.groupby("source"):
    plt.plot(group["Time"], group["MWh"], label=source)

plt.xlabel("Time")
plt.ylabel("MWh")
plt.title("Supply, demand, and residual load at node 1 (7–10 Aug 2013)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
