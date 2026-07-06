import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_parquet("dataset/renewables-dataset.parquet")
dataset["Time"] = pd.to_datetime(dataset["Time"])

# Group 11 residual load analysis
node_id = "11"

# These scale the already-computed solar_MWh and wind_MWh columns.
a_s = 0.01
a_w = 0.05

dataset_residual = dataset.assign(
    solar_scaled_MWh=lambda df: a_s * df["solar_MWh"],
    wind_scaled_MWh=lambda df: a_w * df["wind_MWh"],
    supply_scaled_MWh=lambda df: df["solar_scaled_MWh"] + df["wind_scaled_MWh"],
    residual_MWh=lambda df: df["demand_MWh"] - df["supply_scaled_MWh"],
)

group_data = dataset_residual.loc[dataset_residual["ID"] == node_id].copy()

print("Group 11 residual load data:")
print(
    group_data[
        [
            "Time",
            "ID",
            "demand_MWh",
            "solar_scaled_MWh",
            "wind_scaled_MWh",
            "supply_scaled_MWh",
            "residual_MWh",
        ]
    ].head()
)

print("\nSummary statistics:")
print(
    group_data[
        [
            "demand_MWh",
            "solar_scaled_MWh",
            "wind_scaled_MWh",
            "supply_scaled_MWh",
            "residual_MWh",
        ]
    ].describe()
)

plt.figure(figsize=(12, 5))
plt.plot(group_data["Time"], group_data["demand_MWh"], label="Demand")
plt.plot(
    group_data["Time"],
    group_data["supply_scaled_MWh"],
    label="Scaled renewable supply",
)
plt.plot(group_data["Time"], group_data["residual_MWh"], label="Residual load")

plt.xlabel("Time")
plt.ylabel("MWh")
plt.title("Residual Load Analysis for Group 11")
plt.legend()
plt.tight_layout()
plt.show()
