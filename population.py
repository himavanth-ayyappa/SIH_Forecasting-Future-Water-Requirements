import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# File path to the Excel sheet
file_path = 'KARNATAKAPOPULATIONUPDATED.xlsx'  # Replace with your actual file path

# Load the dataset
data = pd.read_excel(file_path, header=0)

# Reshape the data to a long format for easier processing
data_long = data.melt(id_vars=["Year"], var_name="District", value_name="Population")

# Clean the data
data_long["Population"] = pd.to_numeric(data_long["Population"], errors="coerce")
data_long["Year"] = pd.to_numeric(data_long["Year"], errors="coerce")
data_long = data_long.dropna()

# Filter only valid year and population data
data_long = data_long[data_long["Year"] <= 2022]

# Directory to save graphs
os.makedirs("District_Population_Graphs", exist_ok=True)

# Initialize a dictionary to store predictions for all years up to 2035
all_predictions = {}

# Process each district
for district in data_long["District"].unique():
    # Filter data for the current district
    district_data = data_long[data_long["District"] == district][["Year", "Population"]]
    district_data.rename(columns={"Year": "ds", "Population": "y"}, inplace=True)
    district_data["ds"] = pd.to_datetime(district_data["ds"], format='%Y')

    # Train the Prophet model
    model = Prophet()
    model.fit(district_data)

    # Create a DataFrame for future predictions up to 2035
    future = model.make_future_dataframe(periods=(2035 - 2022), freq='Y')
    forecast = model.predict(future)

    # Store predictions for this district
    all_predictions[district] = forecast[["ds", "yhat"]]

    # Plot the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(district_data["ds"], district_data["y"], label="Historical Data", marker="o")
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="orange", linestyle="--")
    plt.fill_between(
        forecast["ds"], 
        forecast["yhat_lower"], 
        forecast["yhat_upper"], 
        color="orange", 
        alpha=0.2, 
        label="Confidence Interval"
    )
    plt.title(f"{district} - Population Forecast (up to 2035)")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)

    # Save the graph as a PNG file
    graph_path = os.path.join("District_Population_Graphs", f"{district}_forecast.png")
    plt.savefig(graph_path)
    plt.close()

# Combine predictions into a single DataFrame
combined_predictions = pd.DataFrame()

for district, forecast in all_predictions.items():
    forecast["District"] = district
    combined_predictions = pd.concat([combined_predictions, forecast])

# Filter predictions for the years 2022 to 2035
combined_predictions = combined_predictions[combined_predictions["ds"].dt.year >= 2022]

# Save the predictions to a CSV file
combined_predictions.to_csv("Population_Predictions_2022_to_2035.csv", index=False)

print("Predictions for all years up to 2035 saved to 'Population_Predictions_2022_to_2035.csv'.")
print("Graphs saved in the 'District_Population_Graphs' folder.")
