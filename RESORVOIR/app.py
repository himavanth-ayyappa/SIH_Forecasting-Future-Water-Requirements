import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset (update the file path as needed)
file_path = 'reservoir_karnataka.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# Ensure Date column is in datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m')

# Select relevant columns
forecast_data = data[['Date', 'Current Live Storage', 'District', 'Reservoir Name']].rename(
    columns={'Date': 'ds', 'Current Live Storage': 'y'}
)

# Drop rows with missing values
forecast_data = forecast_data.dropna()

# Get unique reservoirs
reservoirs = forecast_data['Reservoir Name'].unique()

# Initialize an empty list to store the merged data for each reservoir
all_reservoirs_forecast = []

for reservoir in reservoirs:
    # Filter data for each reservoir
    reservoir_data = forecast_data[forecast_data['Reservoir Name'] == reservoir]

    # Initialize and fit the Prophet model for the current reservoir
    model = Prophet()
    model.fit(reservoir_data[['ds', 'y']])

    # Create future dataframe for the next 10 years (120 months)
    future = model.make_future_dataframe(periods=120, freq='M')

    # Make predictions
    forecast = model.predict(future)

    # Merge predictions with original data for actual vs. prediction comparison
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    merged_data = pd.merge(reservoir_data, forecast, on='ds', how='outer')

    # Append the merged data for each reservoir to the list
    all_reservoirs_forecast.append(merged_data)

    # Plot actual vs. predicted data for the current reservoir
    plt.figure(figsize=(12, 6))
    plt.plot(reservoir_data['ds'], reservoir_data['y'], label='Actual Live Storage', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Live Storage', color='orange', linestyle='--')
    plt.fill_between(
        forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.2, label='Confidence Interval'
    )
    plt.title(f'Actual vs Predicted Live Storage for {reservoir}')
    plt.xlabel('Date')
    plt.ylabel('Live Storage (in million cubic meters)')
    plt.legend()
    plt.grid()
    plt.show()

# Combine all reservoirs' forecast data
final_merged_data = pd.concat(all_reservoirs_forecast)

# Save the forecasted values to an Excel file
output_file = 'forecasted_live_storage_per_reservoir.xlsx'
final_merged_data.to_excel(output_file, index=False)
print(f"Forecast saved to {output_file}")
