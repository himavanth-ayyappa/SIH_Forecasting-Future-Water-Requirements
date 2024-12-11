import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'KarnatakaRainfall.xlsx'  # Update with the correct file path
rainfall_data = pd.read_excel(file_path)

# Prepare to store predictions and plots for each district
all_districts = rainfall_data['DISTRICT'].unique()
predictions = {}

# Iterate through each district
for district in all_districts:
    # Filter data for the district
    district_data = rainfall_data[rainfall_data['DISTRICT'] == district]
    district_data = district_data[['Timestamp', 'VALUE']].rename(columns={'Timestamp': 'ds', 'VALUE': 'y'})
    
    # Train a Prophet model
    model = Prophet()
    model.fit(district_data)
    
    # Create future dataframe for 10 years
    future = model.make_future_dataframe(periods=10 * 12, freq='M')  # Monthly data for 10 years
    forecast = model.predict(future)
    
    # Store predictions
    predictions[district] = forecast[['ds', 'yhat']]
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(district_data['ds'], district_data['y'], label='Actual', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted', color='orange')
    plt.title(f"Rainfall Prediction for {district}")
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.legend()
    plt.show()

# Save predictions to an Excel file
output_data = []
for district, forecast in predictions.items():
    forecast['District'] = district
    output_data.append(forecast)

output_df = pd.concat(output_data, ignore_index=True)
output_df.to_excel('Rainfall_Predictions.xlsx', index=False)
print("Predictions saved to 'Rainfall_Predictions.xlsx'.")
