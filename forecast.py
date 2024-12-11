import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# File path to the Excel sheet
file_path = 'KARNATAKAPOPULATION.xlsx'  # Replace with your actual file path

# Read the dataset
data = pd.read_excel(file_path, header=None)

# Assign column names based on your data
data.columns = ['District', '1991', '2001', '2011', '2022']

# Function to clean population data
def clean_population_column(column):
    # Remove commas, replace non-numeric values with NaN, and drop rows with NaN
    return pd.to_numeric(column.astype(str).str.replace(',', ''), errors='coerce')

# Clean population columns
for column in data.columns[1:]:
    data[column] = clean_population_column(data[column])

# Drop rows with invalid data
data = data.dropna()

# Extract years for creating the DataFrame
years = ['1991', '2001', '2011', '2022']

# Create a dictionary to store district-wise data
district_data = {}
for i, row in data.iterrows():
    district = row['District']
    populations = row[1:].values

    # Create a DataFrame for Prophet with 'ds' as years and 'y' as population
    district_data[district] = pd.DataFrame({
        'ds': pd.to_datetime(years, format='%Y'),
        'y': populations
    })

# Initialize a dictionary to store models and future predictions
models = {}
future_predictions = {}
future_years = 5  # Number of years to predict

# Process each district's data
for district, df in district_data.items():
    # Ensure each district is trained on its own data
    print(f"Training model for {district} using its own data...")
    
    # Create and fit the Prophet model
    model = Prophet()
    model.fit(df)
    models[district] = model

    # Create a DataFrame for future years
    future = model.make_future_dataframe(periods=future_years, freq='Y')
    forecast = model.predict(future)

    # Store the forecast
    future_predictions[district] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Plot the forecast
    plt.figure(figsize=(10, 5))
    plt.plot(df['ds'], df['y'], label="Historical Data", marker='o')
    plt.plot(forecast['ds'], forecast['yhat'], label="Forecast", color='orange', linestyle='--')
    plt.fill_between(
        forecast['ds'], 
        forecast['yhat_lower'], 
        forecast['yhat_upper'], 
        color='orange', alpha=0.2, label='Confidence Interval'
    )
    plt.title(f"{district} - Population Forecast")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.grid(True)
    plt.show()

# Display future predictions
for district, forecast in future_predictions.items():
    print(f"{district} - Predicted Population for Next {future_years} Years:")
    print(forecast.tail(future_years))
