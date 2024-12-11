import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the data from your Excel file
file_path = 'Groundwater.xlsx'  # Replace with the path to your Excel file
df = pd.read_excel(file_path)

# Assuming the Excel file has columns: 'Year', 'District', and 'GroundWaterlevel'
# Make sure to inspect your data first using df.head() to check the exact column names

# Prepare the data for Prophet
# Prophet expects columns "ds" for dates and "y" for values
df['ds'] = pd.to_datetime(df['Year'], format='%Y')  # Convert 'Year' to datetime format
df['y'] = df['GroundWaterlevel']  # Groundwater level as the target variable

# Initialize the Prophet model
model = Prophet(yearly_seasonality=True)  # Set yearly seasonality

# Fit the model
model.fit(df[['ds', 'y']])

# Make a future dataframe for predictions (next 5 years, change as needed)
future = model.make_future_dataframe(df, periods=5, freq='Y')

# Make predictions
forecast = model.predict(future)

# Plot the forecast
model.plot(forecast)
plt.title('Groundwater Level Forecast')
plt.xlabel('Year')
plt.ylabel('Groundwater Level')
plt.show()

# To view the forecasted data (optional)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
