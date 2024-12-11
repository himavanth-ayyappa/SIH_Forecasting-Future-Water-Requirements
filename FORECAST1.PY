import pandas as pd

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset file
df = pd.read_csv('your_dataset.csv')

# Checking the first few rows of the data to understand its structure
print("Initial Data:")
print(df.head())

# Convert the columns to numeric, coercing errors (non-numeric values) to NaN
df['1991'] = pd.to_numeric(df['1991'], errors='coerce')
df['2001'] = pd.to_numeric(df['2001'], errors='coerce')
df['2011'] = pd.to_numeric(df['2011'], errors='coerce')
df['2022'] = pd.to_numeric(df['2022'], errors='coerce')

# Print the data types of the columns before cleaning
print("\nData types before cleaning:")
print(df.dtypes)

# Check for missing values (NaNs) in the dataset
print("\nMissing values per column:")
print(df.isna().sum())

# Fill missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

# Verify if missing values are filled
print("\nMissing values after filling with mean:")
print(df.isna().sum())

# Print the data types of the columns after cleaning
print("\nData types after cleaning:")
print(df.dtypes)

# Save the cleaned data to a new CSV file (optional)
df.to_csv('cleaned_dataset.csv', index=False)

# Continue with your forecasting or analysis process
# For example, if you're making a forecast using a model, add that code below:
# Assuming you have some forecasting model here...
# forecast = your_model.predict(df)

# Print the cleaned data to confirm it is processed
print("\nCleaned Data:")
print(df.head())
