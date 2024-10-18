import pandas as pd
import os

# Define file paths
ts1_path = 'C:/Users/Chopp/Desktop/1.csv'  # High-frequency transaction data
ts2_path = 'C:/Users/Chopp/Desktop/2.csv'  # Minute-based OHLC data
output_path = 'C:/Users/Chopp/Desktop/3.csv'  # Final output path renamed to '3.csv'

# Function to check file existence
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found at the specified path: {file_path}")

# Verify that both input files exist
check_file_exists(ts1_path)
check_file_exists(ts2_path)

# ------------------------ #
#       Process 1.csv      #
# ------------------------ #

# Load the high-frequency transaction data (1.csv)
try:
    ts1 = pd.read_csv(ts1_path)
    print("Loaded '1.csv' successfully with headers.\n")
except pd.errors.ParserError:
    # If parsing fails, read without headers and assign default column names
    ts1 = pd.read_csv(ts1_path, header=None)
    ts1.columns = ['Time', 'Last', 'Size', 'Aggressor']
    print("Loaded '1.csv' without headers. Assigned default column names.\n")

# Inspect the first few rows
print("First few rows of '1.csv':")
print(ts1.head(), "\n")


# Verify required columns exist
required_columns_ts1 = ['Time', 'Size', 'Aggressor']
for col in required_columns_ts1:
    if col not in ts1.columns:
        raise ValueError(f"Missing required column in '1.csv': '{col}'. Please check your CSV file.")

# Convert 'Time' column to datetime objects
ts1['Time'] = pd.to_datetime(ts1['Time'], errors='coerce')

# Check for and remove rows with invalid datetime formats
invalid_time_rows_ts1 = ts1['Time'].isnull().sum()
if invalid_time_rows_ts1 > 0:
    print(f"Warning: {invalid_time_rows_ts1} rows in '1.csv' have invalid datetime formats and will be excluded.\n")
    ts1 = ts1.dropna(subset=['Time'])

# Ensure 'Size' column is numeric
ts1['Size'] = pd.to_numeric(ts1['Size'], errors='coerce')

# Check for and remove rows with invalid 'Size' values
invalid_size_rows_ts1 = ts1['Size'].isnull().sum()
if invalid_size_rows_ts1 > 0:
    print(f"Warning: {invalid_size_rows_ts1} rows in '1.csv' have invalid 'Size' values and will be excluded.\n")
    ts1 = ts1.dropna(subset=['Size'])

# Convert 'Size' to integer type
ts1['Size'] = ts1['Size'].astype(int)

# Floor the 'Time' to the nearest minute to create 'Minute' column
ts1['Minute'] = ts1['Time'].dt.floor('T')  # 'T' stands for minute

# Ensure 'Aggressor' column is string type and clean it
ts1['Aggressor'] = ts1['Aggressor'].astype(str).str.strip().str.lower()

# Aggregate transactions by minute
aggregated_ts1 = ts1.groupby('Minute').agg(
    Total_Transactions=('Size', 'sum'),
    Buys=('Size', lambda x: ts1.loc[x.index, 'Aggressor'].eq('buy').multiply(x).sum()),
    Sells=('Size', lambda x: ts1.loc[x.index, 'Aggressor'].eq('sell').multiply(x).sum())
).reset_index()

print("Aggregated transaction data from '1.csv':")
print(aggregated_ts1.head(), "\n")

import pandas as pd
import os

# Define file paths
ts1_path = 'C:/Users/Chopp/Desktop/1.csv'  # High-frequency transaction data
ts2_path = 'C:/Users/Chopp/Desktop/2.csv'  # Minute-based OHLC data
output_path = 'C:/Users/Chopp/Desktop/3.csv'  # Final output path renamed to '3.csv'

# Function to check file existence
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found at the specified path: {file_path}")

# Verify that both input files exist
check_file_exists(ts1_path)
check_file_exists(ts2_path)

# ------------------------ #
#       Process 1.csv      #
# ------------------------ #

# Load the high-frequency transaction data (1.csv)
try:
    ts1 = pd.read_csv(ts1_path)
    print("Loaded '1.csv' successfully with headers.\n")
except pd.errors.ParserError:
    # If parsing fails, read without headers and assign default column names
    ts1 = pd.read_csv(ts1_path, header=None)
    ts1.columns = ['Time', 'Last', 'Size', 'Aggressor']
    print("Loaded '1.csv' without headers. Assigned default column names.\n")

# Inspect the first few rows
print("First few rows of '1.csv':")
print(ts1.head(), "\n")


# Verify required columns exist
required_columns_ts1 = ['Time', 'Size', 'Aggressor']
for col in required_columns_ts1:
    if col not in ts1.columns:
        raise ValueError(f"Missing required column in '1.csv': '{col}'. Please check your CSV file.")

# Convert 'Time' column to datetime objects
ts1['Time'] = pd.to_datetime(ts1['Time'], errors='coerce')

# Check for and remove rows with invalid datetime formats
invalid_time_rows_ts1 = ts1['Time'].isnull().sum()
if invalid_time_rows_ts1 > 0:
    print(f"Warning: {invalid_time_rows_ts1} rows in '1.csv' have invalid datetime formats and will be excluded.\n")
    ts1 = ts1.dropna(subset=['Time'])

# Ensure 'Size' column is numeric
ts1['Size'] = pd.to_numeric(ts1['Size'], errors='coerce')

# Check for and remove rows with invalid 'Size' values
invalid_size_rows_ts1 = ts1['Size'].isnull().sum()
if invalid_size_rows_ts1 > 0:
    print(f"Warning: {invalid_size_rows_ts1} rows in '1.csv' have invalid 'Size' values and will be excluded.\n")
    ts1 = ts1.dropna(subset=['Size'])

# Convert 'Size' to integer type
ts1['Size'] = ts1['Size'].astype(int)

# Floor the 'Time' to the nearest minute to create 'Minute' column
ts1['Minute'] = ts1['Time'].dt.floor('T')  # 'T' stands for minute

# Ensure 'Aggressor' column is string type and clean it
ts1['Aggressor'] = ts1['Aggressor'].astype(str).str.strip().str.lower()

# Aggregate transactions by minute
aggregated_ts1 = ts1.groupby('Minute').agg(
    Total_Transactions=('Size', 'sum'),
    Buys=('Size', lambda x: ts1.loc[x.index, 'Aggressor'].eq('buy').multiply(x).sum()),
    Sells=('Size', lambda x: ts1.loc[x.index, 'Aggressor'].eq('sell').multiply(x).sum())
).reset_index()

print("Aggregated transaction data from '1.csv':")
print(aggregated_ts1.head(), "\n")

# ------------------------ #
#       Process 2.csv      #
# ------------------------ #

# Load the minute-based OHLC data (2.csv)
try:
    ts2 = pd.read_csv(ts2_path)
    print("Loaded '2.csv' successfully with headers.\n")
except pd.errors.ParserError:
    # If parsing fails, read without headers and assign default column names
    ts2 = pd.read_csv(ts2_path, header=None)
    ts2.columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume(from bar)']
    print("Loaded '2.csv' without headers. Assigned default column names.\n")
    
    # Rename 'Volume(from bar)' to 'Volume'
    if 'Volume(from bar)' in ts2.columns:
        ts2.rename(columns={'Volume(from bar)': 'Volume'}, inplace=True)
        print("Renamed 'Volume(from bar)' to 'Volume' in '2.csv'.\n")
    else:
        print("'Volume(from bar)' column not found in '2.csv'. No renaming applied.\n")

# Inspect the first few rows
print("First few rows of '2.csv':")
print(ts2.head(), "\n")

# Remove any empty or unnecessary columns (e.g., 'Unnamed: 5')
ts2 = ts2.loc[:, ~ts2.columns.str.contains('^Unnamed|Empty_Column')]
print("Columns after removing empty/unnecessary columns in '2.csv':")
print(ts2.columns.tolist(), "\n")

# Rename 'Volume(from bar)' to 'Volume' if not already renamed
if 'Volume(from bar)' in ts2.columns:
    ts2.rename(columns={'Volume(from bar)': 'Volume'}, inplace=True)
    print("Renamed 'Volume(from bar)' to 'Volume' in '2.csv'.\n")
else:
    print("'Volume(from bar)' column already renamed or not present.\n")

# Verify required columns exist
required_columns_ts2 = ['DateTime']
for col in required_columns_ts2:
    if col not in ts2.columns:
        raise ValueError(f"Missing required column in '2.csv': '{col}'. Please check your CSV file.")

# Convert 'DateTime' column to datetime objects
ts2['DateTime'] = pd.to_datetime(ts2['DateTime'], errors='coerce')

# Check for and remove rows with invalid datetime formats
invalid_datetime_ts2 = ts2['DateTime'].isnull().sum()
if invalid_datetime_ts2 > 0:
    print(f"Warning: {invalid_datetime_ts2} rows in '2.csv' have invalid datetime formats and will be excluded.\n")
    ts2 = ts2.dropna(subset=['DateTime'])

# Floor the 'DateTime' to the nearest minute to ensure alignment
ts2['DateTime'] = ts2['DateTime'].dt.floor('T')

# ------------------------ #
#          Merge           #
# ------------------------ #

# Merge the aggregated transaction data with the OHLC data on the minute
merged_data = pd.merge(ts2, aggregated_ts1, left_on='DateTime', right_on='Minute', how='left')

print("First few rows of the merged data before filling missing values:")
print(merged_data.head(), "\n")

# Fill any missing 'Total_Transactions', 'Buys', or 'Sells' with 0
merged_data['Total_Transactions'] = merged_data['Total_Transactions'].fillna(0).astype(int)
merged_data['Buys'] = merged_data['Buys'].fillna(0).astype(int)
merged_data['Sells'] = merged_data['Sells'].fillna(0).astype(int)

# Drop the 'Minute' column as it's redundant after merging
merged_data = merged_data.drop(columns=['Minute'])

print("First few rows of the merged data after filling missing values:")
print(merged_data.head(), "\n")

# ------------------------ #
#         Clean Data       #
# ------------------------ #

# Remove the date part from the 'DateTime' column and keep only the time
merged_data['DateTime'] = pd.to_datetime(merged_data['DateTime']).dt.strftime('%H:%M:%S')

# Check if the "Unnamed: 5" column exists, and remove it if present
if 'Unnamed: 5' in merged_data.columns:
    merged_data.drop(columns=['Unnamed: 5'], inplace=True)
    print("'Unnamed: 5' column removed.\n")
else:
    print("'Unnamed: 5' column was not found, no need to remove it.\n")

# Rename 'Buys' and 'Sells' to 'buyers' and 'sellers'
merged_data.rename(columns={'Buys': 'buyers', 'Sells': 'sellers'}, inplace=True)

print("First few rows of the merged and cleaned data with renamed columns:")
print(merged_data.head(), "\n")

# ------------------------ #
#          Save            #
# ------------------------ #

# Save the final cleaned and merged dataset to '3.csv'
merged_data.to_csv(output_path, index=False)

print(f"Final data cleaned and saved successfully to {output_path}")


# ------------------------ #
#          Merge           #
# ------------------------ #

# Merge the aggregated transaction data with the OHLC data on the minute
merged_data = pd.merge(ts2, aggregated_ts1, left_on='DateTime', right_on='Minute', how='left')

print("First few rows of the merged data before filling missing values:")
print(merged_data.head(), "\n")

# Fill any missing 'Total_Transactions', 'Buys', or 'Sells' with 0
merged_data['Total_Transactions'] = merged_data['Total_Transactions'].fillna(0).astype(int)
merged_data['Buys'] = merged_data['Buys'].fillna(0).astype(int)
merged_data['Sells'] = merged_data['Sells'].fillna(0).astype(int)

# Drop the 'Minute' column as it's redundant after merging
merged_data = merged_data.drop(columns=['Minute'])

print("First few rows of the merged data after filling missing values:")
print(merged_data.head(), "\n")

# ------------------------ #
#         Clean Data       #
# ------------------------ #

# Remove the date part from the 'DateTime' column and keep only the time
merged_data['DateTime'] = pd.to_datetime(merged_data['DateTime']).dt.strftime('%H:%M:%S')

# Check if the "Unnamed: 5" column exists, and remove it if present
if 'Unnamed: 5' in merged_data.columns:
    merged_data.drop(columns=['Unnamed: 5'], inplace=True)
    print("'Unnamed: 5' column removed.\n")
else:
    print("'Unnamed: 5' column was not found, no need to remove it.\n")

# Rename 'Buys' and 'Sells' to 'buyers' and 'sellers'
merged_data.rename(columns={'Buys': 'buyers', 'Sells': 'sellers'}, inplace=True)

print("First few rows of the merged and cleaned data with renamed columns:")
print(merged_data.head(), "\n")

# ------------------------ #
#          Save            #
# ------------------------ #

# Save the final cleaned and merged dataset to '3.csv'
merged_data.to_csv(output_path, index=False)

print(f"Final data cleaned and saved successfully to {output_path}")
