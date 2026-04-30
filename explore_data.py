import pandas as pd

# 1. Load CSV with pandas
file_path = "lead101_realtime_5000.csv"
df = pd.read_csv(file_path)

# 2. Print: shape, column names, dtypes
print("--- DataFrame Shape ---")
print(df.shape)
print("\n--- Column Names ---")
print(df.columns.tolist())
print("\n--- Data Types ---")
print(df.dtypes)

# 3. Check for missing values (nulls) per column
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 4. Show first 10 rows
print("\n--- First 10 Rows ---")
print(df.head(10))

# 5. Basic statistics: describe() for numerical columns
print("\n--- Basic Statistics ---")
print(df.describe())
