import pandas as pd

# Load the dataset
file_path = "./datasets/mental-health-yearly-score-USA.csv"
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Info:")
data.info()  # Shows data types, non-null counts, and memory usage

# Display column names
print("\nColumn Names:")
print(data.columns.tolist())

# Display the first few rows
print("\nFirst 5 Rows:")
print(data.head())

# Show the number of rows and columns
num_rows, num_columns = data.shape
print(f"\nNumber of Rows: {num_rows}")
print(f"Number of Columns: {num_columns}")

"""

# Terminal output of this script

Dataset Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 28 entries, 0 to 27
Data columns (total 8 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   Year                       28 non-null     int64  
 1   Schizophrenia (%)          28 non-null     float64
 2   Bipolar disorder (%)       28 non-null     float64
 3   Eating disorders (%)       28 non-null     float64
 4   Anxiety disorders (%)      28 non-null     float64
 5   Drug use disorders (%)     28 non-null     float64
 6   Depression (%)             28 non-null     float64
 7   Alcohol use disorders (%)  28 non-null     float64
dtypes: float64(7), int64(1)
memory usage: 1.9 KB

Column Names:
['Year', 'Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 'Alcohol use disorders (%)']

First 5 Rows:
   Year  Schizophrenia (%)  ...  Depression (%)  Alcohol use disorders (%)
0  1990           0.340897  ...        4.677591                   2.173751
1  1991           0.338913  ...        4.660871                   2.139292
2  1992           0.337343  ...        4.651949                   2.107931
3  1993           0.336207  ...        4.648701                   2.080669
4  1994           0.335536  ...        4.649294                   2.058632

[5 rows x 8 columns]

Number of Rows: 28
Number of Columns: 8
"""