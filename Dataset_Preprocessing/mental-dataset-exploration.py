import pandas as pd

# Load the dataset
file_path = "./datasets/mental-health-disorders-data.csv"
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
RangeIndex: 108553 entries, 0 to 108552
Data columns (total 11 columns):
 #   Column                     Non-Null Count   Dtype  
---  ------                     --------------   -----  
 0   index                      108553 non-null  int64  
 1   Entity                     108553 non-null  object 
 2   Code                       103141 non-null  object 
 3   Year                       108553 non-null  object 
 4   Schizophrenia (%)          25875 non-null   object 
 5   Bipolar disorder (%)       19406 non-null   object 
 6   Eating disorders (%)       100236 non-null  object 
 7   Anxiety disorders (%)      6468 non-null    float64
 8   Drug use disorders (%)     6468 non-null    float64
 9   Depression (%)             6468 non-null    float64
 10  Alcohol use disorders (%)  6468 non-null    float64
dtypes: float64(4), int64(1), object(6)
memory usage: 9.1+ MB

Column Names:
['index', 'Entity', 'Code', 'Year', 'Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 'Alcohol use disorders (%)']

First 5 Rows:
   index       Entity Code  Year  ... Anxiety disorders (%) Drug use disorders (%) Depression (%)  Alcohol use disorders (%)
0      0  Afghanistan  AFG  1990  ...              4.828830               1.677082       4.071831                   0.672404
1      1  Afghanistan  AFG  1991  ...              4.829740               1.684746       4.079531                   0.671768
2      2  Afghanistan  AFG  1992  ...              4.831108               1.694334       4.088358                   0.670644
3      3  Afghanistan  AFG  1993  ...              4.830864               1.705320       4.096190                   0.669738
4      4  Afghanistan  AFG  1994  ...              4.829423               1.716069       4.099582                   0.669260

[5 rows x 11 columns]

Number of Rows: 108553
Number of Columns: 11
"""