import pandas as pd

# Load the dataset
file_path = "./datasets/lyrics_and_years.csv"
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
RangeIndex: 90647 entries, 0 to 90646
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   id      90647 non-null  object
 1   lyrics  90647 non-null  object
 2   year    90647 non-null  int64 
dtypes: int64(1), object(2)
memory usage: 2.1+ MB

Column Names:
['id', 'lyrics', 'year']

First 5 Rows:
                       id                                             lyrics  year
0  0gNNToCW3qjabgTyBSjt3H  With pictures and words\n Is this communicatin...  1966
1  0tMgFpOrXZR6irEOLNWwJL  I waited patiently for the Lord\n He inclined ...  1983
2  2ZywW3VyVx6rrlrX75n3JB  I waited patiently for the Lord\n He inclined ...  1983
3  3vMmwsAiLDCfyc1jl76lQE  Two, three, four\n ♪\n I waited patiently for ...  1983
4  6DdWA7D1o5TU2kXWyCLcch  Two, three, four\n ♪\n I waited patiently for ...  1983

Number of Rows: 90647
Number of Columns: 3

"""