import pandas as pd

# Load the dataset
file_path = "./datasets/160k-songs-1921-2020.csv"
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
RangeIndex: 169907 entries, 0 to 169906
Data columns (total 19 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   id                169907 non-null  object 
 1   name              169907 non-null  object 
 2   artists           169907 non-null  object 
 3   duration_ms       169907 non-null  int64  
 4   release_date      169907 non-null  object 
 5   year              169907 non-null  int64  
 6   acousticness      169907 non-null  float64
 7   danceability      169907 non-null  float64
 8   energy            169907 non-null  float64
 9   instrumentalness  169907 non-null  float64
 10  liveness          169907 non-null  float64
 11  loudness          169907 non-null  float64
 12  speechiness       169907 non-null  float64
 13  tempo             169907 non-null  float64
 14  valence           169907 non-null  float64
 15  mode              169907 non-null  int64  
 16  key               169907 non-null  int64  
 17  popularity        169907 non-null  int64  
 18  explicit          169907 non-null  int64  
dtypes: float64(9), int64(6), object(4)
memory usage: 24.6+ MB

Column Names:
['id', 'name', 'artists', 'duration_ms', 'release_date', 'year', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'mode', 'key', 'popularity', 'explicit']

First 5 Rows:
                       id                       name   artists  duration_ms release_date  ...  valence  mode  key  popularity  explicit
0  0gNNToCW3qjabgTyBSjt3H  !Que Vida! - Mono Version  ['Love']       220560      11/1/66  ...    0.547     1    9          26         0
1  0tMgFpOrXZR6irEOLNWwJL                       "40"    ['U2']       157840      2/28/83  ...    0.338     1    8          21         0
2  2ZywW3VyVx6rrlrX75n3JB                "40" - Live    ['U2']       226200      8/20/83  ...    0.279     1    8          41         0
3  6DdWA7D1o5TU2kXWyCLcch     "40" - Remastered 2008    ['U2']       157667      2/28/83  ...    0.310     1    8          37         0
4  3vMmwsAiLDCfyc1jl76lQE     "40" - Remastered 2008    ['U2']       157667      2/28/83  ...    0.310     1    8          35         0

[5 rows x 19 columns]

Number of Rows: 169907
Number of Columns: 19"""