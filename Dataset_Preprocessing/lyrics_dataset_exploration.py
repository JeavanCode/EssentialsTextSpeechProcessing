import pandas as pd

# Load the dataset
file_path = "./datasets/songs_with_attributes_and_lyrics.csv"
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
RangeIndex: 955320 entries, 0 to 955319
Data columns (total 17 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   id                955320 non-null  object 
 1   name              955309 non-null  object 
 2   album_name        385557 non-null  object 
 3   artists           955318 non-null  object 
 4   danceability      955320 non-null  float64
 5   energy            955320 non-null  float64
 6   key               955320 non-null  object 
 7   loudness          955320 non-null  float64
 8   mode              955320 non-null  object 
 9   speechiness       955320 non-null  float64
 10  acousticness      955320 non-null  float64
 11  instrumentalness  955320 non-null  float64
 12  liveness          955320 non-null  float64
 13  valence           955320 non-null  float64
 14  tempo             955320 non-null  float64
 15  duration_ms       955320 non-null  float64
 16  lyrics            955307 non-null  object 
dtypes: float64(10), object(7)
memory usage: 123.9+ MB

Column Names:
['id', 'name', 'album_name', 'artists', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'lyrics']

First 5 Rows:
                       id             name                              album_name  ...    tempo  duration_ms                                             lyrics
0  0Prct5TDjAnEgIqbxcldY9                !                              UNDEN!ABLE  ...  100.059      79500.0  He said he came from Jamaica,\n he owned a cou...
1  2ASl4wirkeYm3OWZxXKYuq               !!                                     NaN  ...   79.998     114000.0  Fucked a bitch, now she running with my kids\n...
2  69lcggVPmOr9cvPx9kLiiN  !!! - Interlude                       Where I Belong EP  ...    0.000      11413.0                     Oh, my God, I'm going crazy\n 
3  4U7dlZjg1s9pjdppqZy0fm   !!De Repente!!  Un Palo Al Agua (20 Grandes Canciones)  ...  123.588     198173.0  Continuamente se extraña la gente si no puede ...
4  4v1IBp3Y3rpkWmWzIlkYju   !!De Repente!!                          Fuera De Lugar  ...  123.600     199827.0  Continuamente se extraña la gente si no puede ...

[5 rows x 17 columns]

Number of Rows: 955320
Number of Columns: 17"""