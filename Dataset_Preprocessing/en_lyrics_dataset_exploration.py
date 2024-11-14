import pandas as pd

# Load the dataset
file_path = "./datasets/songs_with_attributes_and_lyrics_english.csv"
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
RangeIndex: 713606 entries, 0 to 713605
Data columns (total 17 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   id                713606 non-null  object 
 1   name              713600 non-null  object 
 2   album_name        295782 non-null  object 
 3   artists           713606 non-null  object 
 4   danceability      713606 non-null  float64
 5   energy            713606 non-null  float64
 6   key               713606 non-null  object 
 7   loudness          713606 non-null  float64
 8   mode              713606 non-null  object 
 9   speechiness       713606 non-null  float64
 10  acousticness      713606 non-null  float64
 11  instrumentalness  713606 non-null  float64
 12  liveness          713606 non-null  float64
 13  valence           713606 non-null  float64
 14  tempo             713606 non-null  float64
 15  duration_ms       713606 non-null  float64
 16  lyrics            713606 non-null  object 
dtypes: float64(10), object(7)
memory usage: 92.6+ MB

Column Names:
['id', 'name', 'album_name', 'artists', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'lyrics']

First 5 Rows:
                       id                 name      album_name         artists  ...  valence    tempo duration_ms                                             lyrics
0  0Prct5TDjAnEgIqbxcldY9                    !      UNDEN!ABLE    ['HELLYEAH']  ...    0.193  100.059     79500.0  He said he came from Jamaica,\n he owned a cou...
1  2ASl4wirkeYm3OWZxXKYuq                   !!             NaN         Yxngxr1  ...    0.287   79.998    114000.0  Fucked a bitch, now she running with my kids\n...
2  5tA3ImW310llKo8EMBj2Ga  !!Noble Stabbings!!             NaN  Dillinger Four  ...    0.349  175.317    197400.0  You like to stand on the other side\n Point an...
3  0fROT4kK5oTm8xO8PX6EJF       !I'll Be Back!  !I'll Be Back!           Rilès  ...    0.688  142.959    178533.0  It's been a while, shit I missed the rehab, ps...
4  1xBFhv5faebv3mmwxx7DnS               !Lost!             NaN           Rilès  ...    0.380   86.103    186197.0  I would like to give you all my time\n I would...

[5 rows x 17 columns]

Number of Rows: 713606
Number of Columns: 17
"""