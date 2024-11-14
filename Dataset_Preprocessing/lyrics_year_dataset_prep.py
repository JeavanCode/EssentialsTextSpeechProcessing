import pandas as pd
from tqdm import tqdm

# Enable tqdm for Pandas operations
tqdm.pandas()

# Load the English lyrics dataset and select 'id' and 'lyrics' columns with progress indicator
english_lyrics_path = "./datasets/songs_with_attributes_and_lyrics_english.csv"
print("Loading English lyrics dataset...")
english_lyrics_df = pd.read_csv(english_lyrics_path, usecols=['id', 'lyrics'])
print("English lyrics dataset loaded.")

# Load the 160k songs dataset and select 'id' and 'year' columns with progress indicator
songs_metadata_path = "./datasets/160k-songs-1921-2020.csv"
print("Loading 160k songs metadata dataset...")
songs_metadata_df = pd.read_csv(songs_metadata_path, usecols=['id', 'year'])
print("160k songs metadata dataset loaded.")

# Merge the datasets on the 'id' column with tqdm progress bar
print("Merging datasets on 'id' column...")
merged_df = pd.merge(english_lyrics_df.progress_apply(lambda x: x), songs_metadata_df.progress_apply(lambda x: x), on='id', how='inner')
print("Merging complete.")

# Save the result to a new CSV file
output_path = "./datasets/lyrics_and_years.csv"
merged_df.to_csv(output_path, index=False)

print(f"New dataset with lyrics and years saved to {output_path}")
print(merged_df.head())
