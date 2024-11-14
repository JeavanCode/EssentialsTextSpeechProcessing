import pandas as pd
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

# Ensure consistency in language detection results
DetectorFactory.seed = 0

# Load the dataset
file_path = "./datasets/songs_with_attributes_and_lyrics.csv"
data = pd.read_csv(file_path)

# Function to detect language with error handling
def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return None  # Return None if detection fails

# Apply language detection with tqdm progress bar
tqdm.pandas(desc="Detecting language")
data['language'] = data['lyrics'].progress_apply(lambda x: detect_language(x) if pd.notnull(x) else None)

# Filter for English lyrics
english_lyrics = data[data['language'] == 'en']

# Drop the temporary 'language' column if not needed
english_lyrics = english_lyrics.drop(columns=['language'])

# Save the filtered data to a new CSV file
output_path = "./datasets/songs_with_attributes_and_lyrics_english.csv"
english_lyrics.to_csv(output_path, index=False)

print(f"Filtered English lyrics dataset saved to {output_path}")

# Display the filtered data
print(f"Number of English songs: {english_lyrics.shape[0]}")
print(english_lyrics.head())