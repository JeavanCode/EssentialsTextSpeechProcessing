import pandas as pd
import json
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from emotion_score_calculator import EmotionScoreCalculator
from transformers import GPT2Tokenizer

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to preprocess lyrics by removing special characters and symbols
def preprocess_lyrics(lyrics):
    lyrics = re.sub(r'[♪\n]', ' ', lyrics)  # Replace symbols and newline characters with a space
    lyrics = re.sub(r'[^a-zA-Z0-9\s]', '', lyrics)  # Keep only alphanumeric characters and spaces
    return lyrics.strip()

# Function to split text into chunks based on token limit and return each chunk's text and length
def split_into_chunks(text, max_tokens=1024):
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        yield tokenizer.decode(chunk_tokens), len(chunk_tokens)

# Load the dataset
file_path = "./datasets/lyrics_and_years.csv"
data = pd.read_csv(file_path)

# Filter data for years between 1990 and 2017
data = data[(data['year'] >= 1990) & (data['year'] <= 2017)]

# Initialize the emotion score calculator
calculator = EmotionScoreCalculator()

# Dictionary to store total emotion scores for each year
yearly_scores = {}

# Calculate the emotion score for each song and accumulate by year
print("Calculating emotion scores for each song...")
for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing songs"):
    lyrics = row['lyrics']
    year = row['year']
    
    # Preprocess the entire song lyrics
    preprocessed_lyrics = preprocess_lyrics(lyrics)
    
    # Split preprocessed lyrics into manageable chunks
    chunk_scores = []
    chunk_lengths = []
    for chunk, length in split_into_chunks(preprocessed_lyrics):
        chunk_score = calculator.calculate_emotion_score(chunk)
        chunk_scores.append(chunk_score)
        chunk_lengths.append(length)

    # Calculate weighted average score for the song based on chunk lengths
    if chunk_scores:
        song_score = sum(score * length for score, length in zip(chunk_scores, chunk_lengths)) / sum(chunk_lengths)
    else:
        song_score = 0

    # Add song's score to the corresponding year
    if year in yearly_scores:
        yearly_scores[year] += song_score
    else:
        yearly_scores[year] = song_score

# Normalize yearly scores by the number of songs in each year to get an average
year_counts = data['year'].value_counts().to_dict()  # Number of songs per year
average_yearly_scores = {year: yearly_scores[year] / year_counts[year] for year in yearly_scores}

# Save the results to a JSON file
output_json_path = "./datasets/yearly_emotion_scores.json"
with open(output_json_path, 'w') as f:
    json.dump(average_yearly_scores, f, indent=4)

print(f"Yearly emotion scores saved to {output_json_path}")

# Plotting the scores
plt.figure(figsize=(10, 6))
plt.plot(list(average_yearly_scores.keys()), list(average_yearly_scores.values()), marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Average Emotion Score')
plt.title('Average Emotion Score by Year (1990-2017)')
plt.grid(True)

# Save the plot
output_plot_path = "./Figures/yearly_emotion_scores_plot.png"
plt.savefig(output_plot_path)
plt.show()

print(f"Plot of yearly emotion scores saved to {output_plot_path}")
