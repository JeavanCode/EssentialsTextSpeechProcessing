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

# Dictionaries to store total positive and negative emotion scores for each year
yearly_positive_scores = {}
yearly_negative_scores = {}

# Calculate the emotion score for each song and accumulate by year
print("Calculating emotion scores for each song...")
for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing songs"):
    lyrics = row['lyrics']
    year = row['year']
    
    # Preprocess the entire song lyrics
    preprocessed_lyrics = preprocess_lyrics(lyrics)
    
    # Split preprocessed lyrics into manageable chunks
    positive_chunk_scores = []
    negative_chunk_scores = []
    chunk_lengths = []
    
    for chunk, length in split_into_chunks(preprocessed_lyrics):
        positive_score, negative_score = calculator.calculate_emotion_score(chunk)
        positive_chunk_scores.append(positive_score)
        negative_chunk_scores.append(negative_score)
        chunk_lengths.append(length)

    # Calculate weighted average scores for positive and negative emotions
    if chunk_lengths:
        positive_song_score = sum(score * length for score, length in zip(positive_chunk_scores, chunk_lengths)) / sum(chunk_lengths)
        negative_song_score = sum(score * length for score, length in zip(negative_chunk_scores, chunk_lengths)) / sum(chunk_lengths)
    else:
        positive_song_score = 0
        negative_song_score = 0

    # Add song's scores to the corresponding year
    if year in yearly_positive_scores:
        yearly_positive_scores[year] += positive_song_score
        yearly_negative_scores[year] += negative_song_score
    else:
        yearly_positive_scores[year] = positive_song_score
        yearly_negative_scores[year] = negative_song_score

# Normalize yearly scores by the number of songs in each year to get an average
year_counts = data['year'].value_counts().to_dict()  # Number of songs per year
average_yearly_positive_scores = {year: yearly_positive_scores[year] / year_counts[year] for year in yearly_positive_scores}
average_yearly_negative_scores = {year: yearly_negative_scores[year] / year_counts[year] for year in yearly_negative_scores}

# Calculate the overall score as positive score minus negative score
average_yearly_overall_scores = {year: average_yearly_positive_scores[year] - average_yearly_negative_scores[year] for year in average_yearly_positive_scores}

# Sort the years in ascending order for all score dictionaries
average_yearly_positive_scores = dict(sorted(average_yearly_positive_scores.items()))
average_yearly_negative_scores = dict(sorted(average_yearly_negative_scores.items()))
average_yearly_overall_scores = dict(sorted(average_yearly_overall_scores.items()))

# Save the results to JSON files
output_positive_json = "./datasets/average_yearly_positive_scores.json"
output_negative_json = "./datasets/average_yearly_negative_scores.json"
output_overall_json = "./datasets/average_yearly_overall_scores.json"

with open(output_positive_json, 'w') as f:
    json.dump(average_yearly_positive_scores, f, indent=4)
with open(output_negative_json, 'w') as f:
    json.dump(average_yearly_negative_scores, f, indent=4)
with open(output_overall_json, 'w') as f:
    json.dump(average_yearly_overall_scores, f, indent=4)

print(f"Positive scores saved to {output_positive_json}")
print(f"Negative scores saved to {output_negative_json}")
print(f"Overall scores saved to {output_overall_json}")

# Plotting each score separately and saving as distinct figures
# Plot Positive Scores
plt.figure(figsize=(10, 6))
plt.plot(list(average_yearly_positive_scores.keys()), list(average_yearly_positive_scores.values()), marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Average Positive Emotion Score')
plt.title('Average Positive Emotion Score by Year (1990-2017)')
plt.grid(True)
output_positive_plot_path = "./Figures/average_yearly_positive_emotion_scores_plot.png"
plt.savefig(output_positive_plot_path)
plt.show()
print(f"Positive emotion scores plot saved to {output_positive_plot_path}")

# Plot Negative Scores
plt.figure(figsize=(10, 6))
plt.plot(list(average_yearly_negative_scores.keys()), list(average_yearly_negative_scores.values()), marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Average Negative Emotion Score')
plt.title('Average Negative Emotion Score by Year (1990-2017)')
plt.grid(True)
output_negative_plot_path = "./Figures/average_yearly_negative_emotion_scores_plot.png"
plt.savefig(output_negative_plot_path)
plt.show()
print(f"Negative emotion scores plot saved to {output_negative_plot_path}")

# Plot Overall Scores
plt.figure(figsize=(10, 6))
plt.plot(list(average_yearly_overall_scores.keys()), list(average_yearly_overall_scores.values()), marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Average Overall Emotion Score')
plt.title('Average Overall Emotion Score by Year (1990-2017)')
plt.grid(True)
output_overall_plot_path = "./Figures/average_yearly_overall_emotion_scores_plot.png"
plt.savefig(output_overall_plot_path)
plt.show()
print(f"Overall emotion scores plot saved to {output_overall_plot_path}")