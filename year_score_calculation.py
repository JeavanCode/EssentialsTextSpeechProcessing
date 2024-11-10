import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from yearly_emotion_score_calculator import YearlyEmotionScoreCalculator

# Load the dataset
file_path = "./datasets/lyrics_and_years.csv"
data = pd.read_csv(file_path)

# Filter data for years between 1990 and 2017
data = data[(data['year'] >= 1990) & (data['year'] <= 2017)]

# Initialize the emotion score calculator
calculator = YearlyEmotionScoreCalculator()

# Dictionary to store total emotion scores for each year
yearly_scores = {}

# Initialize tqdm for the main loop
print("Calculating emotion scores for each song...")
for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing songs"):
    lyrics = row['lyrics']
    year = row['year']
    
    # Calculate score for each line in the lyrics and sum up for the song
    song_score = sum([calculator.calculate_emotion_score(line) for line in lyrics.splitlines() if line.strip()])
    
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
