import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind
from statsmodels.tsa.stattools import grangercausalitytests

# Load emotion scores
with open('./datasets/average_yearly_negative_scores.json') as f:
    negative_scores = json.load(f)
with open('./datasets/average_yearly_positive_scores.json') as f:
    positive_scores = json.load(f)
with open('./datasets/average_yearly_overall_scores.json') as f:
    overall_scores = json.load(f)

# Load mental health data
mental_health_data = pd.read_csv('./datasets/mental-health-yearly-score-USA.csv')

# Filter data for the years 2010 to 2017
years = list(range(2010, 2018))
negative_scores = {int(year): negative_scores[str(year)] for year in years}
positive_scores = {int(year): positive_scores[str(year)] for year in years}
overall_scores = {int(year): overall_scores[str(year)] for year in years}
mental_health_data = mental_health_data[mental_health_data['Year'].isin(years)]

# Convert emotion scores to DataFrame
emotion_scores_df = pd.DataFrame({
    'Year': years,
    'Positive': [positive_scores[year] for year in years],
    'Negative': [negative_scores[year] for year in years],
    'Overall': [overall_scores[year] for year in years]
})

# Merge with mental health data
merged_data = pd.merge(emotion_scores_df, mental_health_data, on='Year')

# Correlation Analysis
correlations = {}
for column in merged_data.columns[4:]:
    correlations[column] = {
        'Positive': pearsonr(merged_data['Positive'], merged_data[column]),
        'Negative': pearsonr(merged_data['Negative'], merged_data[column]),
        'Overall': pearsonr(merged_data['Overall'], merged_data[column])
    }

# Print Correlation Results
print("Correlation Results (2010-2017):")
for disorder, scores in correlations.items():
    print(f"{disorder}:")
    for score_type, corr in scores.items():
        print(f"  {score_type} - Pearson r: {corr[0]:.3f}, p-value: {corr[1]:.3f}")

# Visualization with Normalization
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# Normalize emotion scores
merged_data['Positive_norm'] = normalize(merged_data['Positive'])
merged_data['Negative_norm'] = normalize(merged_data['Negative'])
merged_data['Overall_norm'] = normalize(merged_data['Overall'])

# Plot normalized emotion scores
plt.figure(figsize=(12, 6))
plt.plot(merged_data['Year'], merged_data['Positive_norm'], label='Positive Emotion Score (Normalized)')
plt.plot(merged_data['Year'], merged_data['Negative_norm'], label='Negative Emotion Score (Normalized)')
plt.plot(merged_data['Year'], merged_data['Overall_norm'], label='Overall Emotion Score (Normalized)')
plt.xlabel("Year")
plt.ylabel("Normalized Emotion Score")
plt.title("Normalized Emotion Scores by Year (2010-2017)")
plt.legend()
plt.grid(True)
plt.savefig('./Figures/normalized_emotion_scores_2010_2017.png')
plt.show()

# Plot normalized mental health trends with emotion scores (2010-2017)
for column in merged_data.columns[4:]:
    merged_data[f'{column}_norm'] = normalize(merged_data[column])
    plt.figure(figsize=(12, 6))
    plt.plot(merged_data['Year'], merged_data[f'{column}_norm'], label=f'{column} (Normalized)', color='purple')
    plt.plot(merged_data['Year'], merged_data['Negative_norm'], label='Negative Emotion Score (Normalized)', color='red')
    plt.xlabel("Year")
    plt.ylabel("Normalized Rate")
    plt.title(f"Normalized {column} vs Negative Emotion Score (2010-2017)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./Figures/normalized_{column}_vs_negative_emotion_2010_2017.png')
    plt.show()


"""
Terminal Output:

Correlation Results (2010-2017):
Schizophrenia (%):
  Positive - Pearson r: 0.795, p-value: 0.018
  Negative - Pearson r: -0.837, p-value: 0.010
  Overall - Pearson r: 0.835, p-value: 0.010
Bipolar disorder (%):
  Positive - Pearson r: 0.783, p-value: 0.022
  Negative - Pearson r: -0.838, p-value: 0.009
  Overall - Pearson r: 0.830, p-value: 0.011
Eating disorders (%):
  Positive - Pearson r: 0.793, p-value: 0.019
  Negative - Pearson r: -0.837, p-value: 0.010
  Overall - Pearson r: 0.835, p-value: 0.010
Anxiety disorders (%):
  Positive - Pearson r: 0.771, p-value: 0.025
  Negative - Pearson r: -0.826, p-value: 0.012
  Overall - Pearson r: 0.818, p-value: 0.013
Drug use disorders (%):
  Positive - Pearson r: -0.792, p-value: 0.019
  Negative - Pearson r: 0.838, p-value: 0.009
  Overall - Pearson r: -0.834, p-value: 0.010
Depression (%):
  Positive - Pearson r: -0.778, p-value: 0.023
  Negative - Pearson r: 0.822, p-value: 0.012
  Overall - Pearson r: -0.819, p-value: 0.013
Alcohol use disorders (%):
  Positive - Pearson r: -0.778, p-value: 0.023
  Negative - Pearson r: 0.826, p-value: 0.012
  Overall - Pearson r: -0.821, p-value: 0.012
"""