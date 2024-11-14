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

# Filter data for the years 1990 to 2017
years = list(range(1990, 2018))
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
print("Correlation Results:")
for disorder, scores in correlations.items():
    print(f"{disorder}:")
    for score_type, corr in scores.items():
        print(f"  {score_type} - Pearson r: {corr[0]:.3f}, p-value: {corr[1]:.3f}")

# Significance Testing (e.g., t-test between two time periods)
before_2000 = merged_data[merged_data['Year'] < 2000]['Negative']
after_2000 = merged_data[merged_data['Year'] >= 2000]['Negative']
ttest_results = ttest_ind(before_2000, after_2000)
print("\nT-test for Negative Scores (before vs after 2000):")
print(f"T-statistic: {ttest_results.statistic:.3f}, p-value: {ttest_results.pvalue:.3f}")

# Granger Causality Test
print("\nGranger Causality Test Results:")
grangercausalitytests(merged_data[['Negative', 'Depression (%)']], maxlag=2)

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
plt.title("Normalized Emotion Scores by Year (1990-2017)")
plt.legend()
plt.grid(True)
plt.savefig('./Figures/normalized_emotion_scores_1990_2017.png')
plt.show()

# Plot normalized mental health trends with emotion scores
for column in merged_data.columns[4:]:
    merged_data[f'{column}_norm'] = normalize(merged_data[column])
    plt.figure(figsize=(12, 6))
    plt.plot(merged_data['Year'], merged_data[f'{column}_norm'], label=f'{column} (Normalized)', color='purple')
    plt.plot(merged_data['Year'], merged_data['Negative_norm'], label='Negative Emotion Score (Normalized)', color='red')
    plt.xlabel("Year")
    plt.ylabel("Normalized Rate")
    plt.title(f"Normalized {column} vs Negative Emotion Score (1990-2017)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./Figures/normalized_{column}_vs_negative_emotion_1990_2017.png')
    plt.show()
"""
Terminal Output:

Correlation Results:
Schizophrenia (%):
  Positive - Pearson r: 0.154, p-value: 0.434
  Negative - Pearson r: -0.234, p-value: 0.230
  Overall - Pearson r: 0.226, p-value: 0.249
Bipolar disorder (%):
  Positive - Pearson r: 0.445, p-value: 0.018
  Negative - Pearson r: 0.182, p-value: 0.354
  Overall - Pearson r: 0.173, p-value: 0.380
Eating disorders (%):
  Positive - Pearson r: -0.571, p-value: 0.001
  Negative - Pearson r: -0.146, p-value: 0.459
  Overall - Pearson r: -0.271, p-value: 0.164
Anxiety disorders (%):
  Positive - Pearson r: -0.299, p-value: 0.123
  Negative - Pearson r: 0.012, p-value: 0.953
  Overall - Pearson r: -0.191, p-value: 0.331
Drug use disorders (%):
  Positive - Pearson r: -0.706, p-value: 0.000
  Negative - Pearson r: -0.050, p-value: 0.800
  Overall - Pearson r: -0.407, p-value: 0.032
Depression (%):
  Positive - Pearson r: -0.661, p-value: 0.000
  Negative - Pearson r: -0.086, p-value: 0.663
  Overall - Pearson r: -0.359, p-value: 0.061
Alcohol use disorders (%):
  Positive - Pearson r: 0.494, p-value: 0.007
  Negative - Pearson r: -0.095, p-value: 0.632
  Overall - Pearson r: 0.357, p-value: 0.062

T-test for Negative Scores (before vs after 2000):
T-statistic: 0.902, p-value: 0.376

Granger Causality Test Results:

Granger Causality
number of lags (no zero) 1
ssr based F test:         F=1.0741  , p=0.3103  , df_denom=24, df_num=1
ssr based chi2 test:   chi2=1.2084  , p=0.2716  , df=1
likelihood ratio test: chi2=1.1821  , p=0.2769  , df=1
parameter F test:         F=1.0741  , p=0.3103  , df_denom=24, df_num=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=1.7968  , p=0.1904  , df_denom=21, df_num=2
ssr based chi2 test:   chi2=4.4492  , p=0.1081  , df=2
likelihood ratio test: chi2=4.1071  , p=0.1283  , df=2
parameter F test:         F=1.7968  , p=0.1904  , df_denom=21, df_num=2

"""