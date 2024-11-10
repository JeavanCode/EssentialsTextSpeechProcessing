import pandas as pd

# Load the mental health disorders dataset
file_path = "./datasets/mental-health-disorders-data.csv"
data = pd.read_csv(file_path)

# Filter data to include only rows where the Entity is 'United States'
data = data[data['Entity'] == 'United States']

# Convert the 'Year' column to numeric, dropping rows where conversion fails
data = data[pd.to_numeric(data['Year'], errors='coerce').notna()]
data['Year'] = data['Year'].astype(int)  # Convert to integer

# Convert columns with numeric data stored as objects to float
for col in ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 
            'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 
            'Alcohol use disorders (%)']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with any missing values in any column
data = data.dropna()

# Filter data for years between 1990 and 2017
data = data[(data['Year'] >= 1990) & (data['Year'] <= 2017)]

# Group by 'Year' and calculate the mean for each disorder column
yearly_scores = data.groupby('Year').mean(numeric_only=True).reset_index()

# Convert 'Year' column to integer
yearly_scores['Year'] = yearly_scores['Year'].astype(int)

# Select only the relevant columns for the output
columns_to_keep = ['Year', 'Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 
                   'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 
                   'Alcohol use disorders (%)']
yearly_scores = yearly_scores[columns_to_keep]

# Save the result to a new CSV file
output_path = "./datasets/mental-health-yearly-score-USA.csv"
yearly_scores.to_csv(output_path, index=False)

print(f"Yearly mental health scores dataset for United States saved to {output_path}")
print(yearly_scores.head())
