import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
file_path = "./datasets/mental-health-yearly-score-USA.csv"
data = pd.read_csv(file_path)

# Ensure the 'Year' column is integer
data['Year'] = data['Year'].astype(int)

# Filter the data to include only years from 1990 to 2017
data = data[(data['Year'] >= 1990) & (data['Year'] <= 2017)]

# Directory to save figures
output_dir = "./Figures"
os.makedirs(output_dir, exist_ok=True)

# List of mental health disorders to plot
disorders = ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 
             'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 
             'Alcohol use disorders (%)']

# Plot each disorder's trend over time and save the plot
for disorder in disorders:
    plt.figure(figsize=(10, 6))
    plt.plot(data['Year'], data[disorder], marker='o', linestyle='-')
    plt.xlabel('Year')
    plt.ylabel(f'{disorder} Percentage')
    plt.title(f'Trend of {disorder} from 1990 to 2017')
    plt.grid(True)
    
    # Save plot
    filename = disorder.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'percent') + ".png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()  # Close the plot to free memory
    
    print(f"Saved plot for {disorder} as {output_path}")
