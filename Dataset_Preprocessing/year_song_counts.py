import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "./datasets/lyrics_and_years.csv"
data = pd.read_csv(file_path)

# Count the number of rows for each year and sort by year
year_counts = data['year'].value_counts().sort_index()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(year_counts.index, year_counts.values, marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Number of Songs')
plt.title('Number of Songs per Year')
plt.grid(True)
plt.show()
