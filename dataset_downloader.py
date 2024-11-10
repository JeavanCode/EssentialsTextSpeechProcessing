"""
# This script downloads the datasets that will be used in this project.

import kagglehub

# Download 160k songs datasets
path = kagglehub.dataset_download("fcpercival/160k-spotify-songs-sorted")

print("Path to song dataset files:", path)

# Download 960k lyrics dataset
path = kagglehub.dataset_download("bwandowando/spotify-songs-with-attributes-and-lyrics")

print("Path to lyrics dataset files:", path)
"""