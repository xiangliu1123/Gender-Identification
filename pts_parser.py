import pandas as pd
import numpy as np
import os

directory_path = 'points_22'

# Generate folder names
male_folders = [f'm-{i:03d}' for i in range(1, 76)]
female_folders = [f'w-{i:03d}' for i in range(1, 61)]
target_folders = male_folders + female_folders

def read_pts(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the start and end of the points block
    start = lines.index('{\n') + 1
    end = lines.index('}\n', start)

    # Extract points
    points = []
    for line in lines[start:end]:
        x, y = map(float, line.strip().split())
        points.append((x, y))

    # Convert to DataFrame
    points_df = pd.DataFrame(points, columns=['X', 'Y'])
    return points_df

# Initialize an empty DataFrame to accumulate all points
all_points_data = pd.DataFrame()

for root, dirs, files in os.walk(directory_path):
    current_folder = os.path.basename(root)
    if current_folder in target_folders:
        for filename in files:
            if filename.endswith('.pts'):
                file_path = os.path.join(root, filename)
                points_data = read_pts(file_path)

                # Add an identifier for the file and folder
                points_data['file'] = filename
                points_data['folder'] = current_folder

                # Append the points data from this file to the accumulated DataFrame
                all_points_data = all_points_data._append(points_data, ignore_index=True)

# Now all_points_data contains points from all .pts files in the targeted folders
print(all_points_data)

# Write the DataFrame to a CSV file
all_points_data.to_csv('all_points_data.csv', index=False)


# features' positioning
features = [
    ["Eye length ratio", [8, 13]],
    ["Eye distance ratio", [8, 13]],  # Eye distance ratio: distance between center of two eyes over distance between points 8 and 13, calculate the centers of the eyes first
    ["Nose ratio", [15, 16, 20, 21]],
    ["Lip size ratio", [2, 3, 17, 18]],
    ["Lip length ratio", [2, 3, 20, 21]],
    ["Eye-brow length ratio", [4, 5, 6, 7, 8, 13]],  # We include both sets of points for eyebrows
    ["Aggressive ratio", [10, 19, 20, 21]]
]


# Euclidean distance between two points in the dataframe
def euclidean_distance(df, idx1, idx2):
    point1 = df.iloc[idx1-1]
    point2 = df.iloc[idx2-1]
    return np.sqrt((point2['x'] - point1['x']) ** 2 + (point2['y'] - point1['y']) ** 2)

