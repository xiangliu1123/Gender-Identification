import pandas as pd
import numpy as np
import os

directory_path = 'points_22'

# Adjust the folder names to match the actual range provided
male_folders = [f'm-{i:03d}' for i in range(1, 77)]
female_folders = [f'w-{i:03d}' for i in range(1, 61)]
target_folders = male_folders + female_folders

def read_pts(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the start and end of the points block
    start = lines.index('{\n') + 1
    end = lines.index('}\n', start)

    # Extract points and convert to list of tuples
    points = [tuple(map(float, line.strip().split())) for line in lines[start:end]]
    return points

# Initialize an empty list to accumulate all points data
all_points_data = []

for root, dirs, files in os.walk(directory_path):
    current_folder = os.path.basename(root)
    if current_folder in target_folders:
        for filename in files:
            if filename.endswith('.pts'):
                file_path = os.path.join(root, filename)
                points = read_pts(file_path)

                # Create a dictionary for the file and folder with the points list
                file_data = {
                    'File': filename,
                    'Folder': current_folder,
                    'Points': points
                }

                # Append the file data to the list
                all_points_data.append(file_data)

# Convert the list of dictionaries to a DataFrame
df_points = pd.DataFrame(all_points_data)

#print("df_points::", df_points['Points'])

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
    point1 = df[idx1]
    point2 = df[idx2]
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


for df_point in df_points['Points']:
    print("Euclidian dictance: ", euclidean_distance(df_point, 8, 13))



# Output the complete DataFrame to a CSV file
# Note: Storing lists of tuples in CSV may require using a string format
df_points.to_csv('all_points_data.csv', index=False)
print("Data extraction complete. CSV file created.")
