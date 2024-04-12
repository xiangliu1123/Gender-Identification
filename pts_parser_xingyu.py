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


def euclidean_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def calculate_features(points):
    # Calculate distances needed for feature ratios
    eye_length_1 = euclidean_distance(points[9], points[10])
    eye_length_2 = euclidean_distance(points[11], points[12])
    distance_8_13 = euclidean_distance(points[8], points[13])
    eye_center_1 = ((points[4][0] + points[5][0]) / 2, (points[4][1] + points[5][1]) / 2)
    eye_center_2 = ((points[6][0] + points[7][0]) / 2, (points[6][1] + points[7][1]) / 2)
    eye_distance = euclidean_distance(eye_center_1, eye_center_2)
    nose_dist = euclidean_distance(points[15], points[16])
    base_nose_dist = euclidean_distance(points[20], points[21])
    lip_dist = euclidean_distance(points[2], points[3])
    base_lip_dist = euclidean_distance(points[17], points[18])
    aggression_dist = euclidean_distance(points[10], points[19])

    # Construct the list of feature ratios
    features = [
        max(eye_length_1, eye_length_2) / distance_8_13,  # Eye length ratio
        eye_distance / distance_8_13,  # Eye distance ratio
        nose_dist / base_nose_dist,  # Nose ratio
        lip_dist / base_lip_dist,  # Lip size ratio
        lip_dist / base_nose_dist,  # Lip length ratio
        max(eye_length_1, eye_length_2) / distance_8_13,  # Eye-brow length ratio
        aggression_dist / base_nose_dist  # Aggressive ratio
    ]

    return features


# Apply feature calculation for each row
df_points['Features'] = df_points['Points'].apply(calculate_features)

# Example output
print(df_points.head())

# Save to CSV
df_points.to_csv('facial_features.csv', index=False)
print("Feature extraction and CSV generation complete.")
