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


# 1. Euclidean distance between two points in the dataframe
def euclidean_distance(df, idx1, idx2):
    point1 = df[idx1]
    point2 = df[idx2]
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# 1. Eye length ratio


def get_ratio(df, feature_num):
    dist_1 = euclidean_distance(df, features[feature_num][1][0], features[feature_num][1][1])
    dist_2 = euclidean_distance(df, features[feature_num][1][2], features[feature_num][1][3])
    return dist_1 / dist_2

# 3. get nose ratio, distance between points  15 and 16 over distance between 20 and 21
def nose_ratio(df):
    return get_ratio(df, 2)

# 4. Lip size ratio
def lip_size_ratio(df):
    return get_ratio(df, 3)

# 5. Lip length ratio
def lip_length_ratio(df):
    return get_ratio(df, 4)


# 6. Eye-brow length
def eye_brow_length(df):
    dist_1 = euclidean_distance(df, features[5][1][0], features[5][1][1])
    dist_2 = euclidean_distance(df, features[5][1][2], features[5][1][3])
    dist_3 = euclidean_distance(df, features[5][1][2], features[5][1][3])
    if dist_1 > dist_2:
        return dist_1 / dist_3
    else:
        return  dist_2 / dist_3

# 7. Aggressive ratio
def aggressive_ratio(df):
    return get_ratio(df, 6)


eye_distances = []
nose_ratios = []
lip_size_ratios = []
lip_length_ratios = []
eye_brow_lengths = []
aggressive_ratios = []

for df_point in df_points['Points']:
    eye_distances.append(euclidean_distance(df_point, 8, 13))
    nose_ratios.append(nose_ratio(df_point))
    lip_size_ratios.append(lip_size_ratio(df_point))
    lip_length_ratios.append(lip_length_ratio(df_point))
    eye_brow_lengths.append(eye_brow_length(df_point))
    aggressive_ratios.append(aggressive_ratio(df_point))


print("1. Eye length ratio: ...") # print data for eye length ratio
print("2. Eye distance ratio: ...") # print data for eye distance ratio
print("3. Nose ratios: ", nose_ratios)
print("4. Lip size ratios", lip_size_ratios)
print("5. Lip length ratios", lip_length_ratios)
print("6. Eye-brow length", eye_brow_lengths)
print("7. Aggressive ratio", aggressive_ratios)



# Output the complete DataFrame to a CSV file
# Note: Storing lists of tuples in CSV may require using a string format
df_points.to_csv('all_points_data.csv', index=False)
print("Data extraction complete. CSV file created.")
