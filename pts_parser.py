import pandas as pd
import os

directory_path = 'points_22'

# Adjust the folder names to match the actual range provided
male_folders = [f'm-{i:03d}' for i in range(1, 76)]
female_folders = [f'w-{i:03d}' for i in range(1, 61)]
target_folders = male_folders + female_folders


def read_pts(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the start and end of the points block
    start = lines.index('{\n') + 1
    end = lines.index('}\n', start)

    # Extract points and convert to DataFrame
    points = [tuple(map(float, line.strip().split())) for line in lines[start:end]]
    points_df = pd.DataFrame(points, columns=['X', 'Y'])

    return points_df


# Initialize an empty DataFrame to accumulate all points data
all_points_data = pd.DataFrame()

for root, dirs, files in os.walk(directory_path):
    current_folder = os.path.basename(root)
    if current_folder in target_folders:
        for filename in files:
            if filename.endswith('.pts'):
                file_path = os.path.join(root, filename)
                points_data = read_pts(file_path)

                # Adding an identifier for the file and folder
                points_data['File'] = filename
                points_data['Folder'] = current_folder
                points_data['Point_Index'] = points_data.index  # Add a point index starting from 1

                # Append the points data from this file to the accumulated DataFrame
                all_points_data = all_points_data._append(points_data, ignore_index=True)

# Output the complete DataFrame to a CSV file
all_points_data.to_csv('all_points_data.csv', index=False)
print("Data extraction complete. CSV file created.")
