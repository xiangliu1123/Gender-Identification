import pandas as pd
import numpy as np
import os


def read_pts(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the start and end of the points block
    start = lines.index('{\n') + 1
    end = lines.index('}\n', start)

    # Extract points and convert to list of tuples
    points = [tuple(map(float, line.strip().split())) for line in lines[start:end]]
    return points


def euclidean_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def calculate_features(faceMarkUpPoints):
    # Calculate distances needed for feature ratios
    eye_length_left = euclidean_distance(faceMarkUpPoints[9], faceMarkUpPoints[10])
    eye_length_right = euclidean_distance(faceMarkUpPoints[11], faceMarkUpPoints[12])
    eye_dist = euclidean_distance(faceMarkUpPoints[10], faceMarkUpPoints[11])
    upper_face_dist_horizontal = euclidean_distance(faceMarkUpPoints[8], faceMarkUpPoints[13])

    nose_dist = euclidean_distance(faceMarkUpPoints[15], faceMarkUpPoints[16])
    lip_dist_horizontal = euclidean_distance(faceMarkUpPoints[2], faceMarkUpPoints[3])
    lip_dist_vertical = euclidean_distance(faceMarkUpPoints[17], faceMarkUpPoints[18])
    lower_face_dist_horizontal = euclidean_distance(faceMarkUpPoints[20], faceMarkUpPoints[21])

    eyebrow_length_left = euclidean_distance(faceMarkUpPoints[4], faceMarkUpPoints[5])
    eyebrow_length_right = euclidean_distance(faceMarkUpPoints[6], faceMarkUpPoints[7])

    dist_10_19 = euclidean_distance(faceMarkUpPoints[10], faceMarkUpPoints[19])

    features = {'Eye length ratio': max(eye_length_left, eye_length_right) / upper_face_dist_horizontal,
                'Eye distance ratio': eye_dist / upper_face_dist_horizontal,
                'Nose ratio': nose_dist / lower_face_dist_horizontal,
                'Lip size ratio': lip_dist_horizontal / lip_dist_vertical,
                'Lip length ratio': lip_dist_horizontal / lower_face_dist_horizontal,
                'Eye-brow length ratio': max(eyebrow_length_left, eyebrow_length_right) / upper_face_dist_horizontal,
                'Aggressive ratio': dist_10_19 / lower_face_dist_horizontal}

    return features


def main():
    directory_path = 'points_22'

    # Adjust the folder names to match the actual range provided
    male_folders = [f'm-{i:03d}' for i in range(1, 77)]
    female_folders = [f'w-{i:03d}' for i in range(1, 61)]
    target_folders = male_folders + female_folders

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

    # Apply feature calculation for each row
    df_points['Features'] = df_points['Points'].apply(calculate_features)

    # Example output
    print(df_points.head())

    # Save to CSV
    df_points.to_csv('facial_features.csv', index=False)
    print("Feature extraction and CSV generation complete.")


if __name__ == "__main__":
    main()
