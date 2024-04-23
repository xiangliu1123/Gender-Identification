import pandas as pd
import numpy as np
import os
import ast
import zipfile


def extract_and_list_zip_contents(zip_file_path, extraction_directory):
    try:
        # Create a directory for extraction if it does not exist
        if not os.path.exists(extraction_directory):
            os.makedirs(extraction_directory)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # # List contents of the ZIP file
            # zip_contents = zip_ref.namelist()
            # print("Contents of the ZIP file:")
            # for item in zip_contents:
            #     print(item)
            zip_ref.extractall(extraction_directory)
            print(f"Successfully extracted the ZIP file to '{extraction_directory}'.")

    except zipfile.BadZipFile:
        print("Error: The file is not a zip file or it is corrupted.")

    except Exception as e:
        print(f"An error occurred: {e}")


def read_pts(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the start and end of the points block
    start = lines.index('{\n') + 1
    end = lines.index('}\n', start)

    # Extract points and convert to list of tuples
    points = [tuple(map(float, line.strip().split())) for line in lines[start:end]]
    return points


def collect_points_data(directory_path, csv_directory_path):
    # Define target folders
    male_folders = [f'm-{i:03d}' for i in range(1, 77)]
    female_folders = [f'w-{i:03d}' for i in range(1, 61)]
    target_folders = male_folders + female_folders

    # Initialize an empty list to accumulate all points data
    all_points_data = []

    # Iterate through the directory
    for root, dirs, files in os.walk(directory_path):
        current_folder = os.path.basename(root)
        if current_folder in target_folders:
            for filename in files:
                if filename.endswith('.pts'):
                    file_path = os.path.join(root, filename)
                    points = read_pts(file_path)

                    # Create a dictionary for the file and folder with the point list
                    file_data = {
                        'File': filename,
                        'Folder': current_folder,
                        'Points': points
                    }

                    # Append the file data to the list
                    all_points_data.append(file_data)

    # Convert the list of dictionaries to a DataFrame
    df_points = pd.DataFrame(all_points_data)

    # Save the DataFrame to CSV
    if not os.path.exists(csv_directory_path):
        os.makedirs(csv_directory_path)

    csv_path = os.path.join(csv_directory_path, 'all_points_data.csv')
    df_points.to_csv(csv_path, index=False)
    print(f"Data exported to CSV at: {csv_path}")
    return df_points


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

    nose_tip_to_chin_dist = euclidean_distance(faceMarkUpPoints[14], faceMarkUpPoints[19])

    left_cheek_to_chin_dist = euclidean_distance(faceMarkUpPoints[20], faceMarkUpPoints[19])
    right_cheek_to_chin_dist = euclidean_distance(faceMarkUpPoints[21], faceMarkUpPoints[19])

    nose_tip_to_upper_lips = euclidean_distance(faceMarkUpPoints[14], faceMarkUpPoints[17])
    lower_lips_to_chin = euclidean_distance(faceMarkUpPoints[18], faceMarkUpPoints[19])

    end_of_eye_brow_to_end_eye_left = euclidean_distance(faceMarkUpPoints[7], faceMarkUpPoints[12])
    end_of_eye_brow_to_end_eye_right = euclidean_distance(faceMarkUpPoints[4], faceMarkUpPoints[9])

    features = {
        'Eye length ratio': max(eye_length_left, eye_length_right) / upper_face_dist_horizontal,  # 1
        'Eye distance ratio': eye_dist / upper_face_dist_horizontal,  # 2
        'Nose ratio': nose_dist / lower_face_dist_horizontal,  #3
        'Lip size ratio': lip_dist_horizontal / lip_dist_vertical,  #4
        'Lip length ratio': lip_dist_horizontal / lower_face_dist_horizontal,  #5
        'Eye-brow length ratio': max(eyebrow_length_left, eyebrow_length_right) / upper_face_dist_horizontal,  #6
        'Aggressive ratio left': dist_10_19 / lower_face_dist_horizontal,  #7

        # Extra
        'Jaw line ratio': nose_tip_to_chin_dist/ lower_face_dist_horizontal,
        'Jaw horizontal length ratio': lip_dist_horizontal / lower_face_dist_horizontal,
        'Nose tip to lips distance': nose_tip_to_upper_lips,
        'Lower lips to chin distance': lower_lips_to_chin,
        'Nose tip to chin': nose_tip_to_chin_dist,
        'Lip horizontal ratio': lip_dist_horizontal/nose_tip_to_chin_dist,


        # Unused
        # 'end_of_eye_brow_to_end_eye_left': end_of_eye_brow_to_end_eye_left,
        # 'end_of_eye_brow_to_end_eye_right': end_of_eye_brow_to_end_eye_right,
        # 'Brow to eye ratio': end_of_eye_brow_to_end_eye_right/end_of_eye_brow_to_end_eye_left,
        # 'Lower face ratio': nose_tip_to_chin_dist/lower_face_dist_horizontal,
        # 'Nose to mouse': nose_dist / lip_dist_horizontal,
    }

    return features


def main():
    extract_and_list_zip_contents("..\\project_material\\Face Markup AR Database.zip", "..\\zip_content")
    point_df = collect_points_data("..\\zip_content\\Face Markup AR Database\\points_22", "..\\all_csv")

    # Apply feature calculation for each row
    feature_df = pd.DataFrame()
    feature_df["File"] = point_df["File"]
    feature_df['Features'] = point_df['Points'].apply(calculate_features)

    csv_path = os.path.join("../all_csv", 'feature_data.csv')
    feature_df.to_csv(csv_path, index=False)
    print(f"Data exported to CSV at: {csv_path}")

    # Load the dataset to examine its structure
    file_path = '..\\all_csv\\feature_data.csv'
    feature_data = pd.read_csv(file_path)

    # Convert the string representation of dictionaries in the 'Features' column to actual dictionaries
    feature_data['Features'] = feature_data['Features'].apply(ast.literal_eval)

    preProcessed_df = feature_data['Features'].apply(pd.Series)
    preProcessed_df['Gender'] = feature_data['File'].str[0]

    csv_path = os.path.join("../all_csv", 'preProcess_df.csv')
    preProcessed_df.to_csv(csv_path, index=False)
    print(f"Data exported to CSV at: {csv_path}")


if __name__ == "__main__":
    main()
