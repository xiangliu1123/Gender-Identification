import pandas as pd
import numpy as np
from ast import literal_eval

# Parsing the points column string into a list of (x, y) tuples
def parse_points(points_str):
    return literal_eval(points_str)

# Calculating the Euclidean distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Calculating each of the requested feature ratios
def eye_length_ratio(points):
    eye_points = [points[i] for i in [9, 0, 10, 11, 1, 12]]
    eye_length = sum(distance(eye_points[i], eye_points[i + 1]) for i in range(len(eye_points) - 1))
    dist_8_13 = distance(points[8], points[13])
    return eye_length / dist_8_13

def eye_distance_ratio(points):
    dist_10_11 = distance(points[10], points[11])
    dist_8_13 = distance(points[8], points[13])
    return dist_10_11 / dist_8_13

def nose_ratio(points):
    dist_15_16 = distance(points[15], points[16])
    dist_20_21 = distance(points[20], points[21])
    return dist_15_16 / dist_20_21

def lip_size_ratio(points):
    dist_2_3 = distance(points[2], points[3])
    dist_17_18 = distance(points[17], points[18])
    return dist_2_3 / dist_17_18

def lip_length_ratio(points):
    dist_2_3 = distance(points[2], points[3])
    dist_20_21 = distance(points[20], points[21])
    return dist_2_3 / dist_20_21

def eye_brow_length_ratio(points):
    dist_4_5 = distance(points[4], points[5])
    dist_6_7 = distance(points[6], points[7])
    dist_8_13 = distance(points[8], points[13])
    return max(dist_4_5, dist_6_7) / dist_8_13

def aggressive_ratio(points):
    dist_10_19 = distance(points[10], points[19])
    dist_20_21 = distance(points[20], points[21])
    return dist_10_19 / dist_20_21

# Load the CSV file
file_path = 'all_points_data.csv'
data = pd.read_csv(file_path)

# Apply the functions to the first row as an example
data['Points'] = data['Points'].apply(parse_points)
first_row_points = data['Points'][0]

example_ratios = {
    "Eye Length Ratio": eye_length_ratio(first_row_points),
    "Eye Distance Ratio": eye_distance_ratio(first_row_points),
    "Nose Ratio": nose_ratio(first_row_points),
    "Lip Size Ratio": lip_size_ratio(first_row_points),
    "Lip Length Ratio": lip_length_ratio(first_row_points),
    "Eye-brow Length Ratio": eye_brow_length_ratio(first_row_points),
    "Aggressive Ratio": aggressive_ratio(first_row_points)
}

print(example_ratios)
