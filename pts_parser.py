import pandas as pd
#import numpy as np

# path to the .pts file
file_path = 'face_raw_data/points_22/m-001/m-001-01.pts'

# parse a .pts file and extract points into a DataFrame
def parse_pts(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # find the start of the point data, marked by '{'
        start = lines.index('{\n') + 1
        # end of the point data is marked by '}'
        end = lines.index('}\n', start)
        # extract points between the start and end
        points_data = [line.strip().split() for line in lines[start:end]]
        # create DataFrame
        df = pd.DataFrame(points_data, dtype=float, columns=['x', 'y'])
    return df


data = parse_pts(file_path)

print(data)