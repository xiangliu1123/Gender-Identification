import ast
import pandas as pd

# Load the dataset to examine its structure
file_path = 'facial_features.csv'
data = pd.read_csv(file_path)

# Convert the string representation of dictionaries in the 'Features' column to actual dictionaries
data['Features'] = data['Features'].apply(ast.literal_eval)

# Create new columns for each key in the dictionary of the 'Features' column
features_df = data['Features'].apply(pd.Series)

# Create a new column 'extra' with the first letter of each row from the 'File' column
features_df['Gender'] = data['File'].str[0]

# Output the complete DataFrame to a CSV file
# Note: Storing lists of tuples in CSV may require using a string format
features_df.to_csv('processed_data.csv', index=False)
