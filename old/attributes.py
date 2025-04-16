import pandas as pd

# Function to load the dataset
def load_data(file_path):
    """Loads the dataset into a pandas DataFrame"""
    return pd.read_csv(file_path)

# Function to determine the type of attribute for each column
def determine_attribute_type(df):
    """Determines the attribute type for each column in the DataFrame"""
    attribute_types = {}

    for column in df.columns:
        # Check if the column is of type int64 (Integer columns)
        if df[column].dtype == 'int64':
            # If it contains only 2 unique values (binary 0/1)
            if df[column].nunique() == 2:
                attribute_types[column] = 'Binary (0/1)'
            # If it contains more than 2 unique values (Discrete Integer)
            else:
                attribute_types[column] = 'Discrete Integer'
        # Check if the column is of type object (Categorical columns)
        elif df[column].dtype == 'object':
            attribute_types[column] = 'Categorical'
        # Check if the column is of type float64 (Numeric columns)
        elif df[column].dtype == 'float64':
            attribute_types[column] = 'Numeric'
        else:
            attribute_types[column] = 'Unknown'

    return attribute_types

# Function to print the min and max values for Discrete Integer and Numeric attributes
def print_min_max_values(df, attribute_types):
    """Prints the min and max values for Discrete Integer and Numeric columns"""
    for column, attr_type in attribute_types.items():
        print(f"{column}: {attr_type}")
        # If the column is Discrete Integer or Numeric, print min and max values
        if attr_type in ['Discrete Integer', 'Numeric']:
            min_val = df[column].min()
            max_val = df[column].max()
            print(f"  Min: {min_val}, Max: {max_val}")
            
# Main function to execute the workflow
def main(file_path):
    """Main function to load data, determine attribute types, and print results"""
    # Load the dataset
    df = load_data(file_path)
    
    # Determine the attribute types
    attribute_types = determine_attribute_type(df)
    
    # First, print the categories of attributes
    print("Attribute Types:")
    for column, attr_type in attribute_types.items():
        print(f"{column}: {attr_type}")
    
    # Then, print the min and max values for Discrete Integer and Numeric attributes
    print("\nMin and Max Values for Discrete Integer and Numeric attributes:")
    print_min_max_values(df, attribute_types)

# Execute the main function with the path to your dataset
if __name__ == "__main__":
    # Replace the path with the actual path to your dataset
    main("../alzheimers_disease_data.csv")
