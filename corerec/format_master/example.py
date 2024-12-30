import pandas as pd
import logging
import yaml  # Import PyYAML

from cr_formatMaster import FormatMaster  # Ensure you import FormatMaster

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize the FormatMaster
format_master = FormatMaster()

# Load the sample dataset
path = 'corerec/format_master/1.yaml'  # Adjust the path as necessary

# Load the YAML file into a DataFrame
with open(path, 'r') as file:
    data = yaml.safe_load(file)  # Load the YAML data
    df = pd.DataFrame(data)  # Convert the loaded data into a DataFrame

# Log the DataFrame columns
logging.info(f"DataFrame columns: {df.columns.tolist()}")

# Detect the format
data_format = format_master.detect(df)  # Pass the DataFrame to the detect method

# Print the detected format
print(f"Detected data format: {data_format}")