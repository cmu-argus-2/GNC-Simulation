import os
import urllib.request

# Define the file and directory
file_name = 'de440.bsp'
directory = 'data/'
file_path = os.path.join(directory, file_name)
url = 'https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp'

# Check if the file exists
if not os.path.exists(file_path):
    print(f"{file_name} not found in {directory}. Downloading...")
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Download the file
    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded {file_name} to {directory}.")
    except Exception as e:
        print(f"Failed to download {file_name}: {e}")
else:
    print(f"{file_name} is already present in {directory}.")