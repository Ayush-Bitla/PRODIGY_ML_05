import os
import zipfile

# Download Food-101 from Kaggle using Kaggle API
DATASET = 'jayaprakashpondy/food-101-dataset'
OUTPUT_DIR = 'food-101'
ZIP_NAME = 'food-101-dataset.zip'

# Check if already downloaded
if os.path.exists(OUTPUT_DIR):
    print(f"{OUTPUT_DIR}/ already exists. Skipping download...")
else:
    print("Downloading Food-101 from Kaggle...")
    # Make sure Kaggle API is installed and kaggle.json is set up
    # You can place kaggle.json in ~/.kaggle/ or set KAGGLE_CONFIG_DIR
    os.system(f"kaggle datasets download -d {DATASET} -p .")
    if os.path.exists(ZIP_NAME):
        print(f"Extracting {ZIP_NAME}...")
        with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Extraction complete.")
        os.remove(ZIP_NAME)
    else:
        print("Download failed or zip file not found.") 