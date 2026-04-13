import os
import urllib.request
import zipfile

# Replace this with the actual Kaggle/Drive link or kagglehub dataset URL
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/mohitsingh1804/plantvillage"
RAW_DATA_DIR = "data/raw/images"
ZIP_PATH = "data/raw/dataset.zip"

def download_and_extract_dataset():
    """
    Downloads the dataset from a placeholder URL and extracts it
    to the raw data directory for the Vision CNN to consume in ImageFolder format.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    if DATASET_URL == "https://placeholder-url.com/dataset.zip":
        print("[Notice] Using placeholder URL. Please update DATASET_URL in src/data_pipeline/download_dataset.py with your Kaggle URL.")
        # Create mock directories to prevent ImageFolder crashes while the placeholder is active
        for mock_class in ['healthy', 'diseased', 'stressed']:
            os.makedirs(os.path.join(RAW_DATA_DIR, mock_class), exist_ok=True)
            # Create a mock empty file just to satisfy loader validation (optional)
            with open(os.path.join(RAW_DATA_DIR, mock_class, 'placeholder.jpg'), 'wb') as f:
                f.write(b"") # Dummy bytes
        return
        
    print(f"Downloading from {DATASET_URL}...")
    try:
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
            
        print(f"Successfully extracted dataset to {RAW_DATA_DIR}")
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        
    finally:
        # Cleanup zip file
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)

if __name__ == "__main__":
    download_and_extract_dataset()
