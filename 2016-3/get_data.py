import os
import requests
import zipfile
import pandas as pd
import concurrent.futures
from tqdm import tqdm

# Create directories
os.makedirs('melanoma_dataset', exist_ok=True)
os.makedirs('melanoma_dataset/raw', exist_ok=True)

# URLs for the dataset
urls = {
    'train_data': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_Data.zip',
    'train_gt': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Training_GroundTruth.csv',
    'test_data': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_Data.zip',
    'test_gt': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3_Test_GroundTruth.csv'
}


# Download function with progress bar
def download_file(name, url):
    destination = os.path.join('melanoma_dataset/raw', os.path.basename(url))
    print(f"Downloading {os.path.basename(destination)}...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

    # Extract if it's a zip file
    if destination.endswith('.zip'):
        print(f"Extracting {os.path.basename(destination)}...")
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            extract_dir = destination.replace('.zip', '')
            os.makedirs(extract_dir, exist_ok=True)
            zip_ref.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")

    return name, destination


# Download files concurrently
file_paths = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    # Submit all download tasks
    future_to_url = {executor.submit(download_file, name, url): name for name, url in urls.items()}

    # Process results as they complete
    for future in concurrent.futures.as_completed(future_to_url):
        name, path = future.result()
        file_paths[name] = path

print("All files downloaded and extracted successfully!")