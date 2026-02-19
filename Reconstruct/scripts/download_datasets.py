import os
import urllib.request
import zipfile
import argparse

def download_and_extract(url, extract_to):
    """
    Download a zip file from a URL and extract its contents to a specified directory.
    
    Args:
        url (str): The URL of the zip file to download.
        extract_to (str): The directory where the contents will be extracted.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Define the local filename
    local_filename = os.path.join(extract_to, os.path.basename(url))

    # Download the file
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, local_filename)

    # Extract the zip file
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        print(f"Extracting {local_filename}...")
        zip_ref.extractall(extract_to)
    # Remove the zip file after extraction
    os.remove(local_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract datasets.")
    parser.add_argument("--datasets", nargs='+', default=["TUM", "MipNeRF360", "StaticHikes"], 
                        help="List of datasets to download. Default is TUM, MipNeRF360, StaticHikes.")
    parser.add_argument("--out_dir", type=str, default="data", 
                        help="Output base directory for the datasets.")
    args = parser.parse_args()
    
    # Define the base URL and dataset names
    base_url = "https://repo-sam.inria.fr/nerphys/on-the-fly-nvs/datasets"    

    # Download and extract each dataset
    for dataset in args.datasets:
        download_and_extract(f"{base_url}/{dataset}.zip", args.out_dir)
    
    print("Done")