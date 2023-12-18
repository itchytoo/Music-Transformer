import requests
import zipfile
import tarfile
import os
import argparse

def download_and_unzip(url, destination_folder):
    # Check if the destination folder exists, create if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Extract the filename from the URL
    filename = url.split('/')[-1]
    filepath = os.path.join(destination_folder, filename)

    # Download the file
    print(f"Downloading {filename}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print("Download completed.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
        return

    # Unzipping the file
    if filename.endswith('.zip'):
        print("Unzipping the file...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        print("Unzipping completed.")

    elif filename.endswith('.tar.gz'):
        print("Untarring the file...")
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(destination_folder)
        print("Untarring completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and unzip dataset.')
    parser.add_argument('--dataset', type=str, choices=['maestro', 'lakh'], default='maestro',
                        help='Specify the dataset to download')
    args = parser.parse_args()

    if args.dataset == 'maestro':
        download_and_unzip(
            url='https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip',
            destination_folder='data'
        )

    elif args.dataset == 'lakh':
        # get midi files
        download_and_unzip(
            url='http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz',
            destination_folder='data'
        )
        # get genre labels 
        download_and_unzip(
            url='https://www.tagtraum.com/genres/msd_tagtraum_cd1.cls.zip',
            destination_folder='data'
        )