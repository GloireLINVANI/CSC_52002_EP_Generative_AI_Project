import requests

# create downlaod directory
import os
os.makedirs("downloads", exist_ok=True)
os.chdir("downloads")

def download_file(url):
    filename = url.split("/")[-1]
    response = requests.get(url, stream=True)
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    print(f"Downloaded {filename}")

# URLs to download
urls = [
    "http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar",
    # "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "https://github.com/ndb796/Small-ImageNet-Validation-Dataset-1000-Classes/archive/refs/heads/main.zip"
]

for url in urls:
    download_file(url)

# after downlaoding the files, extract them
import os
import tarfile
import zipfile

def extract_file(file_path):
    if file_path.endswith(".tar"):
        with tarfile.open(file_path, "r") as file:
            file.extractall()
    elif file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as file:
            file.extractall()
    print(f"Extracted {file_path}") 

# extract all files in the current directory
for file in os.listdir():
    extract_file(file)
    os.remove(file)  # remove the compressed file after extracting
