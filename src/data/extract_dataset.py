import tarfile
from pathlib import Path

archive_path = Path("data/raw/tracing-data.tar.gz")
extract_path=Path("data/raw")
dataset_folder=Path("data/raw/tracing-data")

if  dataset_folder.is_dir():
  print(f"{dataset_folder} allready exists")
else:
  print(f"{dataset_folder} is downloading")

with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=extract_path)
    print("Dataset extracted successfully.")