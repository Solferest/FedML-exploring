import argparse
import os
import numpy as np
import wget
import zipfile
import logging
from google_drive_downloader import GoogleDriveDownloader as gdd
import random
import shutil

cwd = os.getcwd()
DATASET_URL='https://drive.google.com/uc?export=download&confirm=1qeQbob94err5Zd7DLoq3Q2RsAUbRH0cb'
def main(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    file_path = os.path.join(data_cache_dir, "Dataset.zip")
    logging.info(file_path)

    if not os.path.exists(file_path):
        wget.download(DATASET_URL, out=file_path)
    # print(file_path)
    if not os.path.exists(os.path.join(data_cache_dir, "Dataset")):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(data_cache_dir)


    Set up directories
    base_dir = data_cache_dir+"/Dataset/Dataset 1" #path/to/base/directory
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    classes = ["Bruce Force FTP-Patator", "class2", "class3"]  # replace with your actual class names

    for d in [train_dir, test_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Move files to train/test directories by class
    for c in classes:
        class_dir = os.path.join(base_dir, c)
        files = os.listdir(class_dir)
        n_train = int(len(files) * 0.8)  # adjust split ratio here

        for f in files[:n_train]:
            src = os.path.join(class_dir, f)
            dst = os.path.join(train_dir, c, f)
            if not os.path.exists(os.path.join(train_dir, c)):
                os.makedirs(os.path.join(train_dir, c))
            shutil.move(src, dst)

        for f in files[n_train:]:
            src = os.path.join(class_dir, f)
            dst = os.path.join(test_dir, c, f)
            if not os.path.exists(os.path.join(test_dir, c)):
                os.makedirs(os.path.join(test_dir, c))
            shutil.move(src, dst)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--output-folder",
    #     type=str,
    #     help="Where to store the downloaded data.",
    #     required=True,
    # )
    # args = parser.parse_args()
    # main=(args.data_cache_dir)
    data_cache_dir = "/home/etu/test/data"
    main(data_cache_dir)
