from .constants import CAD_FILES, DATASET_IDS, CAMERA_NAMES

import os
import urllib.request
import zipfile
from tqdm import tqdm
from typing import Optional, Union

import logging
logger = logging.getLogger(__name__)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url, to_dir: Union[str, os.PathLike]) -> Optional[Union[str, os.PathLike]]:
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    file_path = os.path.join(to_dir, url.split("/")[-1])
    if not os.path.exists(file_path):
        try:
            with DownloadProgressBar(unit='B', unit_scale=True,
                                miniters=1, desc=url.split('/')[-1]) as t:
                file_path, _ = urllib.request.urlretrieve(url, filename=file_path, reporthook=t.update_to)
        except Exception as e:
            os.remove(file_path)
            raise e
    else:
        logger.warn(f"{file_path} already exists")
        return
    return file_path

def extract(zip_path, to_dir: Union[str, os.PathLike]) -> None:
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=to_dir)
    print(f"Extracted {zip_path} to {to_dir}")

def download_cads(to_dir: Union[str, os.PathLike]) -> None:
    for cad_file in CAD_FILES:
        url = f"https://storage.googleapis.com/akasha-public/industrial_plenoptic_dataset/cad_models/{cad_file}"
        download(url, to_dir=f"{to_dir}/models")

def download_dataset(dataset_id : str, camera_name : str, to_dir : Union[str, os.PathLike]) -> Optional[Union[str, os.PathLike]]:
    assert dataset_id in DATASET_IDS, f"Invalid dataset id {dataset_id}, must be one of {DATASET_IDS}"
    assert camera_name in CAMERA_NAMES, f"Invalid camera name {camera_name}, must be one of {CAMERA_NAMES}"
    dataset_name = f'{camera_name}-{dataset_id}'
    url = f"https://storage.googleapis.com/akasha-public/industrial_plenoptic_dataset/{dataset_name}.zip"
    zip_path = download(url, to_dir=to_dir)
    return zip_path


class DisableLogger():
    def __enter__(self):
       logging.disable(logging.CRITICAL)
    def __exit__(self, exit_type, exit_value, exit_traceback):
       logging.disable(logging.NOTSET)