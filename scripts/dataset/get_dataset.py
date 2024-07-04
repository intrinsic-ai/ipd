import os
import urllib.request
import zipfile
import argparse
from tqdm import tqdm

dataset_ids = [
    "dataset_basket_0",
    "dataset_basket_1",
    "dataset_basket_2",
    "dataset_basket_3",
    "dataset_basket_4",
    "dataset_basket_5",
    "dataset_basket_6",
    "dataset_basket_7",
    "dataset_basket_8",
    "dataset_basket_9",
    "dataset_darkbg_0",
    "dataset_darkbg_1",
    "dataset_darkbg_2",
    "dataset_darkbg_3",
    "dataset_darkbg_4",
    "dataset_darkbg_5",
    "dataset_darkbg_6",
    "dataset_darkbg_7",
    "dataset_darkbg_8",
    "dataset_texturedbg_0",
    "dataset_texturedbg_1",
    "dataset_texturedbg_2",
    "dataset_texturedbg_3"
]

cad_files = [
    "corner_bracket.stl",
    "corner_bracket0.stl",
    "corner_bracket1.stl",
    "corner_bracket2.stl",
    "corner_bracket3.stl",
    "corner_bracket4.stl",
    "corner_bracket6.stl",
    "gear1.stl",
    "gear2.stl",
    "handrail_bracket.stl",
    "hex_manifold.stl",
    "l_bracket.stl",
    "oblong_float.stl",
    "pegboard_basket.stl",
    "pipe_fitting_unthreaded.stl",
    "single_pinch_clamp.stl",
    "square_bracket.stl",
    "t_bracket.stl",
    "u_bolt.stl",
    "wraparound_bracket.stl",
]

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url, to_dir="./datasets"):
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
        print(f"{file_path} already exists")
    return file_path

def extract(zip_path, to_dir="./datasets"):
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=to_dir)
    print(f"Extracted {zip_path} to {to_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["ALL", "FLIR_polar", "Photoneo", "Basler-HR", "Basler-LR"], default="Basler-LR", help="Supply a camera id or or 'ALL' to download all cameras for the specified dataset")
    parser.add_argument("--id", choices=["ALL"] + dataset_ids, default="dataset_basket_0", help="Supply a dataset id or 'ALL' to download all datasets for specified camera")
    parser.add_argument("--folder", default="./datasets", help="Folder to download/extract the datasets to")
    parser.add_argument("--extract", action="store_true", default=False, help="Flag to extract the downloaded dataset(s)")
    args = parser.parse_args()
    if args.camera == "ALL":
        cameras = ["FLIR_polar", "Photoneo", "Basler-HR", "Basler-LR"]
    else:
        cameras = [args.camera]

    if args.id == "ALL":
        selected_datasets = dataset_ids
    else:
        selected_datasets = [args.id]
    
    

    #download the datasets
    print(f"Downloading datasets to {args.folder}")
    for camera in cameras:
        for dataset in selected_datasets:
            dataset_name = f'{camera}-{dataset}'
            url = f"https://storage.googleapis.com/akasha-public/industrial_plenoptic_dataset/{dataset_name}.zip"
            zip_path = download(url, to_dir=args.folder)
            if args.extract:
                extract(zip_path)
    
    #download the cad_models
    print(f"Downloading cad models to {args.folder}/models")
    for cad_file in cad_files:
        url = f"https://storage.googleapis.com/akasha-public/industrial_plenoptic_dataset/cad_models/{cad_file}"
        download(url, to_dir=f"{args.folder}/models")

if __name__ == "__main__":
    main()

