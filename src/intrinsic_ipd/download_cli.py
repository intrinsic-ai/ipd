from .constants import DATASET_IDS, CAMERA_NAMES

from .utils import download_dataset, download_cads, extract
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["ALL"] + CAMERA_NAMES, default="Basler-LR", help="Supply a camera id or or 'ALL' to download all cameras for the specified dataset")
    parser.add_argument("--id", choices=["ALL"] + DATASET_IDS, default="dataset_basket_0", help="Supply a dataset id or 'ALL' to download all datasets for specified camera")
    parser.add_argument("--root", default="./datasets", help="Folder to download/extract the datasets to")
    parser.add_argument("--extract", action="store_true", default=False, help="Flag to extract the downloaded dataset(s)")
    args = parser.parse_args()
    if args.camera == "ALL":
        cameras = CAMERA_NAMES
    else:
        cameras = [args.camera]

    if args.id == "ALL":
        selected_datasets = DATASET_IDS
    else:
        selected_datasets = [args.id]
    
    #download the datasets
    print(f"Downloading datasets to {args.root}")
    for camera in cameras:
        for dataset in selected_datasets:
            zip_path = download_dataset(dataset, camera, args.root)
            if args.extract and zip_path:
                extract(zip_path, args.root)
    
    #download the cad_models
    print(f"Downloading cad models to {args.root}/models")
    download_cads(args.root)
    
if __name__ == "__main__":
    main()