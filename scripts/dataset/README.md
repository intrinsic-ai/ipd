# Scripts to download and extract the dataset, cad models

## Option 1: Bash Script

Will download one specified dataset, and extract it. Then will download all cad models to `models` subfolder.

```bash
export ID=dataset_basket_0
export CAMERA=Basler-LR
export FOLDER=./datasets

bash scripts/dataset/get_dataset.sh $ID $CAMERA $FOLDER
```

## Option 2: Python Script

Has options to download one or more datasets to specified folder, option to extract (will not extract by default). Then will download all cad models to `models` subfolder.

- Python requirements in [`get_dataset.requirements.txt`](./get_dataset.requirements.txt).
    ```bash
    pip install -r scripts/dataset/get_dataset.requirements.txt
    ```

- To download and extract all datasets:
    ```bash
    python scripts/dataset/get_dataset.py --id ALL --camera ALL --folder ./datasets --extract
    ```

- To download and extract one dataset:
    ```bash
    python scripts/dataset/get_dataset.py --id dataset_basket_1  --camera Basler-LR --folder ./datasets --extract
    ```

- To download and extract all cameras for one dataset:
    ```bash
    python scripts/dataset/get_dataset.py --id dataset_basket_1  --camera ALL --folder ./datasets --extract
    ```

- To download and extract all datasets for one camera:
    ```bash
    python scripts/dataset/get_dataset.py --id ALL  --camera Basler-LR --folder ./datasets --extract
    ```

- Command line options:
    ```bash
    ./scripts/dataset/get_dataset.py

    options:
    -h, --help              show this help message and exit
    --camera                    {ALL, FLIR_polar, Photoneo, Basler-HR, Basler-LR}
                            Supply a camera id or or 'ALL' to download all cameras for the specified dataset
    --id                        {ALL, dataset_basket_0, dataset_basket_1,..., dataset_texturedbg_3}
                            Supply a dataset id or 'ALL' to download all datasets for specified camera
    --folder FOLDER         Folder to download/extract the datasets to
    --extract               Flag to extract the downloaded dataset(s)
    ```